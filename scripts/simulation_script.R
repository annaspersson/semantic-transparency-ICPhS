# LIBRARIES AND HELPERS----
library(tidyverse)  # data wrangling and plotting
library(magrittr)   # for all pipes
library(ggplot2)    # for plotting
library(patchwork)  # combining plots
library(broom)      # for regression analysis
library(ggdist)     # for plotting
library(ggside) # for plotting densities on y-axis
library(furrr)  # for parallelizing loop in grid search
library(magick) # for stitching a GIF together
library(here) # relative pathing

# HELPER FUNCTIONS CALLED IN SIMULATION LOOP
clamp01 <- function(x) pmax(0, pmin(1, x))
clamp02 <- function(x) pmax(0, pmin(0.95, x))

normalize_01 <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# At signal production, amount of noise is dependent on guess rate AND distance to attractor centres
# k controls how strongly sd react to speaker_guess; mindful that if p = 1 and k >2, it will return negative sd's
produce_signal <- function(stored_signal, speaker_guess, drift_sd, k_production, 
                           attractor_centers, circle_radius, center_sd, k_attractor_production) {
  # Calculate baseline trial-specific SD based on speaker's guess
  baseline_sd <- drift_sd * (1 + k_production * (0.5 - speaker_guess))
  # Dynamically calculate Euclidean distance to every defined attractor center
  distances <- sapply(attractor_centers, function(center) sqrt(sum((stored_signal - center)^2)))
  dist_to_nearest <- min(distances)
  nearest_id <- which.min(distances)
  # Find if the signal is inside any attractor, and locate the closest one
  inside_attractor <- distances < circle_radius
  is_inside <- any(inside_attractor)
  # noise f(dist) is calibrated against the worst-case (most inflated) baseline_sd (for when speaker_guess = 0), so the attractor pull is guaranteed strong enough;
  # keeping this noise a pure function of distance -- independent of guess on any given trial.
  max_possible_baseline_sd <- drift_sd * (1 + k_production * 0.5)
  f_dist_max <- max_possible_baseline_sd - center_sd
  attractor_reduction <- 0
  
  if (is_inside) {
    # Target the closest attractor that the signal is currently inside
    closest_attractor_dist <- min(distances[inside_attractor])
    rel_dist <- closest_attractor_dist / circle_radius
    # scale distance dynamically; as with signal evidence
    magnitude <- exp(-k_attractor_production * rel_dist)
    max_mag <- exp(-k_attractor_production * 0)
    min_mag <- exp(-k_attractor_production * 1)
    exp_scale <- (magnitude - min_mag) / (max_mag - min_mag)
    # subtract for fixed base sd, keeping this independent from the noise dep on associative strength
    attractor_reduction <- f_dist_max * exp_scale
  } 
  # floor at center_sd so that noise never drops below the attractor's minimum, regardless of speaker_guess. 
  # Guess-dependence remains fully intact everywhere except at the exact center where center_sd is by definition
  # the tightest the noise can ever get.
  #final_sd <- max(baseline_sd - attractor_reduction, center_sd, 0.001)
  final_sd <- max(baseline_sd - attractor_reduction, 0.001)
  
  # Generate the signal using the final calculated SD
  signal <- rnorm(
    length(stored_signal),
    mean = stored_signal,
    sd = final_sd)
  
  # Return both the produced signal and the boolean status flag
  return(list(signal = clamp01(signal), 
              inside_attractor = is_inside,
              dist_to_nearest = dist_to_nearest,
              attractor_id = if (is_inside) nearest_id else NA_integer_))
}

# Signal evidence for iconicity bias
# Measures proximity of Y to its size prototype
signal_evidence <- function(produced_signal, center_target, k_perception, circle_radius) {
  # Calculate eucledian distance to target_center and to the opposite attractor
  dist_to_target <- sqrt(sum((produced_signal - center_target)^2))
  opposite_center <- c(1 - center_target[1], 1 - center_target[2])
  dist_to_opposite <- sqrt(sum((produced_signal - opposite_center)^2))
  # If signal is inside the circle, calculate evidence
  if (dist_to_target < circle_radius) {
    # Calculate distance to targer center from edge of attractor, independent of circle size; 0 at center, 1 at edge
    rel_dist <- dist_to_target / circle_radius 
    # as relative dist grows, magnitude shrinks as a function of k_perception
    magnitude <- exp(-k_perception * rel_dist) 
    # Simple normalization: center = 1, edge = 0
    max_mag <- exp(-k_perception * 0)
    min_mag <- exp(-k_perception * 1) 
    evidence <- (magnitude - min_mag) / (max_mag - min_mag)
    return(evidence)
    # Punish anti-iconic behavior: if signal instead is inside the opposite attractor, return negative evidence of the same magnitude
  } else if (dist_to_opposite < circle_radius) {
    rel_dist <- dist_to_opposite / circle_radius
    magnitude <- exp(-k_perception * rel_dist)
    max_mag <- exp(-k_perception * 0)
    min_mag <- exp(-k_perception * 1)
    
    evidence <- (magnitude - min_mag) / (max_mag - min_mag)
    return(-evidence)
    
  } else {
    return(0)
  }
}

# LISTENER RECOGNITION PROBABILITY UPDATED
listener_guess_probability <- function(listener_guess, produced_signal, size_prototypes,
                                       recognition_bias, iconicity_weight, k_perception, circle_radius) {
  icon_ev <- signal_evidence(produced_signal, size_prototypes, k_perception = k_perception, circle_radius = circle_radius)
  effective_weight <- if (recognition_bias) iconicity_weight else 0
  
  # qlogis() is undefined at exactly 0 or 1 (-Inf/Inf), we clamp listener_guess minimally from those exact bounds before converting to 
  # logits (listener_guess should already be within (0, 0.95) after update_logit() which applies 0.95 lapse-rate clamp, so this is 
  # purely a numerical safety net, not where the lapse rate is enforced).
  logit_guess <- qlogis(pmax(1e-9, pmin(1 - 1e-9, listener_guess)))
  logits <- logit_guess + (effective_weight * icon_ev)
  probs <- clamp02(plogis(logits))

  return(list(probs = probs, evidence = icon_ev))
}

# Update learning as dependent on success
update_logit <- function(x, learning_strength, success, success_scale, failure_scale) {
  delta <- learning_strength * ifelse(success == 1, success_scale, failure_scale)
  plogis(qlogis(clamp02(x)) + delta)
}


# MAIN SIM FUNCTION----
run_interaction_sim <- function(
    data,
    n_sim = 1,
    n_referents = 4,
    n_generations = 1,
    n_rounds = 300,
    # motor/production noise; equivalent to approx. 43% chance of wandering into any attractor in a single production step
    # when the signal is at .5, .5 and speaker_guess ~ 0.3
    drift_sd = 0.2,
    k_attractor_production = 2.5,
    neutral_attractor_centers = list(c(0.15, 0.15), c(0.85, 0.85)),
    # set as a plausible basin size relative to the unit signal space (not independently calibrated ag a spec target)
    circle_radius = 0.3,
    # sd at attractor centers; a signal at the dead centre has ~2% single-step escape probability from the attractor
    # (ratio of 0.4 of circle_radius)
    trap_center_sd = 0.12,
    k_perception = 2.5,
    recognition_bias = FALSE,
    # multiplicator for iconicity; corresponding to ~10% absolute increase for a listener_guess of 0.5 
    # (for the perfectly iconic signal; icon_ev = 1)
    iconicity_weight = 3,
    # amount of added memory strengthening per exposure, dependent on trial success/failure
    learning_strength = 0.015,
    # substantially faster learning for success than failure; from p=0.3, 10 pure-success updates reach ~0.57,
    # 10 pure-failure reach ~0.33; no real referent is reinforced purely by success/failure since updates depend on
    # current associative strength
    success_scale = 7.5,
    failure_scale = 1,
    expressive_agents = TRUE,
    # expressive productions land inside the target attractor w very high probability
    expressive_noise_sd = circle_radius / 4.8,
    # --- two-level hierarchical expressiveness (kept for generational-overturn work) ---
    # expressive_prob_per_agent = 0.10, # per-generation probability an agent is expressive
    # expressive_trial_prob = 0.20,     # per-trial probability of override, conditional on being that type
    # --- simplified single-parameter version ---
    # flat per-trial probability of an expressive override, no agent-level persistence; 1 expressive signal in 100 interactions
    expressive_probability = 0.01
) {
  
  referents_blueprint <- tibble(
    id = seq_len(n_referents),
    type = rep(c("small", "large"), length.out = n_referents),
    size_prototypes = if_else(type == "small",
                              list(c(0.15, 0.85)),
                              list(c(0.85, 0.15))))
  
  semantic_attractor_centers <- unique(referents_blueprint$size_prototypes)
  attractor_centers <- c(semantic_attractor_centers, neutral_attractor_centers)
  is_semantic_attractor <- c(
    rep(TRUE, length(semantic_attractor_centers)),
    rep(FALSE, length(neutral_attractor_centers))
  )
  
  # --- OLD: probability of expressive speakers (agent-level trait) ---
  # expressive_prob <- if (expressive_agents) expressive_prob_per_agent else 0
  # --- NEW: flat per-trial probability, no agent-level draw needed ---
  trial_expressive_prob <- if (expressive_agents) expressive_probability else 0
  
  simulation_log <- list()
  
  for (sim in 1:n_sim) {
    
    referents_info <- referents_blueprint %>%
      mutate(
        agentA_stored_signal = rep(list(c(0.5, 0.5)), n_referents),
        agentB_stored_signal = rep(list(c(0.5, 0.5)), n_referents))
    
    for (gen in 1:n_generations) {
      trial_counter <- 0
      
      # --- agent-level expressive assignment, drawn once per generation ---
      # expressive_A <- runif(1) < expressive_prob
      # expressive_B <- runif(1) < expressive_prob
      
      agentA_guess <- rbeta(n_referents, 3, 9)
      agentB_guess <- rbeta(n_referents, 3, 9)
      
      for (round in 1:n_rounds) {
        referent_order <- sample(1:n_referents)
        roles <- sample(rep(c("A", "B"), length.out = n_referents))
        
        for (trial in 1:n_referents) {
          trial_counter <- trial_counter + 1
          ref_id <- referent_order[trial]
          speaker <- roles[trial]
          listener <- ifelse(speaker == "A", "B", "A")
          
          if (speaker == "A") {
            speaker_guess <- agentA_guess
            listener_guess <- agentB_guess
          } else {
            speaker_guess <- agentB_guess
            listener_guess <- agentA_guess
          }
          
          old_guess_A <- agentA_guess[ref_id]
          old_guess_B <- agentB_guess[ref_id]
          old_stored_signal_A <- referents_info$agentA_stored_signal[[ref_id]]
          old_stored_signal_B <- referents_info$agentB_stored_signal[[ref_id]]
          old_stored_signal <- if (speaker == "A") old_stored_signal_A else old_stored_signal_B
          
          production_output <- produce_signal(
            stored_signal = old_stored_signal,
            speaker_guess = speaker_guess[ref_id],
            drift_sd = drift_sd,
            k_production = 1.5,
            attractor_centers = attractor_centers,
            circle_radius = circle_radius,
            center_sd = trap_center_sd,
            k_attractor_production = k_attractor_production)
          
          is_expressive_trial <- FALSE
          
          # --- override only if speaker is the pre-assigned expressive agent for this generation ---
          # if (
          #   ((speaker == "A" && expressive_A) ||
          #    (speaker == "B" && expressive_B)) &&
          #   runif(1) < expressive_trial_prob
          # ) {
          
          # --- flat per-trial draw, no agent identity involved ---
          if (runif(1) < trial_expressive_prob) {
            is_expressive_trial <- TRUE
            target_center <- referents_info$size_prototypes[[ref_id]]
            signal <- clamp01(rnorm(2, mean = target_center, sd = expressive_noise_sd))
            
            distances <- sapply(attractor_centers, function(center) sqrt(sum((signal - center)^2)))
            dist_to_nearest <- min(distances)
            nearest_id <- which.min(distances)
            in_attractor <- dist_to_nearest < circle_radius
            attractor_id <- if (in_attractor) nearest_id else NA_integer_
          } else {
            signal <- production_output$signal
            in_attractor <- production_output$inside_attractor
            dist_to_nearest <- production_output$dist_to_nearest
            attractor_id    <- production_output$attractor_id
          }
          
          recognition <- listener_guess_probability(
            listener_guess[ref_id],
            signal,
            referents_info$size_prototypes[[ref_id]],
            recognition_bias = recognition_bias,
            iconicity_weight = iconicity_weight,
            k_perception = k_perception,
            circle_radius = circle_radius)
          
          prob <- recognition$probs
          success <- rbinom(1, 1, prob)
          
          if (listener == "A") {
            agentA_guess[ref_id] <- update_logit(prob, learning_strength, success, success_scale, failure_scale)
          } else {
            agentB_guess[ref_id] <- update_logit(prob, learning_strength, success, success_scale, failure_scale)
          }
          
          if (success == 1) {
            referents_info$agentA_stored_signal[[ref_id]] <- 
              (signal + referents_info$agentA_stored_signal[[ref_id]]) / 2
            referents_info$agentB_stored_signal[[ref_id]] <- 
              (signal + referents_info$agentB_stored_signal[[ref_id]]) / 2
          }
          
          new_guess_A <- agentA_guess[ref_id]
          new_guess_B <- agentB_guess[ref_id]
          new_stored_signal_A <- referents_info$agentA_stored_signal[[ref_id]]
          new_stored_signal_B <- referents_info$agentB_stored_signal[[ref_id]]
          
          has_semantic <- !is.na(attractor_id) && is_semantic_attractor[attractor_id]
          log_is_semantic <- if (is.na(attractor_id)) FALSE else is_semantic_attractor[attractor_id]
          is_correct_semantic_attractor <- if (has_semantic) {
            identical(attractor_centers[[attractor_id]], referents_info$size_prototypes[[ref_id]])
          } else {
            FALSE
          }
          
          simulation_log[[length(simulation_log) + 1]] <- tibble(
            simulation = sim, generation = gen, round = round, trial = trial, trial_counter = trial_counter,
            referent = ref_id, speaker = speaker, listener = listener, type = referents_info$type[ref_id],
            produced_signal = list(signal), dist_to_nearest = dist_to_nearest, in_attractor = in_attractor,
            attractor_id = attractor_id, is_semantic_attractor = log_is_semantic,
            is_correct_semantic_attractor = is_correct_semantic_attractor,
            prob = prob, evidence = recognition$evidence, success = success,
            # --- agent-level expressive flags, no longer meaningful under flat design ---
            # expressive_A = expressive_A, expressive_B = expressive_B,
            is_expressive_trial = is_expressive_trial,
            old_guess_A = old_guess_A, new_guess_A = new_guess_A,
            old_guess_B = old_guess_B, new_guess_B = new_guess_B,
            old_stored_signal_A = list(old_stored_signal_A), old_stored_signal_B = list(old_stored_signal_B),
            new_stored_signal_A = list(new_stored_signal_A), new_stored_signal_B = list(new_stored_signal_B)
          )
        }
      }
    }
  }
  full_history <- bind_rows(simulation_log)
  return(full_history)
}

# Call it
d.empty <- data.frame(
  sim = integer(), gen = integer(), round = integer(), trial = integer(),
  trial_counter = integer(), referent = integer(), 
  speaker = character(), listener = character(), type = character(),
  produced_signal = I(list()), dist_to_nearest = numeric(), in_attractor = logical(), 
  attractor_id = integer(), is_semantic_attractor = logical(), is_correct_semantic_attractor = logical(),
  old_stored_signal_A = I(list()), new_stored_signal_A = I(list()),
  old_stored_signal_B = I(list()), new_stored_signal_B = I(list()),
  prob = numeric(), success = integer(), evidence = numeric(),
  #expressive_A = logical(), expressive_B = logical(),
  is_expressive_trial = logical(),
  old_guess_A = numeric(), new_guess_A = numeric(),
  old_guess_B = numeric(), new_guess_B = numeric(),
  stringsAsFactors = FALSE)

# # Run simulation function
# set.seed(2534)
# d.simulation <- rbind(
#   d.empty %>%
#     run_interaction_sim(n_sim = 1000, n_rounds = 300, n_generations = 1, recognition_bias = FALSE, expressive_agents = FALSE) %>%
#     mutate(model_type = "baseline"),
#   d.empty %>%
#     run_interaction_sim(n_sim = 1000, n_rounds = 300, n_generations = 1, recognition_bias = FALSE, expressive_agents = TRUE) %>%
#     mutate(model_type = "expressiveAgents"),
#   d.empty %>%
#     run_interaction_sim(n_sim = 1000, n_rounds = 300, n_generations = 1, recognition_bias = TRUE, expressive_agents = FALSE) %>%
#     mutate(model_type = "recognitionBias"))

# # save simulation data
# saveRDS(d.simulation, file = here::here("scripts", "temp_data", "d_simulation.rds"), compress = TRUE)

# use "git lfs pull" in terminal to pull large data files
d.simulation <- readRDS(here::here("scripts", "temp_data", "d_simulation.rds")) %>%
  # split signal cols for easier processing
  mutate(
    produced_signal_x = map_dbl(produced_signal, 1),
    produced_signal_y = map_dbl(produced_signal, 2),
    old_stored_signal_A_x = map_dbl(old_stored_signal_A, 1),
    old_stored_signal_A_y = map_dbl(old_stored_signal_A, 2),
    old_stored_signal_B_x = map_dbl(old_stored_signal_B, 1),
    old_stored_signal_B_y = map_dbl(old_stored_signal_B, 2),
    new_stored_signal_A_x = map_dbl(new_stored_signal_A, 1),
    new_stored_signal_A_y = map_dbl(new_stored_signal_A, 2),
    new_stored_signal_B_x = map_dbl(new_stored_signal_B, 1),
    new_stored_signal_B_y = map_dbl(new_stored_signal_B, 2)) %>%
  select(-produced_signal, -old_stored_signal_A, -old_stored_signal_B,
         -new_stored_signal_A, -new_stored_signal_B)


# PLOT ICONICITY----

# Signal space use across simulations
d_signal <- d.simulation %>%
  mutate(total_round = (generation - 1) * 300 + round)

# theme
timo_theme <- theme_classic() + 
  theme(legend.position = "right",
        text = element_text(size = 12, 
                            family = "Roboto"),
        plot.title = element_text(color = "black",
                                  size = 17,
                                  vjust = 0,
                                  face = "bold",
                                  margin = margin(t = 0, r = 0, b = 0.5, l = 0, unit = "cm")),
        plot.subtitle = element_text(size = 12, 
                                     color = "#555555"),
        strip.placement = "outside", 
        strip.background =element_rect(color = NA),
        strip.text = element_text(size = 12, 
                                  hjust = .5,
                                  color = "#555555"),
        axis.title.x = element_text(size = 12, 
                                    hjust = .5,
                                    color = "#555555"),
        axis.title.y = element_text(size = 12, 
                                    angle = 0, 
                                    vjust = 0.5,
                                    color = "#555555"),
        axis.line = element_line(color = "#555555"),
        axis.ticks = element_line(color = "#555555"),
        axis.text = element_text(size = 9,
                                 color = "#555555"),
        plot.margin = unit(c(0.5,0.5,0.5,0.5),
                           "cm"))

## Signal space over time----
signal_space_map <- 
  d_signal  |>
  mutate(model_type = factor(
    model_type, 
    levels = c("baseline", "recognitionBias", "expressiveAgents"),
    labels = c("baseline", "recognition bias", "expressive agents"),
    ordered = TRUE)#,
    # produced_signal_x = sapply(produced_signal, function(x) x[1]),
    # produced_signal_y = sapply(produced_signal, function(x) x[2])
  ) |> 
  filter(type == "small") |> 
  mutate(bins = case_when(
    round >= 0 & round <= 10  ~ "1-10",
    round >= 50 & round <= 60  ~ "50-60",
    round >= 290 & round <= 300 ~ "290-300"
  ),
  bins = factor(bins, levels = c("1-10", "50-60", "290-300"), ordered = TRUE)
  ) |> 
  filter(!is.na(bins)) |> 
  ggplot(aes(x = produced_signal_x, y = produced_signal_y)) +
  # binwidth = 1/3 creates 3 bins for both X and Y across the 0-1 range
  # boundary = 0 forces the bins to start exactly at 0.0
  stat_binhex() +
  # Use a clean color scale for the heatmap counts
  scale_fill_gradientn(
    # Step A: Define the exact sequence of colors
    colors = c("#f7f7f7", "#fe9e2a", "#d7191c"),
    
    # Step B: Map those colors to specific numeric points along the data range
    # Rescale your target numbers between 0.0 (min) and 1.0 (max)
    values = scales::rescale(c(0.0, 0.1, 1.0)),
    
    # Step C: Customize the legend appearance
    #guide = guide_colorbar(barwidth = 1, barheight = 15)
  ) +
  
  # Styling
  theme_minimal() +
  labs(
    title = "The evolution of the signal\nfor small referents over time",
    subtitle = "Data binned into equal intervals",
    x = "",
    y = "",
    fill = "Count"
  ) +
  facet_grid(bins~model_type) +
  scale_x_continuous(limits = c(0,1),
                     breaks = seq(0,1,1/3),
                     labels = c(0,"1/3", "2/3", 1)) +
  scale_y_continuous(limits = c(0,1),
                     breaks = seq(0,1,1/3),
                     labels = c(0,"1/3", "2/3", 1)) +
  timo_theme +
  theme(
    legend.position = "none",
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text = element_blank(),
    strip.text.x = element_text(size = 6),
    strip.text.y = element_text(angle = 0),
    plot.title = element_text(face = "bold")
  ) 

ggsave(here::here("figures",  "signal_space_map.png"),
       signal_space_map,
       device = "png",
       bg = "white",
       width = 110, 
       height = 130, 
       units = "mm", 
       dpi = 300) 


## GIF - Signal space over time ----
# take last plot and create a GIF and track proportions as bar plot underneath

  # --- define attractor centers, now type-specific since "iconic" flips position by referent type ---
  attractors_by_type <- tribble(
    ~type,    ~target_attractor_id, ~target_attractor_x, ~target_attractor_y,
    "small",  "distractor-A",        0.15,                0.15,
    "small",  "iconic",              0.15,                0.85,
    "small",  "anti-iconic",         0.85,                0.15,
    "small",  "distractor-B",        0.85,                0.85,
    "large",  "distractor-A",        0.15,                0.15,
    "large",  "iconic",              0.85,                0.15,
    "large",  "anti-iconic",         0.15,                0.85,
    "large",  "distractor-B",        0.85,                0.85
  )
radius <- 0.3

# --- prep data once, outside the loop (dropped the type filter, added type factor) ---
signal_space_data <- d_signal |>
  mutate(
    model_type = factor(
      model_type,
      levels = c("baseline", "expressiveAgents", "recognitionBias"),
      labels = c("baseline", "expressive agents", "recognition bias"),
      ordered = TRUE
    ),
    type = factor(type, levels = c("small", "large"), ordered = TRUE)#,  # small = top row, large = bottom row
    # produced_signal_x = sapply(produced_signal, function(x) x[1]),
    # produced_signal_y = sapply(produced_signal, function(x) x[2])
  )

# --- helper: compute proportion of points within each attractor circle, now joined by type ---
compute_attractor_props <- function(round_data) {
  round_data |>
    left_join(attractors_by_type, by = "type", relationship = "many-to-many") |>
    mutate(
      dist = sqrt((produced_signal_x - target_attractor_x)^2 + (produced_signal_y - target_attractor_y)^2),
      in_attractor = dist <= radius
    ) |>
    group_by(model_type, target_attractor_id) |>
    summarise(
      n_total = n(),
      n_in    = sum(in_attractor),
      prop    = n_in / n_total,
      .groups = "drop"
    ) |>
    mutate(label = scales::percent(prop, accuracy = 1))
}

# --- output directory ---
frame_dir <- here::here("figures", "gif_frames")
dir.create(frame_dir, showWarnings = FALSE)

rounds <- sort(unique(signal_space_data$round))

# --- shared y-axis max for barplot across all rounds ---
all_props <- map_dfr(rounds, function(r) {
  compute_attractor_props(signal_space_data |> filter(round == r)) |> mutate(round = r)
})
barplot_ymax <- max(all_props$prop) * 1.1

my_colors <- c("distractor-A" = "grey",
               "iconic"       = "#d7191c",
               "anti-iconic"  = "#1A9494",
               "distractor-B" = "grey")

# --- loop: one png per round ---
frame_paths <- map_chr(rounds, function(r) {
  
  round_data <- signal_space_data |> filter(round == r)
  attractor_props <- compute_attractor_props(round_data)
  
  # --- main hexbin map, now faceted by type (rows) x model_type (columns) ---
  p_main <- round_data |>
    ggplot(aes(x = produced_signal_x, y = produced_signal_y)) +
    stat_binhex() +
    scale_fill_gradientn(
      colors = c("#f7f7f7", "#fe9e2a", "#d7191c"),
      values = scales::rescale(c(0.0, 0.2, 1.0)),
      limits = c(0, NA)
    ) +
    theme_minimal() +
    labs(
      title = "The evolution of the signal over time",
      subtitle = paste("Round:", r),
      x = "", y = "", fill = "Count"
    ) +
    facet_grid(type ~ model_type) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 1/3),
                       labels = c(0, "1/3", "2/3", 1)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 1/3),
                       labels = c(0, "1/3", "2/3", 1)) +
    timo_theme +
    theme(
      legend.position = "none",
      axis.line = element_blank(),
      axis.ticks = element_blank(),
      panel.grid.minor = element_blank(),
      axis.text = element_blank(),
      strip.text.y = element_text(angle = 0),
      plot.title = element_text(face = "bold")
    )
  
  # --- barplot underneath, also faceted by type x model_type ---
  p_bar <- attractor_props |>
    ggplot(aes(x = target_attractor_id, y = prop, fill = target_attractor_id)) +
    geom_col(width = 0.7) +
    geom_text(aes(label = scales::percent(prop, accuracy = 1)),
              hjust = -0.3, size = 2.8) +
    coord_flip() +
    scale_y_continuous(limits = c(0, barplot_ymax + 0.1),
                       labels = scales::percent_format(accuracy = 1)) +
    scale_fill_manual(values = my_colors) +
    facet_grid(~model_type) +               # back to one row, matching p_main's columns only
    labs(x = NULL, y = NULL) +
    theme_minimal() +
    theme(
      legend.position = "none",
      strip.text = element_blank(),
      axis.text.y = element_text(size = 7, hjust = 1),
      axis.text.x = element_blank(),
      panel.grid.minor = element_blank()
    )
  
  p_main <- p_main + theme(plot.margin = margin(t = 5, r = 5, b = 0, l = 5))
  p_bar  <- p_bar  + theme(plot.margin = margin(t = 0, r = 5, b = 5, l = 5))
  
  p_combined <- (p_main / p_bar) + plot_layout(heights = c(3, 1))  # back to original ratio
  
  out_path <- file.path(frame_dir, sprintf("round_%03d.png", r))
  ggsave(out_path, p_combined, width = 140, height = 130, units = "mm", dpi = 150)  
  
  out_path
})

# --- gif stitching (unchanged) ---
n <- length(frame_paths)
delay_vector <- round(seq(30, 1, length.out = n))
delay_vector[n] <- 1000

imgs <- image_read(frame_paths)
gif <- image_animate(imgs, delay = delay_vector, loop = 0)

image_write(gif, here::here("figures", "signal_space_evolution.gif"), format = "gif")


## Proportion over time ----
average_prop_interactions_wide <- 
all_props |> 
  ggplot(aes(x = round, y = prop, group = interaction(model_type, target_attractor_id),
             color = target_attractor_id)) +
  geom_path(linewidth = 2, alpha = 1) +
  geom_hline(yintercept = 0,
             color = "white",
             lty = "dashed") +
  # # Add lines at generational overturn
  geom_hline(yintercept = seq(0, 1, by = .25),
             color = "grey",
             lty = "dotted") +
  scale_color_manual(values = my_colors) +
  scale_y_continuous(limits = c(0,1), breaks = seq(0,1,.25),
                     labels = paste0(seq(0,100,25), "%")) +
  scale_x_continuous(breaks = seq(0, 300, by = 50),
                     labels = seq(0,300,50)) +
  guides(x = guide_axis(cap = "both"),
         y = guide_axis(cap = "both")) +
  labs(title = "Iconicity evolves via both expressive\nspeakers and recognition bias",
       subtitle = "% of signals in attractors",
       y = "", 
       x = "\nInteraction rounds",
       color = "") +
  facet_wrap(model_type ~ .) +
  timo_theme +
  # Apply theme changes ONLY to the y-side panel
  theme(
    #legend.position = c(0.2,0.8),
    legend.position = "none",
    ggside.panel.grid.major = element_blank(),
    ggside.panel.grid.minor = element_blank(),
    ggside.axis.text = element_blank(),
    ggside.axis.line = element_blank(),
    ggside.axis.ticks = element_blank(),
    ggside.panel.scale.y = 0.25
  )


ggsave(here::here("figures",  "average_prop_interactions_wide.png"),
       average_prop_interactions_wide,
       device = "png",
       bg = "white",
       width = 150, 
       height = 110, 
       units = "mm", 
       dpi = 300) 

## Proportion over time x Production noise ----
d.grid.recognitionBias <- readRDS("models/grid-search-recognitionBias.rds")
d.grid.expressiveAgents <- readRDS("models/grid-search-expressiveAgents.rds")

# COMMENT need to plot average_prop_interactions_wide for all three production noise conditions