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
# saveRDS(d.simulation, file = "scripts/temp_data/d_simulation.rds", compress = TRUE)

# use "git lfs pull" in terminal to pull large data files
d.simulation <- readRDS("scripts/temp_data/d_simulation.rds") %>%
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


# EVALUATE ICONICITY----
# Signal space use across simulations
d_signal_mean <- d.simulation %>%
  mutate(total_round = (generation - 1) * 300 + round) %>%
  group_by(model_type, total_round, type, generation) %>%
  summarise(
    mean_x = mean(produced_signal_x, na.rm = TRUE),
    mean_y = mean(produced_signal_y, na.rm = TRUE),
    .groups = "drop")

# Signal space use across simulations
d_signal <- d.simulation %>%
  mutate(total_round = (generation - 1) * 300 + round)

# calculate proportion of trials in an attractor, in a semantic attractor, in the correct semantic attractor
d.simulation %>%
  mutate(across(c(in_attractor, is_semantic_attractor, is_correct_semantic_attractor), as.logical)) %>%
  group_by(model_type) %>%
  summarise(
    total_trials = n(),
    n_in = sum(in_attractor == TRUE, na.rm = TRUE),
    n_out = sum(in_attractor == FALSE, na.rm = TRUE),
    n_sem = sum(is_semantic_attractor == TRUE, na.rm = TRUE),
    n_corr_sem = sum(is_correct_semantic_attractor == TRUE, na.rm = TRUE),
    pct_in_attractor = (n_in / total_trials) * 100,
    pct_out_attractor = (n_out / total_trials) * 100,
    pct_in_sem_attractor = (n_sem / n_in) * 100,           # Out of those IN an attractor
    pct_in_correct_sem_attractor = (n_corr_sem / n_sem) * 100,     # Out of those that are semantic
    .groups = "drop") %>%
  select(-n_in, -n_out, -n_sem, -n_corr_sem)

# calculate proportion of trials ENDING UP in an attractor, in a semantic attractor, in the correct semantic attractor
d.simulation %>%
  mutate(across(c(in_attractor, is_semantic_attractor, is_correct_semantic_attractor), as.logical)) %>%
  filter(round %in% c(300)) |> 
  group_by(model_type, round) %>%
  summarise(
    total_trials = n(),
    n_in = sum(in_attractor == TRUE, na.rm = TRUE),
    n_out = sum(in_attractor == FALSE, na.rm = TRUE),
    n_sem = sum(is_semantic_attractor == TRUE, na.rm = TRUE),
    n_corr_sem = sum(is_correct_semantic_attractor == TRUE, na.rm = TRUE),
    pct_in_attractor = (n_in / total_trials) * 100,
    pct_out_attractor = (n_out / total_trials) * 100,
    pct_in_sem_attractor = (n_sem / n_in) * 100,           # Out of those IN an attractor
    pct_in_correct_sem_attractor = (n_corr_sem / n_sem) * 100,     # Out of those that are semantic
    .groups = "drop") %>%
  select(-n_in, -n_out, -n_sem, -n_corr_sem)

# plot iconicity
d.iconicity <- d.simulation |> 
  mutate(model_type = factor(
    model_type, 
    levels = c("baseline", "recognitionBias", "expressiveAgents"),
    labels = c("baseline", "recognition bias", "expressive agents"),
    ordered = TRUE),
    total_round = (generation - 1) * 300 + round,
    strength = abs(evidence)) %>%
  group_by(model_type, simulation, generation, total_round, type, referent) %>%
  summarise(
    evidence = mean(evidence),
    strength = mean(strength),
    .groups = "drop") %>%
  group_by(model_type, simulation, generation, total_round) %>%
  summarise(
    evidence = mean(evidence),
    strength = mean(strength),
    .groups = "drop")

d.iconicity.mean <- d.iconicity |> 
  group_by(model_type, total_round) %>%
  summarise(evidence = mean(evidence),
            strength = mean(strength),
            .groups = "drop")

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


## plot: average iconicity over time----
average_iconicity_interactions_long <- 
  d.iconicity |> 
  ggplot(aes(x = total_round, y = evidence, group = interaction(model_type, simulation),
             color = evidence)) +
  geom_path(linewidth = 0.5, alpha = 0.05,
            color = "black") +
  geom_hline(yintercept = 0,
             color = "white",
             lty = "dashed") +
  geom_path(data = d.iconicity.mean, 
            aes(group = 1), linewidth = 3,
            color = "white") +
  geom_path(data = d.iconicity.mean, 
            aes(group = 1), linewidth = 1.5) +  
  # # Add lines at generational overturn
  # geom_vline(xintercept = seq(0, max(d.iconicity$total_round), by = 50), 
  #            color = "grey", 
  #            lty = "dotted") +
  scale_color_viridis_c(begin = 0,
                        end = 1,
                        values = seq(0,1,0.1))+
  scale_y_continuous(limits = c(-0.6,0.8), breaks = seq(-1,1,0.25)) +
  scale_x_continuous(breaks = seq(0, max(d.iconicity$total_round), by = 50),
                     labels = seq(0, max(d.iconicity$total_round), by = 50)) +
  guides(x = guide_axis(cap = "both"),
         y = guide_axis(cap = "both")) +
  labs(title = "Iconicity evolves via both expressive\nspeakers and recognition bias",
       subtitle = "Iconicity, iconic > 0 > anti-iconic",
       y = "", 
       x = "\nInteraction rounds") +
  facet_wrap(model_type ~ ., ncol = 1) +
  # marginal distribution per facet (y-axis)
  # geom_ysidedensity(
  #   data = d.iconicity |> filter(total_round == 50),
  #   aes(y = evidence, group = model_type),
  #   inherit.aes = FALSE,
  #   color = NA,
  #   fill = "black",
  #   alpha = 0.5) +
  #scale_ysidex_continuous() +
  timo_theme +
  # Apply theme changes ONLY to the y-side panel
  theme(
    legend.position = "none",
    ggside.panel.grid.major = element_blank(),
    ggside.panel.grid.minor = element_blank(),
    ggside.axis.text = element_blank(),
    ggside.axis.line = element_blank(),
    ggside.axis.ticks = element_blank(),
    ggside.panel.scale.y = 0.25
  )

average_iconicity_interactions_wide <- 
  d.iconicity |> 
  ggplot(aes(x = total_round, y = evidence, group = interaction(model_type, simulation),
             color = evidence)) +
  geom_path(linewidth = 0.5, alpha = 0.05,
            color = "black") +
  geom_hline(yintercept = 0,
             color = "white",
             lty = "dashed") +
  geom_path(data = d.iconicity.mean, 
            aes(group = 1), linewidth = 3,
            color = "white") +
  geom_path(data = d.iconicity.mean, 
            aes(group = 1), linewidth = 1.5) +  
  # # Add lines at generational overturn
  # geom_vline(xintercept = seq(0, max(d.iconicity$total_round), by = 50), 
  #            color = "grey", 
  #            lty = "dotted") +
  scale_color_viridis_c(begin = 0,
                        end = 1,
                        values = seq(0,1,0.1))+
  scale_y_continuous(limits = c(-0.6,0.8), breaks = seq(-1,1,0.25)) +
  scale_x_continuous(breaks = seq(0, max(d.iconicity$total_round), by = 50),
                     labels = seq(0, max(d.iconicity$total_round), by = 50)) +
  guides(x = guide_axis(cap = "both"),
         y = guide_axis(cap = "both")) +
  labs(title = "Iconicity evolves via both expressive\nspeakers and recognition bias",
       subtitle = "Iconicity, iconic > 0 > anti-iconic",
       y = "", 
       x = "\nInteraction rounds") +
  facet_wrap(model_type ~ .) +
  # marginal distribution per facet (y-axis)
  # geom_ysidedensity(
  #   data = d.iconicity |> filter(total_round == 50),
  #   aes(y = evidence, group = model_type),
  #   inherit.aes = FALSE,
  #   color = NA,
  #   fill = "black",
  #   alpha = 0.5) +
  #scale_ysidex_continuous() +
  timo_theme +
  # Apply theme changes ONLY to the y-side panel
  theme(
    legend.position = "none",
    ggside.panel.grid.major = element_blank(),
    ggside.panel.grid.minor = element_blank(),
    ggside.axis.text = element_blank(),
    ggside.axis.line = element_blank(),
    ggside.axis.ticks = element_blank(),
    ggside.panel.scale.y = 0.25
  )

ggsave("figures/average_iconicity_interactions_long.png",
       average_iconicity_interactions_long,
       device = "png",
       bg = "white",
       width = 120, 
       height = 280, 
       units = "mm", 
       dpi = 300) 

ggsave("figures/average_iconicity_interactions_wide.png",
       average_iconicity_interactions_wide,
       device = "png",
       bg = "white",
       width = 130, 
       height = 100, 
       units = "mm", 
       dpi = 300) 


## plot: guess rate over time----
guess_rate <- d.simulation |> 
  mutate(model_type = factor(
    model_type, 
    levels = c("baseline", "recognitionBias", "expressiveAgents"),
    labels = c("baseline", "recognition bias", "expressive agents"),
    ordered = TRUE),
    total_round = (generation - 1) * 300 + round,
    strength = abs(evidence)) %>%
  group_by(model_type, simulation, generation, total_round, type, referent) %>%
  summarise(
    evidence = mean(evidence),
    strength = mean(strength),
    guess = mean(new_guess_A, new_guess_B),
    .groups = "drop") %>%
  group_by(model_type, generation, total_round) %>%
  summarise(
    evidence = mean(evidence),
    strength = mean(strength),
    guess = mean(guess),
    .groups = "drop")


average_guess_rate <- 
  guess_rate |> 
  ggplot(aes(x = total_round, y = guess, group = interaction(model_type),
             color = evidence)) +
  geom_path(linewidth = 1,
            color = "black") +
  # # Add lines at generational overturn
  geom_hline(yintercept = 0.25,
             color = "grey",
             lty = "dotted") +
  annotate("text",
           x = 5,
           y = 0.2,
           label = "starting value",
           size = 4,
           hjust = 0) +
  scale_color_viridis_c(begin = 0,
                        end = 1,
                        values = seq(0,1,0.1))+
  scale_y_continuous(limits = c(0,1), breaks = seq(0,1,0.25)) +
  scale_x_continuous(breaks = seq(0, 50, by = 25),
                   labels = seq(0, 50, by = 25)) +
  guides(x = guide_axis(cap = "both"),
         y = guide_axis(cap = "both")) +
  labs(title = "Memory strength increases over time\nthrough learningbut levels off.",
       subtitle = "Memory strength measured between 0 (not learned) to 1 (fully learned)",
       y = "\n", 
       x = "\nInteraction rounds") +
  facet_wrap(model_type ~.) +
  timo_theme 

ggsave("figures/average_guess_rate.png",
       average_guess_rate,
       device = "png",
       bg = "white",
       width = 180, 
       height = 120, 
       units = "mm", 
       dpi = 300) 


## signal space
# Signal space use across simulations
center_small <- c(0.15, 0.85)
center_large <- c(0.85, 0.15)
neutral_attractors <- list(c(0.15, 0.15), c(0.85, 0.85))

# The unified set used by produce_signal() in the simulation above.
plot_attractor_centers <- list(center_small, center_large,
                               neutral_attractors[[1]], neutral_attractors[[2]])

circle_radius <- 0.3
drift_sd <- 0.2
k_production <- 1.5
speaker_guess_fixed <- 0.5
center_sd <- 0.12
k_attractor_fixed <- 2.5

grid <- expand.grid(
  x = seq(0, 1, length.out = 200),
  y = seq(0, 1, length.out = 200),
  type = c("small", "large"))

grid_mapped <- grid %>%
  rowwise() %>%
  mutate(
    ease = signal_evidence(c(x, y),
                           if (type == "small") center_small else center_large,
                           circle_radius = circle_radius, k_perception = k_attractor_fixed),
    # localized production SD -- driven by ALL FOUR attractors
    local_sd = {
      base_sd <- drift_sd * (1 + k_production * (0.5 - speaker_guess_fixed))
      distances <- sapply(
        plot_attractor_centers,
        function(center) sqrt(sum((c(x, y) - center)^2)))
      inside_attractors <- distances < circle_radius
      if (any(inside_attractors)) {
        closest_dist <- min(distances[inside_attractors])
        rel_dist <- closest_dist / circle_radius
        
        magnitude <- exp(-k_attractor_fixed * rel_dist)
        max_mag <- exp(-k_attractor_fixed * 0)
        min_mag <- exp(-k_attractor_fixed * 1)
        exp_scale <- (magnitude - min_mag) / (max_mag - min_mag)
        
        edge_sd <- drift_sd
        base_sd <- edge_sd - (edge_sd - center_sd) * exp_scale
      }
      base_sd
    }
  ) %>%
  ungroup()


# Mark all four attractor centers, distinguishing -- per facet -- this
# referent's correct meaning, the OTHER referent's (wrong) meaning, and
# the two purely arbitrary/neutral attractors. Built with base merge()
# to avoid requiring tidyr::crossing(), but that works equally well.
attractor_points <- merge(
  data.frame(
    x = c(0.15, 0.85, 0.15, 0.85),
    y = c(0.85, 0.15, 0.15, 0.85),
    is_semantic = c(TRUE, TRUE, FALSE, FALSE)),
  data.frame(type = c("small", "large"))) %>%
  mutate(
    role = case_when(
      !is_semantic ~ "neutral",
      type == "small" & x == 0.15 & y == 0.85 ~ "correct",
      type == "large" & x == 0.85 & y == 0.15 ~ "correct",
      is_semantic ~ "wrong",
      TRUE ~ NA_character_))

grid_mapped %>%
  ggplot(
    aes(
      x = x, 
      y = y)) +
  geom_tile(aes(fill = ease)) +
  # Contour lines now map out all four basins, not two -- identical
  # across facets, since production doesn't know about meaning.
  geom_contour(aes(
    z = local_sd), 
    bins = 10, 
    color = "white", alpha = 0.7) +
  geom_point(
    data = attractor_points,
    aes(x = x, y = y, 
        shape = role, color = role),
        size = 4, stroke = 1.2) +
  scale_shape_manual(values = c(correct = 4, wrong = 4, neutral = 1)) +
  scale_color_manual(values = c(correct = "white", wrong = "red", neutral = "white")) +
  facet_wrap(~ type) +
  scale_fill_viridis_c() +
  theme_minimal() +
  coord_fixed(xlim = c(0, 1), ylim = c(0, 1))


# plot average 
## not much movement
d_signal_mean  |> 
  group_by(model_type, type, generation) |> 
  summarise(mean_x = mean(mean_x),
            mean_y = mean(mean_y)) |> 
  #filter(generation == 10) |> 
  ggplot(
    aes(x = mean_x, y = mean_y,
        fill = as.ordered(generation))) +
  facet_grid(model_type ~ type) +
  geom_hline(yintercept = 0.5, 
             lty = "dashed") + 
  geom_vline(xintercept = 0.5, 
             lty = "dashed") +
  geom_point(alpha = 0.8, 
             size = 5, 
             pch = 21,
             color = "white") +
  labs(x = "x dimnesion",
       y = "y dimnesion",
       fill = "generations") 

# plot individual
temp_ind <- d_signal  |> 
  filter(model_type == "baseline") |> 
  filter(simulation == 1) |> 
  filter(generation  == 1)

ggplot(data = temp_ind,
       aes(x = produced_signal_x, y = produced_signal_y,
           fill = as.ordered(round),
           color = as.ordered(round),
           group = interaction(type, referent))) +
  facet_grid(referent ~ type) +
  geom_hline(yintercept = 0.5, 
             lty = "dashed") + 
  geom_vline(xintercept = 0.5, 
             lty = "dashed") +
  geom_path(
    arrow = arrow(length = unit(0.2, "cm"), type = "closed")
  ) +
  # success and fail
  geom_point(data = temp_ind |> filter(success == 1),
             alpha = 1, 
             size = 5, 
             pch = 21,
             color = "white") +
  geom_point(data = temp_ind |> filter(success == 0),
             alpha = 1, 
             size = 5, 
             pch = 24,
             fill = "darkred",
             color = "white") +
  labs(x = "x dimnesion",
       y = "y dimnesion",
       fill = "generations") +
  theme_bw() +
  theme()

# calculation ED to semantic attractor, to phonological distractors and semantic repeller
euc_dist <- function(x1, x2){
  return(sqrt(sum((x1 - x2)^2)))
}

# test
euc_dist(c(0.15,0.85), c(0.44,0.56))

# add as vector
d_signal_ed <- d_signal |> 
  mutate(model_type = factor(
    model_type, 
    levels = c("baseline", "expressiveAgents", "recognitionBias"),
    labels = c("baseline", "expressive agents", "recognition bias"),
    ordered = TRUE)) |> 
  mutate(iconic_ed = ifelse(type == "small",
                            sqrt((produced_signal_x - 0.15)^2 + (produced_signal_y - 0.85)^2),
                            sqrt((produced_signal_x - 0.85)^2 + (produced_signal_y - 0.15)^2)),
         attractor_1_ed = sqrt((produced_signal_x - 0.15)^2 + (produced_signal_y - 0.15)^2),
         attractor_2_ed = sqrt((produced_signal_x - 0.85)^2 + (produced_signal_y - 0.85)^2)
  )

d_signal_ed_agg <- d_signal_ed |> 
  group_by(model_type, simulation, type, round) |> 
  summarise(iconic_ed = mean(iconic_ed),
            attractor_1_ed = mean(attractor_1_ed),
            attractor_2_ed = mean(attractor_2_ed))

d_signal_ed_agg_all <- d_signal_ed_agg |> 
  group_by(model_type, type, round) |> 
  summarise(iconic_ed = mean(iconic_ed),
            attractor_1_ed = mean(attractor_1_ed),
            attractor_2_ed = mean(attractor_2_ed))

# get reference values
## if small where is border for attractor? 
# NOTE TR: that does not take the exponentional nature into account
iconic_border = euc_dist(c(0.15,0.85), c(0.45,0.55)) ## 0.42
neutral = euc_dist(c(0.15,0.85), c(0.5,0.5)) ## ~0.5

# for annotation below
arrow_data <- data.frame(
  x = 2,        # Arrow tail X
  y = 0.22,       # Arrow tail Y
  xend = 2,     # Arrow head X
  yend = 0,    # Arrow head Y
  model_type = "baseline", # Target facet level
  text = "more iconic" 
)

## plot: average ED over time----
average_ED_interactions <- 
  d_signal_ed_agg_all |> 
  ggplot(aes(x = round, y = iconic_ed,
             group = interaction(model_type))) +
  geom_line(data = d_signal_ed_agg |> 
              filter(simulation %in% sample(unique(d_signal_ed_agg$simulation), size = 15)),
            aes(group = interaction(model_type, simulation)),
            color = "black",
            alpha = 0.1) +
  geom_line(data = d_signal_ed_agg_all,
            color = "white",
            size = 3) +
  geom_line(data = d_signal_ed_agg_all,
            color = "#440154ff",
            size = 1.5) +
  # iconic border
  # geom_hline(yintercept = iconic_border,
  #            color = "purple",
  #            lty = "dashed") +
  # annotate("text", 
  #          x = 25,
  #          y = 0.21,
  #          color = "purple",
  #          label = "within iconic pocket") +
  # neutral position
  # geom_hline(yintercept = neutral,
  #            color = "white",
  #            lty = "dashed") +
  geom_segment(data = arrow_data,
               aes(x = x, y = y, xend = xend, yend = yend),
           arrow = arrow(length = unit(0.1, "cm"),
                         type = "closed"),
           size = 6, linewidth = 0.5,
           colour = "black",
  ) +
  geom_text(data = arrow_data,
            aes(x = x+2,
                y = yend,
                label = text),
            #x = 10,
            #y = 1.25,
            hjust = 0,
            size = 3,
            colour = "black",
  ) +
  facet_grid(~model_type) +
  scale_y_continuous(limits = c(0,1.3), 
                     trans = "reverse",
                     breaks = seq(0,1.25,0.25)) +
  scale_x_continuous(breaks = seq(0, max(d.iconicity$total_round), by = 25),
                     labels = seq(0, max(d.iconicity$total_round), by = 25)) +
  guides(x = guide_axis(cap = "both"),
         y = guide_axis(cap = "both")) +
  labs(title = "The signal moves to iconic pocket for both\nexpressive speakers and recognition bias ",
       subtitle = "euclidean distance to center of iconic pocket\nblack lines illustrate 15 random simulations",
       y = "",
       x = "\nInteraction rounds") +
  timo_theme

ggsave("figures/average_ED_interactions.png",
       average_ED_interactions,
       device = "png",
       bg = "white",
       width = 180, 
       height = 120, 
       units = "mm", 
       dpi = 300) 

# plot also for distractor
d_signal_ed_agg_all |> 
  ggplot(aes(x = round, y = attractor_1_ed,
             group = interaction(model_type))) +
  geom_path(data = d_signal_ed_agg,
            alpha = 0.01) +
  geom_line(data = d_signal_ed_agg_all,
            alpha = 1) +
  # attractor border
  geom_hline(yintercept = attractor_border,
             color = "black",
             lty = "dashed") +
  annotate("text", 
           x = 50,
           y = 0.21,
           color = "black",
           label = "within distractor") +
  facet_grid(~model_type) +
  labs(title = "euclidean distance to phonological distractors",
       y = "euclidean distance to phonological distractors") +
  theme_minimal()

# how often does the signal enters pockets
d_signal_ed <- d_signal_ed |> 
  mutate(in_pocket = ifelse(iconic_ed < iconic_border, 1, 0)) 

d_signal_ed_agg <- d_signal_ed |> 
  group_by(model_type, simulation, type, round) |> 
  summarise(iconic_ed = mean(iconic_ed),
            attractor_1_ed = mean(attractor_1_ed),
            attractor_2_ed = mean(attractor_2_ed),
            in_pocket = mean(in_pocket))

d_signal_ed_agg_all <- d_signal_ed_agg |> 
  group_by(model_type, type, round) |> 
  summarise(iconic_ed = mean(iconic_ed),
            attractor_1_ed = mean(attractor_1_ed),
            attractor_2_ed = mean(attractor_2_ed),
            in_pocket = mean(in_pocket))

# calculate when attractor was entered and when 
# plot average ED
d_signal_ed_agg_all |> 
  ggplot(aes(x = round, y = in_pocket,
             group = interaction(model_type))) +
  geom_line() +
  facet_grid(~model_type) +
  labs(title = "proportion of entering the iconic pocket",
       y = "proportion of entering the iconic pocket") +
  theme_minimal()



####
d_signal_mean  |>
  group_by(model_type, type, generation) |>
  summarise(mean_x = mean(mean_x),
            mean_y = mean(mean_y)) |> 
  mutate(quadrant = case_when(mean_x < 0.3 & mean_y < 0.3 ~ "top left",
                              between(mean_x, 0.3, 0.6) & mean_y < 0.3 ~ "top middle",
                              mean_x > 0.6 & mean_y < 0.3 ~ "top right",
                              
                              mean_x < 0.3 & between(mean_y, 0.3, 0.6) ~ "middle left",
                              between(mean_x, 0.3, 0.6) & between(mean_y, 0.3, 0.6) ~ "center",
                              mean_x > 0.6 & between(mean_y, 0.3, 0.6) ~ "middle right",
                              mean_x < 0.3 & mean_y > 0.6 ~ "bottom left",
                              between(mean_x, 0.3, 0.6) & mean_y > 0.6 ~ "bottom middle",
                              mean_x > 0.6 & mean_y > 0.6 ~ "bottom right"
  )) |> 
  group_by(model_type, type) |>
  count(quadrant) |> 
  ggplot(
    aes(x = 0:1, y = 0:1,
        fill = quadrant)) +
  geom_tile()
facet_grid(model_type ~ type) +
  geom_hline(yintercept = 0.5, 
             lty = "dashed") + 
  geom_vline(xintercept = 0.5, 
             lty = "dashed") +
  geom_point(alpha = 0.8, 
             size = 5, 
             pch = 21,
             color = "white") +
  labs(x = "x dimnesion",
       y = "y dimnesion",
       fill = "generations") 

my_breaks = c(2, 10, 50, 250, 1250, 6000)


## plot: signal space over time----
signal_space_map <- 
  d_signal  |>
  mutate(model_type = factor(
    model_type, 
    levels = c("baseline", "recognitionBias", "expressiveAgents"),
    labels = c("baseline", "recognition bias", "expressive agents"),
    ordered = TRUE),
    produced_signal_x = sapply(produced_signal, function(x) x[1]),
    produced_signal_y = sapply(produced_signal, function(x) x[2])
  ) |> 
  filter(type == "small") |> 
  mutate(bins = cut(round, 
                    breaks = seq(0, 50, by = 10),
                    labels = c("1-10", "11-20", "21-30", "31-40", "41-50"))) |> 
  # group_by(bins, simulation, model_type) |> 
  # summarise(produced_signal_x = mean(produced_signal_x),
  #           produced_signal_y = mean(produced_signal_y)) |> 
  ggplot(aes(x = produced_signal_x, y = produced_signal_y)) +
  # binwidth = 1/3 creates 3 bins for both X and Y across the 0-1 range
  # boundary = 0 forces the bins to start exactly at 0.0
  #geom_bin2d(binwidth = c(1/27, 1/27), boundary = 0, color = "white", linewidth = 0.5) +
  stat_binhex() +
  # Use a clean color scale for the heatmap counts
  # scale_fill_gradient2(name = "count", trans = "log",
  #                     breaks = my_breaks, labels = my_breaks,
  #                     low = "white", 
  #                     mid = "blue",
  #                     high = "red"
  #                     ) +
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
  # attractor outline
  # annotate("rect", 
  #          xmin = c(0, 2/3, 0, 2/3), 
  #          xmax = c(1/3, 1, 1/3, 1), 
  #          ymin = c(0, 0, 2/3, 2/3), 
  #          ymax = c(1/3, 1/3, 1, 1), 
  #          colour = "black", 
  #          fill = "transparent", 
  #          size = 0.5) +
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
    strip.text.x = element_text(size = 8),
    strip.text.y = element_text(angle = 0),
    plot.title = element_text(face = "bold")
  ) 

ggsave("figures/signal_space_map.png",
       signal_space_map,
       device = "png",
       bg = "white",
       width = 110, 
       height = 150, 
       units = "mm", 
       dpi = 300) 


## GIF ----
# take last plot and create a GIF and track proportions as bar plot underneath

# --- define the four attractor centers and shared radius ---
attractors <- tibble(
  target_attractor_id = c("distractor-A", "iconic", "anti-iconic", "distractor-B"),
  target_attractor_x  = c(0.15, 0.15, 0.85, 0.85),
  target_attractor_y  = c(0.15, 0.85, 0.15, 0.85)
)
radius <- 0.3

# --- prep data once, outside the loop ---
signal_space_data <- d_signal |>
  mutate(
    model_type = factor(
      model_type,
      levels = c("baseline", "expressiveAgents", "recognitionBias"),
      labels = c("baseline", "expressive agents", "recognition bias"),
      ordered = TRUE
    ),
    produced_signal_x = sapply(produced_signal, function(x) x[1]),
    produced_signal_y = sapply(produced_signal, function(x) x[2])
  ) |>
  filter(type == "small")

# --- helper: compute proportion of points within each attractor circle 
# for a given round's data, grouped by model_type
compute_attractor_props <- function(round_data) {
  round_data |>
    tidyr::crossing(attractors) |>
    mutate(
      dist = sqrt((produced_signal_x - target_attractor_x)^2 + (produced_signal_y - target_attractor_y)^2),
      in_attractor = dist <= radius
    ) |>
    group_by(model_type, target_attractor_id, target_attractor_x, target_attractor_y) |>
    summarise(
      n_total = n(),  # undo the crossing-induced 4x row inflation for the denominator
      n_in    = sum(in_attractor),
      prop    = n_in / n_total,
      .groups = "drop"
    ) |>
    mutate(label = scales::percent(prop, accuracy = 1))
}

# --- set up output directory for frames ---
frame_dir <- here::here("figures","gif_frames")
dir.create(frame_dir, showWarnings = FALSE)

# --- get sorted unique rounds ---
rounds <- sort(unique(signal_space_data$round))

# --- compute a shared y-axis max for the barplot across all rounds ---
# (so the barplot scale doesn't jump around frame to frame)
all_props <- map_dfr(rounds, function(r) {
  compute_attractor_props(signal_space_data |> filter(round == r)) |> 
    mutate(round = r)
})
barplot_ymax <- max(all_props$prop) * 1.1

my_colors <- c("distractor-A" = "grey",
               "iconic" = "#d7191c", 
               "anti-iconic" = "#1A9494", 
               "distractor-B" = "grey")

# --- loop: one png per round ---
frame_paths <- map_chr(rounds, function(r) {
  
  round_data <- signal_space_data |> filter(round == r)
  attractor_props <- compute_attractor_props(round_data)
  
  # --- main hexbin map ---
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
      title = "The evolution of the signal\nfor small referents over time",
      subtitle = paste("Round:", r),
      x = "",
      y = "",
      fill = "Count"
    ) +
    facet_grid(~model_type) +
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
  
  # --- barplot underneath: proportion per attractor, faceted by model_type ---
  p_bar <- attractor_props |>
    ggplot(aes(x = target_attractor_id, y = prop, fill = target_attractor_id)) +
    geom_col(width = 0.7) +
    geom_text(aes(label = scales::percent(prop, accuracy = 1)),
              hjust = -0.3, size = 2.8) +
    coord_flip() +
    scale_y_continuous(limits = c(0, barplot_ymax+0.1),
                       labels = scales::percent_format(accuracy = 1)) +
    scale_fill_manual(values = my_colors) +
    facet_grid(~model_type) +
    labs(x = NULL, y = NULL) +
    theme_minimal() +
    theme(
      legend.position = "none",
      strip.text = element_blank(),   # avoid repeating model_type labels twice
      axis.text.y = element_text(size = 7, hjust = 1),
      axis.text.x = element_blank(),
      panel.grid.minor = element_blank()
    )
  
  p_main <- p_main +
    theme(plot.margin = margin(t = 5, r = 5, b = 0, l = 5))  # b = 0 removes bottom gap
  
  p_bar <- p_bar +
    theme(plot.margin = margin(t = 0, r = 5, b = 5, l = 5))  # t = 0 removes top gap
  
  # --- stack main plot on top of barplot ---
  p_combined <- (p_main / p_bar) + plot_layout(heights = c(3, 1))  
  
  out_path <- file.path(frame_dir, sprintf("round_%03d.png", r))
  ggsave(out_path, p_combined, width = 140, height = 100, units = "mm", dpi = 150)
  out_path
})

# control speed
n <- length(frame_paths)

# --- build a graduated delay vector (in centiseconds, i.e. 1/100 sec) ---
# start fast, end slow -- tune these bounds to taste
delay_vector <- round(seq(10, 50, length.out = n))  # 10 = fast (0.1s/frame), 50 = slow (0.5s/frame)

# --- hold the very last frame for 10 seconds ---
delay_vector[n] <- 1000  # 1000 centiseconds = 10 seconds

# --- read all frames and animate with the explicit delay vector ---
imgs <- image_read(frame_paths)
gif <- image_animate(imgs, delay = delay_vector, loop = 0)

image_write(gif, here::here("figures","signal_space_evolution.gif"), format = "gif")


#---- # double facet version
  
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
    type = factor(type, levels = c("small", "large"), ordered = TRUE),  # small = top row, large = bottom row
    produced_signal_x = sapply(produced_signal, function(x) x[1]),
    produced_signal_y = sapply(produced_signal, function(x) x[2])
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
delay_vector <- round(seq(10, 50, length.out = n))
delay_vector[n] <- 1000

imgs <- image_read(frame_paths)
gif <- image_animate(imgs, delay = delay_vector, loop = 0)

image_write(gif, here::here("figures", "signal_space_evolution.gif"), format = "gif")
