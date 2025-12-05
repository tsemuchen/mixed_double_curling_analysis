# Opening-State Modeling of End Outcomes in Mixed Doubles Curling

James Chen & Charlie Ko, 12/5/2025

---

## 1. Problem Statement
In a competitive mixed doubles curling match, the opening plays can quietly shape the final scoring outcome for each end, even though it is hard to tell who is truly leading after only a few  shots. This study aims to identify which early features and strategies most strongly contribute to score in an end, in both Power Play and normal situations, using stone-level data. In particular, we investigate how execution quality, shot selection, and stone position control (e.g., inner-ring/2-ft control) affect end outcomes, and how teams should develop opening strategies to maximize their winning probability for each end, both with or without the hammer.

Curling rules can be found [here](https://worldcurling.org/about/curling/).

---

## 2. Data and Methodology

### 2.1  Reconstructing End States From Stone-Level Data

The data were collected from games played in the 2026 Connecticut Sports Analytics Symposium competition, downloaded from this [GitHub](https://github.com/CSAS-Data-Challenge/2026) page. Starting from the Stones.csv dataset, we first assign a shot order number within each end, then join this information with Ends.csv to determine whether the end is a power play. Because the team that throws second in an end holds the hammer, identifying the team that throws first allows us to infer hammer ownership for that end.

Library:
```
library(dplyr)
library(tidyverse)
library(ggplot2)
library(ggforce)

library(lme4)
library(mgcv)

library(caret)
library(xgboost)
library(pROC)
```

Scoring in curling depends on which team has the stone closest to the button. Using the stone coordinates in Stones.csv, we compute the distance from each stone to the button. From these distances, we determine (i) which team is currently in the scoring position, (ii) how many of that team’s stones lie closer to the button than the opponent’s nearest stone, and (iii) the number of stones in each scoring ring (button, 2-foot, 4-foot, and 6-foot). These derived features summarize the scoring landscape for each end.

#distance function and other related 


### 2.2 Feature Engineering and Modeling Dataset Construction

After reconstructing end states, we focus on a consistent point in every end: the situation immediately after the fourth shot, which is always thrown by the team with the hammer. This moment captures the early structure of the end, including both teams’ intentions and the emerging scoring landscape. To describe how each end begins, we summarize the first two shots taken by each team. The Task variable labels the intended purpose of a shot, and we use it to classify early shots into broader strategic categories.

Curling usually involves two broad kinds of actions. Build-oriented shots, such as draws and guards, are meant to establish position, create cover, or shape the house in a way that sets up future scoring. These shots rarely involve direct contact with opposing stones. Attack-oriented shots, such as takeouts, hits, tap-backs, and peeling actions, aim to remove or disturb opponent stones and immediately alter the scoring environment. Using this grouping, we map each team’s first two shots into one of four opening sequences: build_build, attack_attack, build_then_attack, and attack_then_build. We also summarize execution quality by averaging the judged shot score (0 to 4) across those two shots.

<details>
<summary><strong>Extract opening strategies (click to show code)</strong></summary>

<br>

```r
stones_type <- Stones_plus %>%
  mutate(
    task_type = case_when(
      Task %in% c(0, 5)           ~ "draw",        # Draw, Freeze
      Task %in% c(1, 2)           ~ "guard",       # Front, Guard
      Task %in% c(6, 7, 8, 9, 10) ~ "hit",         # Take-out variants, clearing
      Task %in% c(3, 4)           ~ "tap_soft",    # Raise / Tap-back, Wick / Soft peel
      Task == 11                  ~ "through",
      Task == 13                  ~ "nostat",
      TRUE                        ~ "other"
    )
  )

opening_by_team_end <- stones_type %>%
  group_by(CompetitionID, SessionID, GameID, EndID, TeamID) %>%
  arrange(ShotID, .by_group = TRUE) %>%
  dplyr::slice(1:2) %>%                       # first two stones this team throws in this end
  summarise(
    first_type  = first(task_type),
    second_type = nth(task_type, 2),
    opening_pair  = paste(first_type, second_type, sep = "_"),
    opening_strategy = case_when(
      first_type  %in% c("draw", "guard") &
      second_type %in% c("draw", "guard") ~ "build_build",        # both stones building

      first_type  %in% c("hit", "tap_soft") &
      second_type %in% c("hit", "tap_soft") ~ "attack_attack",    # both stones attacking

      first_type  %in% c("draw", "guard") &
      second_type %in% c("hit", "tap_soft") ~ "build_then_attack",

      first_type  %in% c("hit", "tap_soft") &
      second_type %in% c("draw", "guard") ~ "attack_then_build",

      TRUE ~ "other"
    ),

    # execution summary
    avg_points   = mean(Points, na.rm = TRUE),
    min_points   = min(Points, na.rm = TRUE),
    .groups = "drop"
  )
```
</details>

To build the fourth-shot dataset, we extract the row in Stones_plus that corresponds to the fourth shot in each end and join onto it the opening sequence and execution measures for both teams. This results in a compact description of how each end has unfolded up to that moment from both the hammer team and the opponent.
We then engineer a set of features that summarize the board state at the time of the fourth shot. Using the ring-level reconstructions from Section 2.1, we convert the raw counts into relative measures from the hammer team’s perspective. These include net differences in stone counts across scoring rings and the difference in each team’s closest stone to the button. When a team has no stone in play, a large, fixed placeholder distance is used so that comparisons remain valid.

#Narrow down to fourth_shot dataset

Finally, we attach end outcomes to each fourth-shot record. These include the hammer’s scoring margin, whether the hammer won the end, and whether the hammer earned a two-point or larger end. Rare opening types are removed to avoid unstable modeling. The resulting dataset, referred to as model_df, contains the core strategic variables, execution summaries, engineered board-state features, and outcomes. It serves as the modeling dataset for analyses in the later sections.

#Construct model_df dataset, ready for model fitting

  




### 2.3 Model Design & Developement

With the modeling dataset constructed at the fourth-shot state of each end, our goal is to understand how opening strategies, execution quality, and early board position relate to scoring outcomes. Because these relationships may involve both structured effects (such as team-level tendencies) and more flexible patterns (such as nonlinear effects of execution quality), we employ a set of complementary modeling approaches. Each model type is chosen to address a particular aspect of the problem, and together they allow us to assess strategy effectiveness from statistical, predictive, and causal perspectives.
We begin with linear mixed-effects regression (LMM) and generalized linear mixed-effects models (GLMM). Curling teams vary in strength, style, and consistency, and these differences can influence both strategy and outcomes. Mixed-effects models allow us to control for these latent team characteristics by including random intercepts for both the hammer team and the opposing team. Both models use the same binary scoring indicator as the response. The LMM treats this indicator as approximately continuous, while the GLMM uses a logistic link to model the log-odds of scoring. Using both formulations allows us to examine whether conclusions about strategic effects are robust to the assumed relationship between predictors and the outcome.

#Linear Mixed-Effects Regression (LMM)
#Logistic Mixed-Effects Model (GLMM)

To study the role of shot execution separately from board-position features, we also consider a reduced GLMM that includes execution quality but removes the engineered board-state variables. This model helps isolate whether strategy and shot quality alone have any explanatory value, independent of the more detailed positional features.

#Reduced / Execution-Only Model

Because some relationships may not be strictly linear, particularly those involving execution quality scores or small differences in stone placement, we incorporate a generalized additive model (GAM). GAMs preserve interpretability while allowing smooth nonlinear effects, which makes them well suited for examining whether improvements in execution quality or relative stone position have diminishing, accelerating, or threshold-like effects on scoring probability.

#Generalized Additive Model

To complement these interpretable models, we also include a gradient boosted decision tree model (XGBoost). Tree-based models can capture higher-order interactions and complex, nonlinear patterns that are difficult to specify manually. Although they are less interpretable, they serve as a useful benchmark for predictive performance and provide variable-importance summaries that highlight which features contribute most strongly to outcome prediction.

#XGBoost

Finally, because opening strategy may be influenced by team-specific factors that also affect scoring, we incorporate a propensity-score weighting approach to approximate a causal comparison between attack-first and build-first openings. This framework models the probability that a team chooses an attack-oriented opening, uses those probabilities to weight observations, and then fits a weighted outcome model that estimates how strategy choice relates to scoring under better-balanced comparisons.

#Prospensity model

Taken together, these models allow us to examine strategy effectiveness from multiple angles. Mixed-effects models control for team variation, GAMs allow for flexible functional forms, XGBoost tests predictive strength in a nonparametric way, and propensity weighting provides an approximate causal perspective. Later sections compare these models and interpret their implications for opening-strategy decision making.

#A table that compares the difference between models



### 3. Main Results


### 4. Limitation and Further Development

### 4.1 Limitation

One of the key limitations of this study is the lack of full stone trajectory data. Although we are able to observe final stone coordinates, we cannot reconstruct how each stone moves from frame to frame: its curl, speed profile, or angle from the intended path. Without this information, our model cannot fully distinguish shot quality, difficulty, or tactical intent, all of which are critical in mixed doubles strategy. In addition, the dataset provides limited information about team-level playing tendencies, such as preferred shot types, aggressiveness, or adaptations to score situations. Because these behavioral patterns are only partially observable in the available data, our ability to model team strategy or opponent-specific decision making is inherently restricted.

### 4.2 Further Development

Feature engineering with more comprehensive data to enhance our models.
 - Stone frame-by-frame coordinates: By using the frame-by-frame coordinates of each stone, we can create trajectory-based features such as the curvature of the path and the entry angle into the house. These features give us a deeper understanding of shot quality and shot type.
 - Team and opponent tendencies: By deriving team-level profiles that summarize preferred playing styles and typical game patterns with hammer (aggregated across all ends), our model can suggest customized strategies based on each team’s strategic tendencies.

Apply visual deep-learning methods to stone-position “images”
 - Convert house layouts into image-like representations and apply CNN-based models to automatically detect meaningful spatial patterns that are difficult to encode through coordinates alone.
 - Use these learned visual features to compare end states, allowing the model to uncover strategic structures that traditional feature engineering may overlook and ultimately improve interpretability and predictive performance.

Build an in-game recommendation tool to evaluate shot and strategy options across different scenarios
 - Develop easily interpretable tables that summarize how each team typically performs in different score situations (e.g., tied, down 1, up 2), how aggressive they are with the hammer, and how often they generate steals when trailing.
