# Opening-State Modeling of End Outcomes in Mixed Doubles Curling

James Chen & Charlie Ko, 12/5/2025

---

## 1. Problem Statement
In a competitive mixed doubles curling match, the opening plays can quietly shape the final scoring outcome for each end, even though it is hard to tell who is truly leading after only a few  shots. This study aims to identify which early features and strategies most strongly contribute to score in an end, in both Power Play and normal situations, using stone-level data. In particular, we investigate how execution quality, shot selection, and stone position control (e.g., inner-ring/2-ft control) affect end outcomes, and how teams should develop opening strategies to maximize their winning probability for each end, both with or without the hammer.

**Curling rules can be found [here](https://worldcurling.org/about/curling/).**

---


===============delete this=================

<details>
<summary><strong>Extract opening strategies</strong></summary>

<br>

```r
code
```
</details>

===========================================


## 2. Data and Methodology

### 2.1  Reconstructing End States From Stone-Level Data

The data were collected from games played in the 2026 Connecticut Sports Analytics Symposium competition, downloaded from this [GitHub](https://github.com/CSAS-Data-Challenge/2026) page. Starting from the Stones.csv dataset, we first assign a shot order number within each end, then join this information with Ends.csv to determine whether the end is a power play. Because the team that throws second in an end holds the hammer, identifying the team that throws first allows us to infer hammer ownership for that end.

<details>
<summary><strong>Library</strong></summary>

<br>

```r
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
</details>

Scoring in curling depends on which team has the stone closest to the button. Using the stone coordinates in Stones.csv, we compute the distance from each stone to the button. From these distances, we determine (i) which team is currently in the scoring position, (ii) how many of that team’s stones lie closer to the button than the opponent’s nearest stone, and (iii) the number of stones in each scoring ring (button, 2-foot, 4-foot, and 6-foot). These derived features summarize the scoring landscape for each end.

#distance function and other related 


### 2.2 Feature Engineering and Modeling Dataset Construction

After reconstructing end states, we focus on a consistent point in every end: the situation immediately after the fourth shot, which is always thrown by the team with the hammer. This moment captures the early structure of the end, including both teams’ intentions and the emerging scoring landscape. To describe how each end begins, we summarize the first two shots taken by each team. The Task variable labels the intended purpose of a shot, and we use it to classify early shots into broader strategic categories.

Curling usually involves two broad kinds of actions. Build-oriented shots, such as draws and guards, are meant to establish position, create cover, or shape the house in a way that sets up future scoring. These shots rarely involve direct contact with opposing stones. Attack-oriented shots, such as takeouts, hits, tap-backs, and peeling actions, aim to remove or disturb opponent stones and immediately alter the scoring environment. Using this grouping, we map each team’s first two shots into one of four opening sequences: build_build, attack_attack, build_then_attack, and attack_then_build. We also summarize execution quality by averaging the judged shot score (0 to 4) across those two shots.

<details>
<summary><strong>Extract opening strategies</strong></summary>

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

<details>
<summary><strong>Narrow down to fourth_shot dataset</strong></summary>

<br>

```r
# Extract 4th shot in each end
fourth_shot <- Stones_plus %>%
  filter(n_stone_thrown == 4) %>% 
  rename(TeamID_4th = TeamID,
         score_diff = point_diff)   

# Add opening strategy for the 4th-shot team
fourth_shot <- fourth_shot %>%
  left_join(
    opening_by_team_end %>%
      rename(
        TeamID_4th          = TeamID,
        first_type_4th    = first_type,
        second_type_4th   = second_type,
        opening_pair_4th    = opening_pair,
        opening_strategy_4th = opening_strategy,
        
        avg_points_4th = avg_points,
        min_points_4th = min_points
      ),
    by = c("CompetitionID", "SessionID", "GameID", "EndID", "TeamID_4th")
  )

# Add opponent opening strategy with a second join
fourth_shot <- fourth_shot %>%
  # join opening info for *all* teams in that end
  left_join(
    opening_by_team_end %>%
      rename(
        OppTeamID             = TeamID,
        first_type_opp        = first_type,
        second_type_opp       = second_type,
        opening_pair_opp      = opening_pair,
        opening_strategy_opp  = opening_strategy,
        
        avg_points_opp = avg_points,
        min_points_opp = min_points
      ),
    by = c("CompetitionID", "SessionID", "GameID", "EndID")
  ) %>%
  # keep only the row where OppTeamID is actually the *other* team
  filter(OppTeamID != TeamID_4th)

# Add some net columns 
# (second - first) since we are using the perspective of hammer team.
MAX_DIST <- 4095 # assigned a large dist for all stones out-of-play situation

fourth_shot <- fourth_shot %>% 
  mutate(net_button = n_button_second - n_button_first,
         net_2ft = n_2ft_second - n_2ft_first,
         net_4ft = n_4ft_second - n_4ft_first,
         net_6ft = n_6ft_second - n_6ft_first,
         net_house = net_2ft + net_4ft + net_6ft,
         closest_first  = if_else(is.na(closest_first),  MAX_DIST, closest_first),
         closest_second = if_else(is.na(closest_second), MAX_DIST, closest_second),
         closest_diff   = closest_first - closest_second) # positive = better
```
</details>

Finally, we attach end outcomes to each fourth-shot record. These include the hammer’s scoring margin, whether the hammer won the end, and whether the hammer earned a two-point or larger end. Rare opening types are removed to avoid unstable modeling. The resulting dataset, referred to as model_df, contains the core strategic variables, execution summaries, engineered board-state features, and outcomes. It serves as the modeling dataset for analyses in the later sections.

<details>
<summary><strong>Construct model_df dataset, ready for model fitting</strong></summary>

<br>

```r
model_df <- fourth_shot %>%
  # some light recoding / extra targets
  mutate(
    opening_strategy_4th = factor(opening_strategy_4th),
    opening_strategy_opp = factor(opening_strategy_opp),
    pp_end               = as.logical(pp_end),
    PowerPlay            = factor(PowerPlay),   # if coded 0/1/2 or "", 1, 2
    win_end              = as.integer(score_diff > 0),   # hammer team scores
    big_end              = as.integer(score_diff >= 2)   # 2+ points for hammer
  ) %>%
  select(
    # IDs
    CompetitionID, SessionID, GameID, EndID,
    TeamID_4th, OppTeamID,

    # Strategy (both teams)
    opening_strategy_4th, opening_strategy_opp,
    first_type_4th,  second_type_4th,
    first_type_opp,  second_type_opp,
    PowerPlay, pp_end,

    # Board state: zone counts (raw)
    n_button_first, n_button_second,
    n_2ft_first,    n_2ft_second,
    n_4ft_first,    n_4ft_second,
    n_6ft_first,    n_6ft_second,
    closest_first,  closest_second,

    # Board state: net summaries
    net_button, net_2ft, net_4ft, net_6ft,
    net_house, closest_diff,

    # Execution quality of first two stones
    avg_points_4th, min_points_4th,
    avg_points_opp, min_points_opp,

    # Outcomes
    score_diff, win_end, big_end
  )

# Remove "other" cases (size too small)
model_df <- model_df %>%
  filter(
    opening_strategy_4th != "other",
    opening_strategy_opp != "other"
  )
```
</details>

  




### 2.3 Model Design & Developement

With the modeling dataset constructed at the fourth-shot state of each end, our goal is to understand how opening strategies, execution quality, and early board position relate to scoring outcomes. Because these relationships may involve both structured effects (such as team-level tendencies) and more flexible patterns (such as nonlinear effects of execution quality), we employ a set of complementary modeling approaches. Each model type is chosen to address a particular aspect of the problem, and together they allow us to assess strategy effectiveness from statistical, predictive, and causal perspectives.

We begin with a **generalized linear mixed-effects model (GLMM)**, which accounts for variation in team strength and style by including random intercepts for both the hammer team and the opposing team. The GLMM models the log-odds of scoring as a function of opening strategy, execution quality, and early board-state features. This model serves as a structured baseline that tests whether strategic and positional factors exhibit systematic associations with scoring after controlling for latent team-level tendencies.


<details>
<summary><strong>Generalized Linear Mixed-effects ModelGLMM</strong></summary>

<br>

```r
model_df2 <- model_df %>%
  mutate(
    opening_strategy_4th = relevel(opening_strategy_4th, ref = "build_build"),
    opening_strategy_opp = relevel(opening_strategy_opp, ref = "build_build")
  )

model_df2 <- model_df2 %>%
  mutate(
    net_button    = scale(net_button),
    net_2ft       = scale(net_2ft),
    net_4ft       = scale(net_4ft),
    net_6ft       = scale(net_6ft),
    closest_diff  = scale(closest_diff),
    avg_points_4th = scale(avg_points_4th),
    avg_points_opp = scale(avg_points_opp)
  )

m_bin <- glmer(
  win_end ~
    opening_strategy_4th * pp_end +
    opening_strategy_opp +
    net_button + net_2ft + net_4ft + net_6ft +
    closest_diff +
    avg_points_4th + avg_points_opp +
    (1 | TeamID_4th) + (1 | OppTeamID),
  data = model_df2,
  family = binomial
)
```
</details>

To study the role of shot execution separately from board-position features, we also consider a **reduced GLMM** that retains execution quality but removes the engineered board-state variables. This specification helps isolate whether opening choice and execution alone provide explanatory power independent of the more detailed positional measures.

<details>
<summary><strong>Reduced GLMM</strong></summary>

<br>

```r
m_exec_only <- glmer(
  win_end ~
    opening_strategy_4th * pp_end +
    opening_strategy_opp +
    avg_points_4th + avg_points_opp +
    (1 | TeamID_4th) + (1 | OppTeamID),
  data = model_df2,
  family = binomial
)
```
</details>

Because some relationships may not be strictly linear—particularly those involving execution scores or small differences in stone placement—we incorporate a **generalized additive model (GAM)**. GAMs preserve interpretability while allowing smooth nonlinear effects, making them well suited for examining whether improvements in execution or stone position exhibit diminishing returns, threshold effects, or other nonlinear patterns that a parametric model might miss.

<details>
<summary><strong>Generalized Additive Model (GAM)</strong></summary>

<br>

```r
gam1 <- gam(
  win_end ~
    opening_strategy_4th * pp_end +
    opening_strategy_opp +
    net_button + net_2ft + net_4ft + net_6ft +  # linear terms
    s(avg_points_4th, k = 5) +
    s(avg_points_opp, k = 5) +
    s(closest_diff,   k = 5),
  data   = model_df2,
  family = binomial(link = "logit"),
  method = "REML"
)
```
</details>

To complement these interpretable models, we include a **gradient boosted decision tree model (XGBoost)**. Tree-based methods are adept at capturing complex interactions and nonlinearities without requiring manual specification. Although they are less interpretable, they offer a useful predictive benchmark and provide variable-importance summaries, which help identify the features most strongly associated with scoring in a flexible, nonparametric framework.

<details>
<summary><strong>XGBoost</strong></summary>

<br>

```r
# Start from model_df2
boost_df <- model_df2 %>%
  select(
    win_end,
    opening_strategy_4th, opening_strategy_opp,
    pp_end, PowerPlay,
    net_button, net_2ft, net_4ft, net_6ft,
    closest_diff,
    avg_points_4th, avg_points_opp
  )

# Ensure outcome is numeric 0/1
boost_df$win_end <- as.numeric(boost_df$win_end)

# One-hot encode factors (opening strategies, pp_end, PowerPlay)
dummies <- dummyVars(win_end ~ ., data = boost_df)
X <- predict(dummies, newdata = boost_df)
y <- boost_df$win_end


# train-test split
set.seed(479)

train_idx <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_test  <- X[-train_idx, ]
y_test  <- y[-train_idx]

dtrain <- xgb.DMatrix(as.matrix(X_train), label = y_train)
dtest  <- xgb.DMatrix(as.matrix(X_test),  label = y_test)

# Train
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 3,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8
)

watchlist <- list(
  train = dtrain,
  eval  = dtest
)

set.seed(479)
bst <- xgb.train(
  params  = params,
  data    = dtrain,
  nrounds = 400,
  watchlist = watchlist,
  early_stopping_rounds = 30,
  verbose = 1
)

# Evaluate and feature importance
pred_prob <- predict(bst, dtest)
roc_obj <- roc(y_test, pred_prob)
auc_val <- auc(roc_obj)
auc_val

imp <- xgb.importance(model = bst)
print(imp)
xgb.plot.importance(imp, top_n = 20)
```
</details>

Finally, because opening strategy may itself be influenced by team characteristics that also affect scoring, we incorporate a **propensity-score weighting model** to approximate a causal comparison between attack-first and build-first openings. This framework estimates the probability that a team selects an attack-oriented opening, uses those probabilities to reweight the data, and then fits a weighted outcome model to assess how strategy relates to scoring under better-balanced comparisons.

<details>
<summary><strong>Propensity Model</strong></summary>

<br>

```r
## Binary treatment: attack first vs build first
model_df2 <- model_df2 %>%
  mutate(
    treat_attack_first = ifelse(
      opening_strategy_4th %in% c("attack_attack", "attack_then_build"),
      1, 0
    )
  )

# Propensity model
ps_model <- glm(
  treat_attack_first ~ pp_end + PowerPlay + TeamID_4th + OppTeamID,
  family = binomial,
  data   = model_df2
)

# Compute stabilized weights
ps <- predict(ps_model, type = "response")

p_t <- mean(model_df2$treat_attack_first)

model_df2$w <- ifelse(
  model_df2$treat_attack_first == 1,
  p_t / ps,
  (1 - p_t) / (1 - ps)
)

# Trim extreme weights
cap <- quantile(model_df2$w, 0.99)
model_df2$w <- pmin(model_df2$w, cap)

# Weighted outcome model (causal estimate)
glm_w <- glm(
  win_end ~ treat_attack_first + opening_strategy_opp + pp_end +
            avg_points_4th + avg_points_opp,
  family  = binomial,
  data    = model_df2,
  weights = w
)

summary(glm_w)

# Effect size (odds ratio)
exp(coef(glm_w)["treat_attack_first"])
exp(confint(glm_w)["treat_attack_first", ])


## Compute propensity scores + plot
ps <- predict(ps_model, type = "response")
summary(ps)


hist(ps[model_df2$treat_attack_first == 1], breaks = 30, col = rgb(1,0,0,0.4),
     main = "Propensity Score Overlap", xlab = "Propensity")
hist(ps[model_df2$treat_attack_first == 0], breaks = 30, col = rgb(0,0,1,0.4), add = TRUE)
legend("topright", c("Attack-first", "Build-first"),
       fill = c(rgb(1,0,0,0.4), rgb(0,0,1,0.4)))


## Compute stabilized inverse probability weights
p_t <- mean(model_df2$treat_attack_first)

w <- ifelse(
  model_df2$treat_attack_first == 1,
  p_t / ps,
  (1 - p_t) / (1 - ps)
)

summary(w)

# Trimming extreme weights
cap <- quantile(w, 0.99)
w <- pmin(w, cap)

model_df2$w <- w
summary(model_df2$w)


## Fit weighted outcome model
glm_w <- glm(
  win_end ~
    treat_attack_first +
    opening_strategy_opp +
    pp_end +
    avg_points_4th + avg_points_opp,
  family  = binomial,
  data    = model_df2,
  weights = w
)

```
</details>

Taken together, these models allow us to examine strategy effectiveness from multiple angles. GLMMs control for team-level differences, the reduced model isolates execution effects, GAMs allow for flexible functional forms, XGBoost evaluates predictive structure, and propensity weighting provides an approximate causal perspective. Later sections compare their results and summarize their implications for opening-strategy decision making.

**Model Comparison Table**
| Model Type | Purpose | Strengths | Limitations |
|------------|---------|-----------|-------------|
| **GLMM** | Structured baseline relating strategy, execution, and board state to scoring | Controls for team effects; interpretable coefficients | Assumes linear relationships in predictors |
| **Reduced GLMM** | Isolate execution quality and strategy effects | Tests influence of execution independent of board-state features | Less comprehensive; board state omitted |
| **GAM** | Model nonlinear effects of execution and stone position | Flexible smooth terms; still interpretable | Higher complexity; may overfit if poorly tuned |
| **XGBoost** | Predictive benchmark and interaction discovery | Captures complex nonlinear patterns; strong predictive power | Low interpretability |
| **Propensity Weighting** | Approximate causal comparison of openings | Addresses confounding from strategy selection | Causal validity depends on correct specification of propensity model |





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

### 5 Appendix

#Some other codes or plots (dimension, visualization code, etc)


