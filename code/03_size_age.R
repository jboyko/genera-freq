##  03_size_age.R
##
##  1. Scatter: genus size vs. crown age
##  2. Proportional-odds (ordinal) logistic regression: crown age → size class
##  3. Weighted proportional-odds regression: same, but misclassifying
##     megadiverse genera penalised most, then big, medium, small least
##
##  Size classes use the same Clauset xmin + Head/Tail Breaks thresholds
##  as script 01.
##
##  Inputs:
##    data/770398df9fbf743cdadb51e63f2f4abffe6e8159/resource.csv
##    data/genus_crown_ages.csv   (from 02)
##
##  Outputs:
##    plots/fig_size_age.png
##    tables/table_logreg.csv

library(ggplot2)
library(MASS)       # polr
library(poweRlaw)
library(scales)

## ---- helper: Head/Tail Breaks ----------------------------------------------
head_tail_breaks <- function(values, max_head_fraction = 0.40) {
  levels  <- list()
  current <- values
  repeat {
    m    <- mean(current)
    head <- current[current > m]
    if (length(head) == 0 || length(head) / length(current) > max_head_fraction)
      break
    levels <- c(levels, list(list(threshold = m, head = head)))
    current <- head
  }
  levels
}

## ---- load ------------------------------------------------------------------
sizes_raw <- read.csv(
  "data/770398df9fbf743cdadb51e63f2f4abffe6e8159/resource.csv",
  stringsAsFactors = FALSE
)
sizes_raw <- sizes_raw[sizes_raw$role == "accepted", c("name", "species")]
colnames(sizes_raw) <- c("genus", "n_species")
sizes_raw <- sizes_raw[sizes_raw$n_species > 0, ]

crown_ages <- read.csv("data/genus_crown_ages.csv", stringsAsFactors = FALSE)

df <- merge(sizes_raw, crown_ages, by = "genus")
cat(sprintf("Genera after join: %d\n", nrow(df)))

## ---- derive size-class thresholds (mirrors script 01) ----------------------
pl  <- displ$new(df$n_species)
est <- estimate_xmin(pl)
xmin <- est$xmin

htb    <- head_tail_breaks(df$n_species[df$n_species >= xmin])
t_big  <- ceiling(htb[[1]]$threshold)
t_mega <- ceiling(htb[[2]]$threshold)

cat(sprintf("Thresholds — xmin: %d  big: %d  mega: %d\n", xmin, t_big, t_mega))

## ---- assign size class (ordered factor) ------------------------------------
df$class <- cut(
  df$n_species,
  breaks = c(0, xmin - 1, t_big - 1, t_mega - 1, Inf),
  labels = c("Small", "Medium", "Big", "Megadiverse"),
  right  = TRUE
)
df$class <- ordered(df$class, levels = c("Small", "Medium", "Big", "Megadiverse"))

cat("\nClass counts:\n")
print(table(df$class))

## ---- Figure: size vs crown age, coloured by class --------------------------
class_cols <- c(Small = "#BBBBBB", Medium = "#4C72B0",
                Big   = "#DD8452", Megadiverse = "#C44E52")

p <- ggplot(df, aes(x = crown_age, y = n_species, colour = class)) +
  geom_point(alpha = 0.35, size = 0.9) +
  geom_smooth(aes(group = 1), method = "lm",
              colour = "black", linewidth = 0.8, se = TRUE) +
  scale_y_log10(labels = comma) +
  scale_colour_manual(values = class_cols, name = "Size class") +
  labs(x = "Crown age (Ma)", y = "Species per genus (log scale)",
       title = "Genus size vs. crown age") +
  theme_classic(base_size = 11) +
  theme(plot.title      = element_text(size = 11),
        legend.position = "right")

out_fig <- "plots/fig_size_age.png"
dir.create("plots", showWarnings = FALSE)
ggsave(out_fig, p, width = 7, height = 4.5, dpi = 180)
cat(sprintf("Saved → %s\n", out_fig))

## ---- Proportional-odds logistic regression (unweighted) --------------------
fit_unw <- polr(class ~ crown_age, data = df, Hess = TRUE)

cat("\n--- Unweighted proportional-odds regression ---\n")
print(summary(fit_unw))

## p-values via normal approximation on t-values
ctable_unw <- coef(summary(fit_unw))
p_unw      <- 2 * pnorm(abs(ctable_unw[, "t value"]), lower.tail = FALSE)
cat("\nCoefficients with p-values:\n")
print(cbind(ctable_unw, "p value" = round(p_unw, 4)))

pred_unw <- predict(fit_unw, df)
cat("\nConfusion matrix (unweighted):\n")
print(table(Predicted = pred_unw, Actual = df$class))
cat(sprintf("Overall accuracy: %.3f\n", mean(as.character(pred_unw) == as.character(df$class))))

## ---- Interpretation: unweighted proportional-odds model --------------------
##
## COEFFICIENT: crown_age = 0.05601 (t = 20.36, p ≈ 0)
##   Each additional Ma of crown age increases the log-odds of belonging to a
##   higher size class by 0.056. As an odds ratio: exp(0.056) ≈ 1.058 — about
##   a 5.8% increase in odds per Ma. Slightly shallower than the weighted model
##   (0.069) because Small genera dominate the likelihood and dilute the signal
##   from the rare large classes.
##
## INTERCEPTS (threshold parameters):
##   Small|Medium    = 4.098  →  P(Small at age 0) = logistic(4.098) ≈ 0.984
##   Medium|Big      = 5.464  →  P(Small or Medium at age 0)   ≈ 0.996
##   Big|Megadiverse = 6.720  →  P(not Megadiverse at age 0)   ≈ 0.999
##
##   The intercepts are notably higher than in the weighted model, reflecting
##   the model's strong pull toward Small in the absence of upweighting.
##
##   Crossover ages (P(exceeding threshold) = 0.5):
##     Small  → Medium:      4.098 / 0.056 ≈  73 Ma
##     Medium → Big:         5.464 / 0.056 ≈  98 Ma
##     Big    → Megadiverse: 6.720 / 0.056 ≈ 120 Ma
##
##   These are older than the weighted model's crossovers (47, 60, 73 Ma),
##   because without upweighting the model needs substantially more age
##   evidence before it will predict a larger class.
##
## CONFUSION MATRIX:
##   The unweighted model is the only one to predict all four classes
##   (6 Medium, 4 Big, 4 Megadiverse predicted). However, none of the rare
##   genera are correctly recovered: all 6 predicted-Medium are actually Small,
##   all 4 predicted-Big are actually Small, and the 4 predicted-Megadiverse
##   are 3 Small + 1 Medium (0 true Megadiverse). All 32 actual Megadiverse
##   and all 78 actual Big genera are predicted as Small. The non-Small
##   predictions arise from old crown ages, not true class membership. The
##   96.4% overall accuracy is essentially meaningless — a model that predicts
##   Small for every genus would score 96.5%.

## ---- Weighted proportional-odds logistic regression ------------------------
## Observation weights: Megadiverse (8) > Big (4) > Medium (2) > Small (1)
weight_map <- c(Small = 1, Medium = 2, Big = 4, Megadiverse = 8)
df$w <- weight_map[as.character(df$class)]

fit_wtd <- polr(class ~ crown_age, data = df, weights = w, Hess = TRUE)

cat("\n--- Weighted proportional-odds regression ---\n")
print(summary(fit_wtd))

ctable_wtd <- coef(summary(fit_wtd))
p_wtd      <- 2 * pnorm(abs(ctable_wtd[, "t value"]), lower.tail = FALSE)
cat("\nCoefficients with p-values:\n")
print(cbind(ctable_wtd, "p value" = round(p_wtd, 4)))

pred_wtd <- predict(fit_wtd, df)
cat("\nConfusion matrix (weighted):\n")
print(table(Predicted = pred_wtd, Actual = df$class))
cat(sprintf("Overall accuracy: %.3f\n", mean(as.character(pred_wtd) == as.character(df$class))))

## ---- Save coefficient table ------------------------------------------------
fmt_coef <- function(fit, label) {
  ct         <- as.data.frame(coef(summary(fit)))
  ct$term    <- rownames(ct)
  ct$p_value <- 2 * pnorm(abs(ct[["t value"]]), lower.tail = FALSE)
  ct$model   <- label
  ct
}

out_tbl <- "tables/table_logreg.csv"
dir.create("tables", showWarnings = FALSE)
write.csv(rbind(fmt_coef(fit_unw, "unweighted"),
                fmt_coef(fit_wtd, "weighted")),
          out_tbl, row.names = FALSE)
cat(sprintf("\nSaved → %s\n", out_tbl))

## ---- Interpretation: weighted proportional-odds model ----------------------
##
## COEFFICIENT: crown_age = 0.069 (t = 33.9, p ≈ 0)
##   For each additional 1 Ma of crown age, the log-odds of belonging to a
##   higher size class (vs. all classes below it) increases by 0.069.
##   As an odds ratio: exp(0.069) ≈ 1.072 — roughly a 7% increase in odds
##   per Ma. Over 50 Ma that compounds to ~32× higher odds of being in a
##   larger class; over 100 Ma, ~1000×. The effect is highly significant and
##   applies uniformly across all three thresholds (proportional-odds assumption).
##
## INTERCEPTS (threshold parameters):
##   These are the log-odds of being at or below each threshold when
##   crown_age = 0 (i.e., a genus that just originated with a single species).
##
##   Small|Medium    = 3.275  →  P(Small at age 0)        ≈ logistic(3.275) ≈ 0.96
##   Medium|Big      = 4.133  →  P(Small or Medium at age 0) ≈ 0.98
##   Big|Megadiverse = 5.019  →  P(not Megadiverse at age 0) ≈ 0.993
##
##   A genus at crown age 0 has a 96% chance of being Small and only a 0.7%
##   chance of being Megadiverse — consistent with the idea that diversity
##   accumulates over time.
##
##   The crossover ages (where P(exceeding threshold) = 0.5) are:
##     Small  → Medium:      3.275 / 0.069 ≈  47 Ma
##     Medium → Big:         4.133 / 0.069 ≈  60 Ma
##     Big    → Megadiverse: 5.019 / 0.069 ≈  73 Ma
##
##   These are not predictions that a genus *will* be that size at that age,
##   but rather the ages at which the model's predicted probability of
##   exceeding each threshold crosses 50%.
##
## CONFUSION MATRIX:
##   The weighted model never predicts Medium or Big — it polarises into
##   Small vs. Megadiverse only. This is a consequence of the 8× weight on
##   Megadiverse: the decision boundary shifts far enough that moderately old
##   genera get labelled Megadiverse, but the two intermediate classes never
##   become the modal prediction for any genus. Of the 73 genera predicted as
##   Megadiverse, 60 are actually Small (old but species-poor relicts), 9
##   Medium, 4 Big, and 0 truly Megadiverse. All 32 actual Megadiverse genera
##   are still predicted Small — their crown ages are not old enough to
##   distinguish them from the large pool of old-but-small genera. This
##   confirms that crown age alone is insufficient to identify megadiverse
##   genera; additional predictors (e.g. diversification rate, geography,
##   clade identity) would be needed.

