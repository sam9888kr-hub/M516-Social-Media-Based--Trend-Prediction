# Social Media-Based Trend Prediction

### Pre-Publication Classification of Viral Posts on Social Media Using Machine Learning

**Student:** Satyaam Kumar Attri

**Student ID:** GH1042675
**Module:** M516 — Business Project in Big Data \& AI

## Project Overview

This project investigates whether a social media post can be classified as likely to go viral **using only information available before publication** — platform, content type, topic, language, region, hashtags, sentiment, and posting time — with no post-publication engagement data (views, likes, comments, shares) used as input.

## Dataset

* File: `social\_media\_viral\_content\_dataset.csv`
* 2,000 posts across 4 platforms, 6 content categories, 5 languages, and 5 regions
* 15 attributes including platform, content type, topic, language, region, hashtags, timestamps, sentiment score, engagement metrics, and the binary target `is\_viral`
* Supplied as part of the module; no external dataset repository link is available for it

## Pipeline

1. **Setup \& Data Loading** — load and inspect the CSV, check for missing values
2. **Exploratory Data Analysis** — class balance, distributions by platform/topic, correlation heatmap
3. **Feature Engineering** — extract `post\_month`, `post\_dayofweek`, `hashtag\_count`, and one-hot flags for the 8 most frequent hashtags
4. **Preprocessing** — `ColumnTransformer` with `StandardScaler` (numeric) and `OneHotEncoder` (categorical); 80/20 stratified train/test split (`random\_state=42`)
5. **Model Comparison** — 5-fold stratified cross-validation (F1) across five candidate algorithms:

   * Logistic Regression
   * Random Forest
   * Hist Gradient Boosting
   * SVM (RBF kernel)
   * XGBoost
6. **Hyperparameter Tuning** — `RandomizedSearchCV` (25 iterations) on Random Forest
7. **Evaluation** — accuracy, F1, precision, recall, ROC-AUC on the held-out test set, compared against a majority-class baseline
8. **Explainability** — Random Forest feature importances and SHAP (`TreeExplainer`) summary plots
9. **Deployment** — `predict\_virality()` function returning a virality probability for a new, unpublished post

## Results

|Metric|Tuned Random Forest|Majority-class baseline|
|-|-|-|
|Accuracy|0.700|0.700|
|F1-score|0.824|—|
|Precision|0.700|—|
|Recall|1.000|—|
|ROC-AUC|0.421|0.500|

Cross-validation F1 ranking (all five algorithms scored within \~4 points of each other): SVM (RBF) 0.822 > Random Forest 0.820 > Logistic Regression 0.820 > Hist Gradient Boosting 0.779 > XGBoost 0.756.

**Honest finding:** the tuned Random Forest's test-set confusion matrix shows it predicted "Viral" for all 400 test posts (0 true negatives, 120 false positives), and its ROC-AUC of 0.421 is below random chance. The F1 score of 0.824 is real but is an artifact of always predicting the majority class, not genuine discrimination between viral and non-viral posts. SHAP analysis confirms no feature carries a strong, consistent signal — pre-publication attributes in this dataset appear insufficient to reliably predict virality in advance.

## Why Random Forest Over SVM

SVM had the marginally higher CV F1 (0.822 vs. 0.820), but Random Forest was selected for tuning because it provides native feature importances and integrates directly with SHAP's `TreeExplainer`, which was a project requirement for the Explainability layer. The gap between the two was under one percentage point.

## Repository Contents

* `M516\_final.ipynb` — full notebook (EDA, modeling, evaluation, explainability, deployment)
* `M516\_final\_corrected.docx` — written project report
* `app.py` — deployment script wrapping the trained pipeline *(if included)*
* Video walkthrough: see link in the report / below

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
```

## Future Work

* Incorporate additional pre-publication signals (e.g. follower count, posting history, account age)
* Reframe as a regression problem (predicting an engagement/virality score) rather than binary classification
* Address the class-imbalance/majority-class collapse observed in the current Random Forest model

## Video Demonstration

\[Insert video link here]

