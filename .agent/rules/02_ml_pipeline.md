---
trigger: glob
globs: ml_pipeline/**/*.py
---

# ML Pipeline Rules (Data Science & Training)
When writing code inside the `ml_pipeline/` directory, adhere strictly to these rules:

# Tech Stack
- Data Manipulation: `pandas`, `numpy`
- Machine Learning: `scikit-learn`, `xgboost` or `lightgbm`
- Imbalanced Data: `imbalanced-learn` (SMOTE or Class Weights)
- Tracking & Registry: `mlflow`
- Explainability: `shap`

# Implementation Directives
1. MLflow Tracking: Every training script must wrap the training process in `with mlflow.start_run():`. Auto-log or manually log all hyperparameters, metrics (F1, PR-AUC), and save the model artifact.
2. Class Imbalance: Explicitly handle the 2.84% fraud rate. Use `scale_pos_weight` in XGBoost/LightGBM or apply SMOTE in the preprocessing step.
3. Feature Engineering: Process contextual features like `tx_count_last_1h` and `distance_from_home_km` appropriately.
4. Responsible AI: Always generate a SHAP summary plot during evaluation to explain feature importance. Calculate False Positive Rates specifically for the 'PRIVATE' and 'PRIORITY' customer segments to ensure fairness.