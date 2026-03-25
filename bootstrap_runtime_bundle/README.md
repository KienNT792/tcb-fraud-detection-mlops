Bootstrap runtime bundle used only for the first deploy when MLflow Registry
does not yet have a model version in the configured deploy stage.

Source of truth after bootstrap remains MLflow Registry + MinIO artifacts.

Contents expected:
- `mlflow_model/MLmodel`
- `mlflow_model/model.xgb`
- `mlflow_model/conda.yaml`
- `mlflow_model/python_env.yaml`
- `mlflow_model/requirements.txt`
- `processed/features.json`
- `processed/customer_stats.parquet`
- `processed/segment_label_map.json`
- `processed/amount_median_train.json`
- `processed/categorical_maps.json`
