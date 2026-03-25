Bootstrap runtime bundle used only for the first deploy when MLflow Registry
does not yet have a model version in the configured deploy stage.

Source of truth after bootstrap remains MLflow Registry + MinIO artifacts.

Contents expected:
- `processed/features.json`
- `processed/customer_stats.parquet`
- `processed/segment_label_map.json`
- `processed/amount_median_train.json`
- `processed/categorical_maps.json`
