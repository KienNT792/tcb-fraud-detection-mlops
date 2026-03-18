from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

from infrastructure.pipeline import PIPELINE_CONFIG

PROJECT_DIR = str(PIPELINE_CONFIG.airflow.project_dir)
PYTHONPATH_EXPORT = f"export PYTHONPATH={PROJECT_DIR}; cd {PROJECT_DIR}"
RAW_DATA_PATH = str(PIPELINE_CONFIG.paths.demo_raw_path)
VALIDATED_DIR = str(PIPELINE_CONFIG.paths.validated_dir)
PROCESSED_DIR = str(PIPELINE_CONFIG.paths.processed_dir)
FEATURE_READY_DIR = str(PIPELINE_CONFIG.paths.feature_ready_dir)
MODELS_DIR = str(PIPELINE_CONFIG.paths.models_dir)
DRIFT_OUTPUT_DIR = str(PIPELINE_CONFIG.paths.drift_output_dir)
DEPLOYMENT_MANIFEST_PATH = str(PIPELINE_CONFIG.paths.deployment_manifest_path)

DEFAULT_ARGS = {
    "owner": PIPELINE_CONFIG.airflow.owner,
    "depends_on_past": False,
    "retries": PIPELINE_CONFIG.airflow.retries,
    "retry_delay": PIPELINE_CONFIG.airflow.retry_delay,
}


def cmd(command: str) -> str:
    return f"{PYTHONPATH_EXPORT} && {command}"


with DAG(
    dag_id="fraud_training_pipeline",
    description="Validate, preprocess, train, evaluate, register and prepare deployment for fraud model",
    schedule=PIPELINE_CONFIG.airflow.training_schedule,
    start_date=PIPELINE_CONFIG.airflow.start_date,
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["fraud", "training", "mlops"],
) as training_dag:
    data_validation_task = BashOperator(
        task_id="data_validation_task",
        bash_command=cmd(
            "python -m ml_pipeline.src.validate_data "
            f"--data-path {RAW_DATA_PATH} --validated-dir {VALIDATED_DIR}"
        ),
    )

    preprocessing_task = BashOperator(
        task_id="preprocessing_task",
        bash_command=cmd(
            "python -c \"from ml_pipeline.src.preprocess import run_preprocessing; "
            f"run_preprocessing(r'{RAW_DATA_PATH}', r'{PROCESSED_DIR}')\""
        ),
    )

    feature_engineering_task = BashOperator(
        task_id="feature_engineering_task",
        bash_command=cmd(
            "python -m ml_pipeline.src.materialize_feature_ready "
            f"--processed-dir {PROCESSED_DIR} --output-dir {FEATURE_READY_DIR}"
        ),
    )

    training_task = BashOperator(
        task_id="training_task",
        bash_command=cmd(
            "python -m ml_pipeline.src.train "
        ),
    )

    evaluation_task = BashOperator(
        task_id="evaluation_task",
        bash_command=cmd(
            "python -m ml_pipeline.src.evaluate "
        ),
    )

    register_model_task = BashOperator(
        task_id="register_model_task",
        bash_command=cmd(
            "python -m ml_pipeline.src.register_model "
            f"--models-dir {MODELS_DIR} --promote-to Staging --min-recall 0.95"
        ),
    )

    deploy_task = BashOperator(
        task_id="deploy_task",
        bash_command=cmd(
            "python -m ml_pipeline.src.generate_deployment_manifest "
            f"--models-dir {MODELS_DIR} "
            f"--output-path {DEPLOYMENT_MANIFEST_PATH}"
        ),
    )

    data_validation_task >> preprocessing_task >> feature_engineering_task
    feature_engineering_task >> training_task >> evaluation_task >> register_model_task >> deploy_task


with DAG(
    dag_id="fraud_monitoring_pipeline",
    description="Run scheduled drift and service monitoring checks",
    schedule=PIPELINE_CONFIG.airflow.monitoring_schedule,
    start_date=PIPELINE_CONFIG.airflow.start_date,
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["fraud", "monitoring"],
) as monitoring_dag:
    monitoring_task = BashOperator(
        task_id="monitoring_task",
        bash_command=cmd(
            "python monitoring/evidently_ai/drift_monitor.py "
            f"--reference-path {PROCESSED_DIR}/train.parquet "
            f"--current-path {PROCESSED_DIR}/test.parquet "
            f"--output-dir {DRIFT_OUTPUT_DIR}"
        ),
    )


with DAG(
    dag_id="fraud_batch_scoring_pipeline",
    description="Run batch scoring every 15 minutes for near real-time feeds",
    schedule=PIPELINE_CONFIG.airflow.batch_schedule,
    start_date=PIPELINE_CONFIG.airflow.start_date,
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["fraud", "batch-scoring"],
) as batch_scoring_dag:
    batch_scoring_task = BashOperator(
        task_id="batch_scoring_task",
        bash_command=cmd(
            "python -m ml_pipeline.src.batch_score "
            f"--input-path {PIPELINE_CONFIG.paths.batch_input_path} "
            f"--output-dir {PIPELINE_CONFIG.paths.scored_dir}"
        ),
    )
