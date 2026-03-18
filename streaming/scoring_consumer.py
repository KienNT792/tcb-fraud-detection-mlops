from __future__ import annotations

import argparse
import json
import os
from typing import Any

from kafka import KafkaConsumer, KafkaProducer

from infrastructure.pipeline import PIPELINE_CONFIG

from ml_pipeline.src.inference import FraudDetector


def consume_and_score(
    bootstrap_servers: str,
    input_topic: str,
    output_topic: str,
    models_dir: str,
    processed_dir: str,
    once: bool,
) -> int:
    detector = FraudDetector(models_dir, processed_dir)
    consumer = KafkaConsumer(
        input_topic,
        bootstrap_servers=bootstrap_servers.split(","),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        consumer_timeout_ms=3000 if once else 0,
        value_deserializer=lambda payload: json.loads(payload.decode("utf-8")),
    )
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers.split(","),
        value_serializer=lambda payload: json.dumps(payload).encode("utf-8"),
    )

    processed = 0
    for message in consumer:
        prediction: dict[str, Any] = detector.predict_single(message.value)
        producer.send(output_topic, prediction)
        processed += 1
        if once and processed >= 1:
            break

    producer.flush()
    producer.close()
    consumer.close()
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consume transactions from Kafka and emit fraud scores.")
    parser.add_argument("--bootstrap-servers", default=PIPELINE_CONFIG.kafka.bootstrap_servers)
    parser.add_argument("--input-topic", default=PIPELINE_CONFIG.kafka.input_topic)
    parser.add_argument("--output-topic", default=PIPELINE_CONFIG.kafka.output_topic)
    parser.add_argument("--models-dir", default=str(PIPELINE_CONFIG.paths.models_dir))
    parser.add_argument("--processed-dir", default=str(PIPELINE_CONFIG.paths.processed_dir))
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    consume_and_score(
        bootstrap_servers=args.bootstrap_servers,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        models_dir=args.models_dir,
        processed_dir=args.processed_dir,
        once=args.once,
    )


if __name__ == "__main__":
    main()
