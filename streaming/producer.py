from __future__ import annotations

import argparse
import json
import os
from typing import Any

from kafka import KafkaProducer

from infrastructure.pipeline import PIPELINE_CONFIG

from scripts.bootstrap_demo_artifacts import make_demo_raw_df


def build_sample_payloads(count: int) -> list[dict[str, Any]]:
    df = make_demo_raw_df(n=max(count, 1)).head(count)
    return df.to_dict(orient="records")


def produce_transactions(
    bootstrap_servers: str,
    topic: str,
    count: int,
) -> int:
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers.split(","),
        value_serializer=lambda payload: json.dumps(payload).encode("utf-8"),
    )

    for payload in build_sample_payloads(count):
        producer.send(topic, payload)

    producer.flush()
    producer.close()
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce demo transactions into Kafka.")
    parser.add_argument("--bootstrap-servers", default=PIPELINE_CONFIG.kafka.bootstrap_servers)
    parser.add_argument("--topic", default=PIPELINE_CONFIG.kafka.input_topic)
    parser.add_argument("--count", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    produce_transactions(args.bootstrap_servers, args.topic, args.count)


if __name__ == "__main__":
    main()
