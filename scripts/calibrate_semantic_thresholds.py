#!/usr/bin/env python3
"""Calibrate semantic score-band thresholds against a live embedding backend."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from scoring.semantic_calibration import (
    calibrate_thresholds_from_similarities,
    degraded_reference_variants,
)
from scoring.semantic_embedder import (
    DEFAULT_DEEPINFRA_SEMANTIC_MODEL,
    create_semantic_embedder,
)
from scoring.semantic_scorer import (
    DEFAULT_SEMANTIC_ANSWERS_FILE,
    DEFAULT_SEMANTIC_MAX_SEQ_LENGTH,
    DEFAULT_SEMANTIC_MODEL,
    DEFAULT_SEMANTIC_PROVIDER,
    _cosine_similarity,
    parse_semantic_references,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate semantic similarity thresholds from answers_v2 references",
    )
    parser.add_argument(
        "--provider",
        choices=["local", "deepinfra"],
        default=DEFAULT_SEMANTIC_PROVIDER,
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--answers", default=DEFAULT_SEMANTIC_ANSWERS_FILE)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    provider = args.provider.lower()
    model_name = args.model or (
        DEFAULT_DEEPINFRA_SEMANTIC_MODEL
        if provider == "deepinfra"
        else DEFAULT_SEMANTIC_MODEL
    )
    api_key = args.api_key or os.environ.get("DEEPINFRA_TOKEN")
    embedder = create_semantic_embedder(
        provider=provider,
        model_name=model_name,
        max_seq_length=DEFAULT_SEMANTIC_MAX_SEQ_LENGTH,
        api_key=api_key,
    )

    references = parse_semantic_references(args.answers)
    buckets = {
        "exact": [],
        "high": [],
        "medium": [],
        "low": [],
        "minimal": [],
    }

    for reference in references.values():
        variants = degraded_reference_variants(reference)
        vectors = embedder.encode(list(variants.values()))
        by_name = dict(zip(variants.keys(), vectors))
        ref_vector = by_name["exact"]
        for name in ("exact", "high", "medium", "low", "minimal"):
            buckets[name].append(_cosine_similarity(ref_vector, by_name[name]))

    thresholds = calibrate_thresholds_from_similarities(
        exact=buckets["exact"],
        high=buckets["high"],
        medium=buckets["medium"],
        low=buckets["low"],
        minimal=buckets["minimal"],
    )

    payload = {
        "provider": provider,
        "model": model_name,
        "answers_file": args.answers,
        "thresholds": thresholds,
        "sample_counts": {name: len(values) for name, values in buckets.items()},
    }

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
