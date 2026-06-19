#!/usr/bin/env python3
"""Compare keyword, semantic, and hybrid scores from one or two benchmark runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_multi_run(results_dir: Path, model_slug: str) -> Optional[Tuple[Path, Dict[str, Any]]]:
    files = sorted(
        results_dir.glob(f"results_{model_slug}_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    for path in reversed(files):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("scoring_methods") or data.get("total_scores"):
            return path, data
    return None


def _load_legacy_runs(
    results_dir: Path, model_slug: str
) -> Tuple[Optional[Path], Optional[Path]]:
    keyword_files = []
    semantic_files = []
    for path in sorted(
        results_dir.glob(f"results_{model_slug}_*.json"),
        key=lambda p: p.stat().st_mtime,
    ):
        data = json.loads(path.read_text(encoding="utf-8"))
        method = data.get("scoring_method", "")
        if method == "keyword":
            keyword_files.append(path)
        elif method == "semantic":
            semantic_files.append(path)
    return (
        keyword_files[-1] if keyword_files else None,
        semantic_files[-1] if semantic_files else None,
    )


def format_multi_comparison(path: Path, data: Dict[str, Any], *, title: str) -> str:
    methods: List[str] = data.get("scoring_methods") or list(
        (data.get("total_scores") or {}).keys()
    )
    total_scores: Dict[str, float] = data.get("total_scores") or {}
    rows = data.get("results", [])

    lines: List[str] = []
    w = lines.append
    w("=" * 78)
    w(f"MULTI-SCORER COMPARISON — {title}")
    w("=" * 78)
    w(f"File:   {path}")
    w(f"Model:  {data.get('model', '?')}")
    w(f"Methods: {', '.join(methods)}")
    w("")
    w(f"{'Method':<12} {'Total':>8} {'Interpretation'}")
    w("-" * 78)
    interpretations = data.get("interpretations") or {}
    for method in methods:
        total = total_scores.get(method, 0.0)
        interp = interpretations.get(method, data.get("interpretation", "?"))
        w(f"{method:<12} {total:>7.1f}% {interp}")
    w("")
    header = f"{'Q#':<3} {'Category':<22} "
    header += "".join(f"{method[:8]:>9}" for method in methods)
    header += "  Notes"
    w(header)
    w("-" * 78)

    for row in rows:
        scores = row.get("scores") or {}
        values = [scores.get(method, row.get("score", 0)) for method in methods]
        spread = max(values) - min(values) if values else 0
        notes = "agree" if spread == 0 else f"spread {spread}"
        line = f"{row['id']:<3} {row['category']:<22} "
        line += "".join(f"{scores.get(method, row.get('score', 0)):>9}" for method in methods)
        line += f"  {notes}"
        w(line)
    w("=" * 78)
    return "\n".join(lines)


def format_legacy_comparison(
    keyword_path: Path,
    semantic_path: Path,
    *,
    title: str,
    model: str,
) -> str:
    kw_data = json.loads(keyword_path.read_text(encoding="utf-8"))
    sem_data = json.loads(semantic_path.read_text(encoding="utf-8"))
    kw_rows = {int(r["id"]): r for r in kw_data.get("results", [])}
    sem_rows = {int(r["id"]): r for r in sem_data.get("results", [])}
    ids = sorted(set(kw_rows) | set(sem_rows))

    lines = [
        "=" * 78,
        f"KEYWORD vs SEMANTIC COMPARISON — {title}",
        "=" * 78,
        f"Keyword:  {keyword_path.name}",
        f"Semantic: {semantic_path.name}",
        "",
        f"{'Metric':<22} {'Keyword':>12} {'Semantic':>12}",
        f"{'Total score':<22} {kw_data.get('total_score', 0):>11.1f}% {sem_data.get('total_score', 0):>11.1f}%",
        "",
        "NOTE: separate benchmark runs — responses may differ.",
        "=" * 78,
    ]
    for qid in ids:
        kw = kw_rows.get(qid, {})
        sem = sem_rows.get(qid, {})
        lines.append(
            f"Q{qid:<2} {kw.get('category', sem.get('category', '?')):<22} "
            f"{kw.get('score', '—'):>4} {sem.get('score', '—'):>4}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-slug", default="tongyi-deepresearch-iq2s")
    parser.add_argument("--title", default=None)
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    results_dir = repo / "results"
    title = args.title or args.model_slug
    out_path = (
        Path(args.output)
        if args.output
        else results_dir / f"overnight_{args.model_slug}_multi_scorer_comparison.txt"
    )

    multi = _load_multi_run(results_dir, args.model_slug)
    if multi:
        path, data = multi
        report = format_multi_comparison(path, data, title=title)
    else:
        kw_path, sem_path = _load_legacy_runs(results_dir, args.model_slug)
        if not kw_path or not sem_path:
            print("No multi-scorer or legacy keyword+semantic results found.", file=sys.stderr)
            return 1
        report = format_legacy_comparison(
            kw_path,
            sem_path,
            title=title,
            model=args.model_slug,
        )

    print(report)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
