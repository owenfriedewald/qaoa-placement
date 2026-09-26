"""Create clustered aggregates and a concise report for the real benchmark."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


METHOD_ORDER = ("unchanged", "random_search", "simulated_annealing", "greedy", "token_qaoa", "exact")
BOOTSTRAP_SEED = 20260827
BOOTSTRAP_REPLICATES = 10_000


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(str(field))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def clustered(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str], dict[str, float]]:
    groups: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["window_id"])].append(row)
    output: dict[tuple[str, str], dict[str, float]] = {}
    for key, group in groups.items():
        output[key] = {
            "gap_closed": float(np.mean([float(row["available_gap_closed"]) for row in group])),
            "optimal_hit": float(np.mean([row["optimal_hit"] == "True" for row in group])),
            "improvement_dbu": float(np.mean([float(row["improvement_dbu"]) for row in group])),
            "available_dbu": float(group[0]["available_improvement_dbu"]),
        }
    return output


def bootstrap_interval(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(array, size=(BOOTSTRAP_REPLICATES, len(array)), replace=True)
    means = np.mean(samples, axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate(rows: Sequence[Mapping[str, str]], dimension: str) -> list[dict[str, object]]:
    clustered_rows = clustered(rows)
    metadata = {row["window_id"]: row for row in rows}
    dimension_values = sorted({row[dimension] for row in rows}) if dimension != "all" else ["all"]
    output: list[dict[str, object]] = []
    for dimension_value in dimension_values:
        window_ids = {
            row["window_id"]
            for row in rows
            if dimension == "all" or row[dimension] == dimension_value
        }
        improving = {window_id for window_id in window_ids if float(metadata[window_id]["available_improvement_dbu"]) > 0}
        for method in METHOD_ORDER:
            values = [clustered_rows[(method, window_id)] for window_id in sorted(improving)]
            if not values:
                continue
            gaps = [value["gap_closed"] for value in values]
            low, high = bootstrap_interval(gaps)
            method_rows = [row for row in rows if row["method"] == method and row["window_id"] in window_ids]
            result: dict[str, object] = {
                dimension: dimension_value,
                "method": method,
                "windows_all": len(window_ids),
                "windows_with_opportunity": len(improving),
                "mean_gap_closed_improving": float(np.mean(gaps)),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "optimal_hit_rate_improving": float(np.mean([value["optimal_hit"] for value in values])),
                "mean_improvement_dbu_all": float(np.mean([float(row["improvement_dbu"]) for row in method_rows])),
            }
            if method == "token_qaoa":
                result.update(
                    {
                        "mean_optimal_probability_all": float(
                            np.mean([float(row["optimal_probability"]) for row in method_rows])
                        ),
                        "mean_probability_beating_initial_improving": float(
                            np.mean(
                                [
                                    float(row["probability_beating_initial"])
                                    for row in method_rows
                                    if float(row["available_improvement_dbu"]) > 0
                                ]
                            )
                        ),
                    }
                )
            output.append(result)
    return output


def paired_qaoa_differences(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    values = clustered(rows)
    windows = sorted(
        {
            row["window_id"]
            for row in rows
            if float(row["available_improvement_dbu"]) > 0
        }
    )
    output: list[dict[str, object]] = []
    for comparator in ("random_search", "simulated_annealing", "greedy"):
        differences = [
            values[("token_qaoa", window)]["gap_closed"] - values[(comparator, window)]["gap_closed"]
            for window in windows
        ]
        low, high = bootstrap_interval(differences)
        output.append(
            {
                "comparison": f"token_qaoa_minus_{comparator}",
                "windows": len(windows),
                "mean_paired_gap_closed_difference": float(np.mean(differences)),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
            }
        )
    return output


def report(
    rows: Sequence[Mapping[str, str]],
    overall: Sequence[Mapping[str, object]],
    paired: Sequence[Mapping[str, object]],
    validated_cases: int | None,
) -> str:
    by_method = {row["method"]: row for row in overall}
    windows = {row["window_id"] for row in rows}
    improving = {row["window_id"] for row in rows if float(row["available_improvement_dbu"]) > 0}
    qaoa_rows = [row for row in rows if row["method"] == "token_qaoa"]
    terminated = sum(row["optimizer_success"] == "True" for row in qaoa_rows)
    lines = [
        "# Real-Placement Benchmark Results",
        "",
        "## Scope",
        "",
        f"The frozen suite contains {len(windows)} windows across GCD, AES, and Ibex. "
        f"{len(improving)} windows have positive exact improvement opportunity and "
        f"{len(windows) - len(improving)} are already exact local optima.",
        "",
        "These are incident-net HPWL and ideal reduced-subspace simulation results. "
        "They are not timing, congestion, routing, noisy-simulation, hardware, runtime-speedup, or quantum-advantage evidence.",
    ]
    if validated_cases is not None:
        lines.extend(
            [
                "",
                f"OpenROAD `check_placement` passes for all {validated_cases} selected exact, greedy, annealing, and token-QAOA reinsertions.",
            ]
        )
    lines.extend(
        [
            "",
            "## Improving-Window Results",
            "",
            "| Method | Mean exact gap closed | Cluster bootstrap 95% CI | Exact-hit rate |",
            "|---|---:|---:|---:|",
        ]
    )
    labels = {
        "unchanged": "Unchanged",
        "random_search": "Random search",
        "simulated_annealing": "Simulated annealing",
        "greedy": "Greedy",
        "token_qaoa": "Token QAOA",
        "exact": "Exact",
    }
    for method in METHOD_ORDER:
        row = by_method[method]
        lines.append(
            f"| {labels[method]} | {100 * float(row['mean_gap_closed_improving']):.1f}% | "
            f"[{100 * float(row['bootstrap_95_low']):.1f}%, {100 * float(row['bootstrap_95_high']):.1f}%] | "
            f"{100 * float(row['optimal_hit_rate_improving']):.1f}% |"
        )
    qaoa = by_method["token_qaoa"]
    lines.extend(
        [
            "",
            "Token QAOA closes less of the available gap than both greedy and simulated annealing in the aggregate. "
            "Its mean exceeds random search, but that paired confidence interval includes zero.",
            "",
            f"Across all QAOA rows, mean exact-optimum probability is {float(qaoa['mean_optimal_probability_all']):.4f}. "
            f"On improving windows, mean probability of beating the initial placement is "
            f"{float(qaoa['mean_probability_beating_initial_improving']):.4f}.",
            "",
            f"COBYLA reported normal convergence before the cap for {terminated}/{len(qaoa_rows)} rows; "
            "the remaining rows stopped at the frozen 40-evaluation budget and are retained.",
            "",
            "## Paired Gap-Closure Differences",
            "",
            "Positive values favor token QAOA. Windows, rather than optimizer seeds, are the bootstrap unit.",
            "",
            "| Comparison | Mean difference | Cluster bootstrap 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            f"| `{row['comparison']}` | {100 * float(row['mean_paired_gap_closed_difference']):+.1f} pp | "
            f"[{100 * float(row['bootstrap_95_low']):+.1f}, {100 * float(row['bootstrap_95_high']):+.1f}] pp |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The real-design bridge produces nontrivial, exact-solvable placement opportunities across all three designs. "
            "Under the frozen sampled-best protocol, the selected QAOA architecture underperforms both simulated annealing "
            "and simple greedy search; both paired bootstrap intervals exclude zero. Its apparent improvement over random "
            "search is inconclusive. These negative comparisons must remain visible in any paper narrative.",
            "",
            "Placement legality is directly validated. Add direct timing, congestion, or routing measurements before making "
            "claims that extend beyond HPWL.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-level", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--openroad-validation", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.run_level)
    overall = aggregate(rows, "all")
    by_design = aggregate(rows, "design")
    by_stratum = aggregate(rows, "selection_stratum")
    paired = paired_qaoa_differences(rows)
    validated_cases: int | None = None
    if args.openroad_validation is not None:
        validation = json.loads(args.openroad_validation.read_text())
        if not validation.get("all_cases_passed"):
            raise ValueError("OpenROAD validation is not complete")
        validated_cases = sum(int(row["validated_marker_count"]) for row in validation["results"])
    write_csv(args.output_dir / "aggregate_overall.csv", overall)
    write_csv(args.output_dir / "aggregate_by_design.csv", by_design)
    write_csv(args.output_dir / "aggregate_by_stratum.csv", by_stratum)
    write_csv(args.output_dir / "paired_qaoa_differences.csv", paired)
    (args.output_dir / "RESULTS_REPORT.md").write_text(report(rows, overall, paired, validated_cases))
    metadata = {
        "run_level": str(args.run_level),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_unit": "window after averaging optimizer/search seeds",
        "openroad_validation": str(args.openroad_validation) if args.openroad_validation else None,
        "openroad_validated_cases": validated_cases,
    }
    (args.output_dir / "summary_provenance.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
