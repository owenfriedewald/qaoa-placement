"""Derive the current submission's claims, TeX numbers, and figures from raw rows."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MIXER = ROOT / "experiments/qaoa_placement/mixer_reduction"
CONFIRMATION = MIXER / "acm_confirmation_20260904"
ROUTING = MIXER / "acm_routing_qiskit252_20260904"
PHASE = MIXER / "exact_phase_completion_20260904"


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values, seed=904004):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    return np.percentile(values[rng.integers(len(values), size=(20000, len(values)))].mean(axis=1), [2.5, 97.5]).tolist()


def quality():
    rows = read(CONFIRMATION / "quality_run_level.csv")
    protocol = json.loads((CONFIRMATION / "protocol.json").read_text())
    manifest = json.loads((CONFIRMATION / "benchmark_manifest.json").read_text())["instances"]
    expected = {(i["instance_id"], m, start, str(seed)) for i in manifest
                for m in ("token", "penalty_row_xy") for start in protocol["initializations"]
                for seed in protocol["optimizer_seeds"]}
    actual = [(r["instance_id"], r["method"], r["init_mode"], r["optimizer_seed"]) for r in rows]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Incomplete or duplicated confirmation quality data")
    instances = []
    for instance in manifest:
        record = dict(instance_id=instance["instance_id"], family=instance["family"],
                      cells=instance["num_cells"], sites=instance["num_sites"])
        for method in ("token", "penalty_row_xy"):
            group = [r for r in rows if r["instance_id"] == instance["instance_id"] and r["method"] == method]
            for metric in ("optimal_probability", "feasible_probability", "improvement_probability", "evaluations"):
                record[f"{method}_{metric}"] = float(np.mean([float(r[metric]) for r in group]))
        record["delta"] = record["token_optimal_probability"] - record["penalty_row_xy_optimal_probability"]
        instances.append(record)
    summaries = []
    for cells in (4, 5):
        group = [r for r in instances if r["cells"] == cells]
        values = np.asarray([r["delta"] for r in group])
        # Fixed equal family allocation: resample instances within each family.
        rng = np.random.default_rng(904000 + cells)
        sampled = []
        for family in sorted({r["family"] for r in group}):
            v = np.asarray([r["delta"] for r in group if r["family"] == family])
            sampled.append(v[rng.integers(len(v), size=(20000, len(v)))].mean(axis=1))
        ci = np.percentile(np.mean(sampled, axis=0), [2.5, 97.5])
        summary = dict(cells=cells, sites=group[0]["sites"], instances=len(group),
                       token_mean=float(np.mean([r["token_optimal_probability"] for r in group])),
                       penalty_mean=float(np.mean([r["penalty_row_xy_optimal_probability"] for r in group])),
                       mean_delta=float(values.mean()), positive=int(np.sum(values > 1e-12)),
                       negative=int(np.sum(values < -1e-12)), ci_low=float(ci[0]), ci_high=float(ci[1]),
                       token_legal=float(np.mean([r["token_feasible_probability"] for r in group])),
                       penalty_legal=float(np.mean([r["penalty_row_xy_feasible_probability"] for r in group])),
                       token_evaluations=float(np.mean([r["token_evaluations"] for r in group])),
                       penalty_evaluations=float(np.mean([r["penalty_row_xy_evaluations"] for r in group])))
        summaries.append(summary)
    write(PAPER / "tables/confirmation_quality_instances.csv", instances)
    write(PAPER / "tables/confirmation_quality_summary.csv", summaries)
    strata = []
    for cells in (4, 5):
        for start in protocol["initializations"]:
            selected = [r for r in rows if int(r["num_cells"]) == cells and r["init_mode"] == start]
            ids = sorted({r["instance_id"] for r in selected})
            means = {m: np.asarray([np.mean([float(r["optimal_probability"]) for r in selected
                                             if r["instance_id"] == i and r["method"] == m]) for i in ids])
                     for m in ("token", "penalty_row_xy")}
            differences = means["token"] - means["penalty_row_xy"]
            strata.append(dict(cells=cells, initialization=start, instances=len(ids),
                               token_mean=float(means["token"].mean()),
                               penalty_mean=float(means["penalty_row_xy"].mean()),
                               delta=float(differences.mean()), positive=int(np.sum(differences > 1e-12))))
    write(PAPER / "tables/confirmation_initialization_summary.csv", strata)
    curves = read(CONFIRMATION / "quality_budget_curves.csv")
    expected_curves = {(i["instance_id"], method, start, str(seed), str(budget))
                       for i in manifest for method in ("token", "penalty_row_xy", "uniform_feasible", "greedy")
                       for start in protocol["initializations"]
                       for seed in (protocol["optimizer_seeds"] if method in ("token", "penalty_row_xy") else [""])
                       for budget in protocol["readout_budgets"]}
    actual_curves = [(r["instance_id"], r["method"], r["init_mode"], r["optimizer_seed"], r["budget"]) for r in curves]
    if set(actual_curves) != expected_curves or len(actual_curves) != len(expected_curves):
        raise ValueError("Incomplete or duplicated budget curves")
    curve_summary = []
    for cells in (4, 5):
        for method in ("token", "penalty_row_xy", "uniform_feasible", "greedy"):
            for budget in protocol["readout_budgets"]:
                group = [r for r in curves if int(r["num_cells"]) == cells and r["method"] == method and int(r["budget"]) == budget]
                per_instance = [np.mean([float(r["expected_best_hpwl"]) for r in group if r["instance_id"] == i])
                                for i in sorted({r["instance_id"] for r in group})]
                # Normalize per-instance by its exact optimum before averaging across instances.
                optimum = {r["instance_id"]: float(r["exact_hpwl"]) for r in rows}
                ratios = [np.mean([float(r["expected_best_hpwl"]) / optimum[i] for r in group if r["instance_id"] == i])
                          for i in sorted({r["instance_id"] for r in group})]
                curve_summary.append(dict(cells=cells, method=method, budget=budget,
                                          mean_expected_best_hpwl=float(np.mean(per_instance)),
                                          mean_expected_best_ratio=float(np.mean(ratios))))
    write(PAPER / "tables/confirmation_budget_summary.csv", curve_summary)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7))
    for ax, cells in zip(axes, (4, 5)):
        group = [r for r in instances if r["cells"] == cells]
        delta = sorted(r["delta"] for r in group)
        ax.scatter(range(len(delta)), delta, s=15, color="#185c83")
        ax.axhline(0, color="black", lw=.7)
        ax.set_title(f"{cells} cells / {6 if cells == 4 else 7} sites")
        ax.set_xlabel("Instance, ordered by paired difference")
        ax.set_ylabel("Token − penalty optimum mass")
    fig.tight_layout(); fig.savefig(PAPER / "figures/fig_confirmation_quality.pdf"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7))
    for ax, cells in zip(axes, (4, 5)):
        for method, label in [("token", "Token"), ("penalty_row_xy", "Penalty Row-XY"),
                              ("uniform_feasible", "Uniform feasible"), ("greedy", "Greedy")]:
            group = [r for r in curve_summary if r["cells"] == cells and r["method"] == method]
            ax.plot([r["budget"] for r in group], [r["mean_expected_best_ratio"] for r in group], marker=".", label=label)
        ax.set_xscale("log", base=2); ax.set_title(f"{cells} cells")
        ax.set_xlabel("Final draws / greedy cost-call cap")
        ax.set_ylabel("Expected best / exact cost")
    axes[0].legend(fontsize=6); fig.tight_layout()
    fig.savefig(PAPER / "figures/fig_confirmation_budgets.pdf"); plt.close(fig)
    return summaries


def routing():
    summaries, instance_rows = [], []
    for cohort, expected in (("frozen", 32), ("fresh", 36)):
        rows = read(ROUTING / f"routing_{cohort}_run_level.csv")
        ids = sorted({r["instance_id"] for r in rows})
        if len(ids) != expected or len(rows) != expected * 40:
            raise ValueError(f"Incomplete {cohort} routing corpus")
        manifest_path = (ROOT / "experiments/qaoa_placement/paper_suite_results/audit_061026/benchmark_manifest.json"
                         if cohort == "frozen" else CONFIRMATION / "benchmark_manifest.json")
        expected_ids = {i["instance_id"] for i in json.loads(manifest_path.read_text())["instances"]
                        if int(i["num_cells"]) == 4 and int(i["num_sites"]) == 6}
        if set(ids) != expected_ids:
            raise ValueError("Routing instance set differs from frozen manifest")
        pairs = []
        for instance in ids:
            record = dict(cohort=cohort, instance_id=instance)
            for method in ("token", "penalty_row_xy"):
                group = [r for r in rows if r["instance_id"] == instance and r["method"] == method]
                if sorted(int(r["routing_seed"]) for r in group) != list(range(101, 121)):
                    raise ValueError("Missing/duplicated routing seed")
                if any(r["symbolic_parameters"] != "5" or r["preparation"] != "True" or
                       any(r[key] != "False" for key in ["first_phase", "row_penalties", "measurements"]) for r in group):
                    raise ValueError("Incompatible circuit policy")
                for metric in ("routed_ecr", "routed_depth", "routed_two_qubit_depth", "logical_cx"):
                    record[f"{method}_{metric}"] = float(np.median([float(r[metric]) for r in group]))
            pairs.append(record); instance_rows.append(record)
        value = dict(cohort=cohort, instances=expected, seeds=20)
        for metric in ("routed_ecr", "routed_depth", "routed_two_qubit_depth", "logical_cx"):
            a = np.asarray([r[f"token_{metric}"] for r in pairs]); b = np.asarray([r[f"penalty_row_xy_{metric}"] for r in pairs])
            value.update({f"token_{metric}_median": float(np.median(a)), f"penalty_{metric}_median": float(np.median(b)),
                          f"{metric}_relative_median": float(np.median((a-b)/b)),
                          f"{metric}_token_wins": int(np.sum(a < b)), f"{metric}_token_ties": int(np.sum(a == b)),
                          f"{metric}_mean_delta_ci_low": interval(a-b)[0], f"{metric}_mean_delta_ci_high": interval(a-b)[1]})
        summaries.append(value)
    write(PAPER / "tables/confirmation_routing_summary.csv", summaries)
    write(PAPER / "tables/confirmation_routing_instances.csv", instance_rows)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7))
    for ax, cohort in zip(axes, ("frozen", "fresh")):
        group = [r for r in instance_rows if r["cohort"] == cohort]
        x = [100*(r["token_routed_ecr"] / r["penalty_row_xy_routed_ecr"]-1) for r in group]
        y = [100*(r["token_routed_depth"] / r["penalty_row_xy_routed_depth"]-1) for r in group]
        ax.scatter(x, y, s=15, color="#185c83"); ax.axhline(0, color="black", lw=.7); ax.axvline(0, color="black", lw=.7)
        ax.set_title(f"{cohort.capitalize()} 4c/6s corpus"); ax.set_xlabel("ECR change (%)"); ax.set_ylabel("Depth change (%)")
    fig.tight_layout(); fig.savefig(PAPER / "figures/fig_confirmation_routing.pdf"); plt.close(fig)
    return summaries


def unseen_quality():
    """Provenance-based sensitivity: exclude the single pre-freeze smoke case.

    This is not an outcome filter. Keep the frozen all-instance endpoint too.
    """
    rows = read(PAPER / "tables/confirmation_quality_instances.csv")
    disclosure = json.loads((CONFIRMATION / "validation_case_disclosure.json").read_text())
    excluded = disclosure["validation_instance"]
    group = [r for r in rows if r["cells"] == "4" and r["instance_id"] != excluded]
    assert len(group) == 35
    rng = np.random.default_rng(904004)
    draws = np.zeros(20000)
    for family in sorted({r["family"] for r in group}):
        values = np.asarray([float(r["delta"]) for r in group if r["family"] == family])
        draws += values[rng.integers(len(values), size=(20000, len(values)))].sum(axis=1)
    ci = np.percentile(draws / len(group), [2.5, 97.5])
    delta = np.asarray([float(r["delta"]) for r in group])
    result = dict(cells=4, sites=6, instances=35,
                  excluded_validation_instance=excluded,
                  token_mean=float(np.mean([float(r["token_optimal_probability"]) for r in group])),
                  penalty_mean=float(np.mean([float(r["penalty_row_xy_optimal_probability"]) for r in group])),
                  mean_delta=float(delta.mean()), positive=int(np.sum(delta > 1e-12)),
                  negative=int(np.sum(delta < -1e-12)), ci_low=float(ci[0]), ci_high=float(ci[1]))
    write(PAPER / "tables/confirmation_unseen_sensitivity.csv", [result])
    return result


def phase():
    rows = read(PHASE / "run_level.csv")
    if len(rows) != 200:
        raise ValueError("Incomplete phase completion audit")
    summaries = []
    for m in (6, 8, 9, 12, 16):
        keyed = {(r["family"], r["repeat"], r["policy"]): r for r in rows if int(r["sites"]) == m}
        pairs = [(keyed[f, str(r), "zero"], keyed[f, str(r), "l1"]) for f in ("compact", "line", "l_shape", "sparse_scatter") for r in range(5)]
        zero = np.array([float(a["logical_cx"]) for a, b in pairs]); completed = np.array([float(b["logical_cx"]) for a, b in pairs])
        summaries.append(dict(sites=m, cases=len(pairs), zero_median=float(np.median(zero)),
                              completed_median=float(np.median(completed)),
                              relative_change_median=float(np.median((completed-zero)/zero)),
                              wins=int(np.sum(completed < zero)), ties=int(np.sum(completed == zero)),
                              losses=int(np.sum(completed > zero)),
                              max_error=max(float(r["legal_distance_max_error"]) for r in rows if int(r["sites"]) == m)))
    write(PAPER / "tables/exact_phase_completion_summary.csv", summaries)
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    for policy, label, offset, color in [("zero", "Zero extension", -.16, "#64748b"), ("l1", "Exact L1 completion", .16, "#16878b")]:
        values = [[float(r["logical_cx"]) for r in rows if r["policy"] == policy and int(r["sites"]) == m] for m in (6, 8, 9, 12, 16)]
        box = ax.boxplot(values, positions=np.arange(5)+offset, widths=.26, patch_artist=True)
        for patch in box["boxes"]: patch.set_facecolor(color)
        ax.plot([], [], color=color, label=label, lw=6)
    ax.set_xticks(range(5), [6, 8, 9, 12, 16]); ax.set_yscale("log")
    ax.set_xlabel("Sites"); ax.set_ylabel("Logical CX, one exact phase")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(PAPER / "figures/fig_exact_phase_completion.pdf"); plt.close(fig)
    return summaries


def main(draft=False):
    (PAPER / "figures").mkdir(exist_ok=True); (PAPER / "tables").mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    from build_integrated_phase import build as build_integrated
    integrated = [] if draft else build_integrated(PAPER, write)
    data = dict(integrated_phase=integrated, quality=quality(), unseen_quality=unseen_quality(), routing=[] if draft else routing(), phase=phase())
    (PAPER / "confirmation_summary.json").write_text(json.dumps(data, indent=2) + "\n")
    macros = [r"\newif\ifRoutingComplete", r"\RoutingCompletefalse" if draft else r"\RoutingCompletetrue"]
    poor = next(r for r in read(PAPER / "tables/confirmation_initialization_summary.csv")
                if r["cells"] == "5" and r["initialization"] == "poor")
    macros.append(f"\\newcommand{{\\FivePoorWins}}{{{poor['positive']}}}")
    for row, name in zip(data["quality"], ("Four", "Five")):
        for key, macro in [("token_mean", "Token"), ("penalty_mean", "Penalty"), ("mean_delta", "Delta"),
                           ("ci_low", "Low"), ("ci_high", "High")]:
            macros.append(f"\\newcommand{{\\{name}{macro}}}{{{row[key]:.4f}}}")
        macros.append(f"\\newcommand{{\\{name}Wins}}{{{row['positive']}}}")
    for key, macro in [("token_mean", "Token"), ("penalty_mean", "Penalty"), ("mean_delta", "Delta"),
                       ("ci_low", "Low"), ("ci_high", "High")]:
        macros.append(f"\\newcommand{{\\UnseenFour{macro}}}{{{data['unseen_quality'][key]:.4f}}}")
    macros.append(f"\\newcommand{{\\UnseenFourWins}}{{{data['unseen_quality']['positive']}}}")
    for row, name in zip(data["routing"], ("Frozen", "Fresh")):
        for key, macro in [("routed_ecr_relative_median", "ECRChange"), ("routed_depth_relative_median", "DepthChange")]:
            macros.append(f"\\newcommand{{\\{name}{macro}}}{{{100*row[key]:+.1f}\\%}}")
    for row in integrated:
        if row['metric'] in ('routed_ecr', 'routed_depth'):
            name = 'CompletedECRChange' if row['metric']=='routed_ecr' else 'CompletedDepthChange'
            macros.append(f"\\newcommand{{\\{name}}}{{{100*row['relative_median']:+.1f}\\%}}")
    (PAPER / "confirmation_numbers.tex").write_text("% Generated from raw confirmation rows.\n" + "\n".join(macros) + "\n")
    def tex_table(name, lines):
        specs = {
            "confirmation_quality.tex": ("lrrrrlr", r"Size & $N$ & Token & Row-XY & Difference & 95\% interval & Wins"),
            "confirmation_routing.tex": ("lrrrrrr", r"Cohort & $N$ & Token ECR & Row-XY ECR & ECR change & Depth change & ECR wins"),
            "exact_phase_completion.tex": ("rrrrr", r"Sites & Zero CX & Completed CX & Paired change & W/T/L"),
        }
        columns, header = specs[name]
        content = ["% Generated; do not edit.", r"\begin{tabular}{"+columns+"}", r"\toprule",
                   header+r"\\", r"\midrule", *lines, r"\bottomrule", r"\end{tabular}"]
        (PAPER / "tables" / name).write_text("\n".join(content) + "\n")
    tex_table("confirmation_quality.tex", [f"{r['cells']}c/{r['sites']}s & {r['instances']} & {r['token_mean']:.4f} & {r['penalty_mean']:.4f} & {r['mean_delta']:+.4f} & [{r['ci_low']:.4f}, {r['ci_high']:.4f}] & {r['positive']}/{r['instances']}\\\\" for r in data["quality"]])
    tex_table("confirmation_routing.tex", [f"{r['cohort'].capitalize()} & {r['instances']} & {r['token_routed_ecr_median']:,.1f} & {r['penalty_routed_ecr_median']:,.1f} & {100*r['routed_ecr_relative_median']:+.1f}\\% & {100*r['routed_depth_relative_median']:+.1f}\\% & {r['routed_ecr_token_wins']}/{r['instances']}\\\\" for r in data["routing"]])
    tex_table("exact_phase_completion.tex", [f"{r['sites']} & {r['zero_median']:,.0f} & {r['completed_median']:,.0f} & {100*r['relative_change_median']:+.1f}\\% & {r['wins']}/{r['ties']}/{r['losses']}\\\\" for r in data["phase"]])
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", action="store_true", help="Explicitly marked draft with no routing aggregate")
    main(draft=parser.parse_args().draft)
