from __future__ import annotations

import unittest
import json
from unittest.mock import patch
from pathlib import Path
import tempfile

import numpy as np

from experiments.qaoa_placement.eda_bridge.build_real_benchmark import (
    add_cost_landscape,
    select_stratified_windows,
)
from experiments.qaoa_placement.eda_bridge.build_development_manifest import (
    build_development_manifest,
)
from experiments.qaoa_placement.eda_bridge.combine_finite_shot_cohorts import (
    analyze as analyze_finite_shot_cohort,
)
from experiments.qaoa_placement.eda_bridge.combine_downstream_classical_cohorts import (
    analyze_rows as analyze_downstream_cohorts,
)
from experiments.qaoa_placement.eda_bridge.develop_qaoa_protocol import (
    ProtocolConfig,
    objective_value,
    qaoa_probabilities,
)
from experiments.qaoa_placement.eda_bridge.placement_windows import (
    build_manifest,
    canonical_manifest_hash,
    parse_def,
    parse_lef,
)
from experiments.qaoa_placement.eda_bridge.run_real_benchmark import (
    EVALUATION_BUDGET,
    TOKEN_STATES,
    evaluate_window,
    load_landscape,
)
from experiments.qaoa_placement.eda_bridge.run_finite_shot_optimization import (
    measurement_seed,
    run_case as run_finite_shot_case,
)
from experiments.qaoa_placement.eda_bridge.prepare_timing_impact import (
    assignment_cost,
    initial_assignment,
    render_tcl as render_timing_tcl,
)
from experiments.qaoa_placement.eda_bridge.prepare_downstream_routing import (
    expected_cells,
    render_apply_tcl as render_routing_apply_tcl,
)
from experiments.qaoa_placement.eda_bridge.prepare_downstream_classical_comparison import (
    select_comparator_rows,
)
from experiments.qaoa_placement.eda_bridge.run_downstream_routing_case import (
    parse_congestion,
    slurm_metadata,
)
from experiments.qaoa_placement.eda_bridge.summarize_downstream_classical_comparison import (
    paired_summary,
)
from experiments.qaoa_placement.eda_bridge.select_finite_shot_readout import (
    load_and_validate,
)
from experiments.qaoa_placement.eda_bridge.test_placement_windows import TINY_DEF, TINY_LEF


class RealBenchmarkTests(unittest.TestCase):
    def window(self):
        manifest = build_manifest(
            TINY_DEF,
            TINY_LEF,
            {"repository": "fixture"},
            count=1,
            cells_per_window=4,
            sites_per_window=6,
            selection_seed=0,
        )
        design = parse_def(TINY_DEF)
        macros = parse_lef(TINY_LEF, design.dbu_per_micron)
        window = add_cost_landscape(design, macros, manifest["windows"][0])
        window["selection_stratum"] = "medium_connectivity"
        return window

    def test_cost_landscape_is_complete(self) -> None:
        landscape = load_landscape(self.window())
        self.assertEqual(len(landscape.assignments), 360)
        self.assertEqual(landscape.exact_cost, min(landscape.cost_by_assignment.values()))

    def test_stratification_is_balanced_and_disjoint(self) -> None:
        candidates = []
        for index in range(12):
            candidates.append(
                {
                    "window_id": f"candidate_{index}",
                    "incident_net_count": index,
                    "movable_cells": [
                        {"cell_id": f"cell_{4 * index + offset}"} for offset in range(4)
                    ],
                }
            )
        selected = select_stratified_windows(candidates, 6, 9)
        counts = {stratum: 0 for stratum in ("low_connectivity", "medium_connectivity", "high_connectivity")}
        cells: list[str] = []
        for window in selected:
            counts[window["selection_stratum"]] += 1
            cells.extend(cell["cell_id"] for cell in window["movable_cells"])
        self.assertEqual(set(counts.values()), {2})
        self.assertEqual(len(cells), len(set(cells)))

    def test_all_methods_run_on_exact_fixture(self) -> None:
        rows = evaluate_window(self.window())
        methods = {row["method"] for row in rows}
        self.assertEqual(
            methods,
            {"unchanged", "exact", "greedy", "random_search", "simulated_annealing", "token_qaoa"},
        )
        qaoa = [row for row in rows if row["method"] == "token_qaoa"]
        self.assertEqual(len(qaoa), 2)
        self.assertTrue(all(row["optimization_objective"] == "cvar_0.25" for row in qaoa))
        self.assertTrue(all(int(row["objective_evaluations"]) <= EVALUATION_BUDGET for row in qaoa))
        self.assertTrue(all(abs(float(row["optimal_probability"])) <= 1.0 for row in qaoa))

    def test_development_split_is_balanced_within_each_design_and_stratum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for design_key in ("alpha", "beta"):
                windows = []
                for stratum in ("low_connectivity", "medium_connectivity", "high_connectivity"):
                    for index in range(2):
                        windows.append(
                            {
                                "window_id": f"{design_key}_{stratum}_{index}",
                                "selection_stratum": stratum,
                            }
                        )
                payload = {
                    "source": {"design": design_key},
                    "selection_policy": {"fixture": True},
                    "windows": windows,
                }
                payload["manifest_hash"] = canonical_manifest_hash(payload)
                path = Path(directory) / f"{design_key}.json"
                path.write_text(json.dumps(payload))
                paths.append(path)
            manifest = build_development_manifest(paths, "fixture", 7)
        self.assertEqual(manifest["split_counts"], {"train": 6, "validation": 6})
        for design_key in ("alpha", "beta"):
            for stratum in ("low_connectivity", "medium_connectivity", "high_connectivity"):
                group = [
                    window
                    for window in manifest["windows"]
                    if window["window_id"].startswith(design_key)
                    and window["selection_stratum"] == stratum
                ]
                self.assertEqual({window["development_split"] for window in group}, {"train", "validation"})
        self.assertEqual(manifest["manifest_hash"], canonical_manifest_hash(manifest))

    def test_configurable_qaoa_distribution_remains_normalized(self) -> None:
        landscape = load_landscape(self.window())
        state_costs = np.asarray(
            [landscape.cost_by_assignment[state[:4]] for state in TOKEN_STATES],
            dtype=float,
        )
        initial_cost = landscape.cost_by_assignment[landscape.initial]
        span = max(float(np.max(np.abs(state_costs - initial_cost))), 1.0)
        config = ProtocolConfig(
            "fixture", "real_empty_priority", "empty_prioritized", 2, "cvar_beat", 5
        )
        probabilities = qaoa_probabilities(
            landscape,
            (state_costs - initial_cost) / span,
            config,
            np.asarray([0.2, 0.1, 0.3, 0.15]),
        )
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0)
        self.assertTrue(np.all(probabilities >= 0.0))

    def test_deployable_objectives_need_only_costs_and_initial_threshold(self) -> None:
        costs = np.asarray([8.0, 10.0, 12.0])
        probabilities = np.asarray([0.25, 0.5, 0.25])
        self.assertAlmostEqual(objective_value("expected", costs, probabilities, 10.0), 0.0)
        self.assertAlmostEqual(objective_value("beat_initial", costs, probabilities, 10.0), -0.25)
        self.assertTrue(np.isfinite(objective_value("cvar_beat", costs, probabilities, 10.0)))

    def test_finite_shot_objective_is_reproducible_and_resource_counted(self) -> None:
        window = self.window()
        window["development_split"] = "train"
        first = run_finite_shot_case(window, 128, 11)
        second = run_finite_shot_case(window, 128, 11)
        self.assertEqual(measurement_seed(11, 128), measurement_seed(11, 128))
        self.assertEqual(first["parameters"], second["parameters"])
        self.assertEqual(
            first["total_objective_shots"],
            first["objective_evaluations"] * 128,
        )
        self.assertAlmostEqual(
            float(first["probability_beating_initial"]),
            float(second["probability_beating_initial"]),
        )

    def test_replication_readout_uses_frozen_seeded_best_of_4096_rule(self) -> None:
        window = self.window()
        manifest = {"windows": [window]}
        rows = []
        for seed, cost in ((11, 20), (17, 18), (23, 18), (29, 19)):
            rows.append(
                {
                    "window_id": window["window_id"],
                    "seed": str(seed),
                    "objective_shots": "8192",
                    "graph": "ring",
                    "schedule": "reversed",
                    "p": "3",
                    "objective": "beat_initial",
                    "sampled_best_4096_hpwl_dbu": str(cost),
                    "sampled_best_4096_assignment": "[0, 1, 2, 3]",
                    "sampled_exact_hit_4096": "False",
                }
            )
        selected = load_and_validate(manifest, rows)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["seed"], "17")
        self.assertEqual(selected[0]["method"], "finite_shot_token_qaoa")

    def test_finite_shot_cohort_analysis_aggregates_seeds_before_windows(self) -> None:
        run_rows = []
        for seed, value in zip((11, 17, 23, 29), (0.2, 0.4, 0.6, 0.8)):
            run_rows.append(
                {
                    "window_id": "fixture_w0000",
                    "design": "fixture",
                    "seed": str(seed),
                    "objective_shots": "8192",
                    "graph": "ring",
                    "schedule": "reversed",
                    "p": "3",
                    "objective": "beat_initial",
                    "exact_opportunity_dbu": "10",
                    "probability_beating_initial": str(value),
                    "expected_best_64_gap_closure": str(value),
                    "expected_best_256_gap_closure": str(value),
                    "expected_best_4096_gap_closure": str(value),
                }
            )
        selected_rows = [
            {
                "window_id": "fixture_w0000",
                "design": "fixture",
                "exact_opportunity_dbu": "10",
                "sampled_exact_hit_4096": "True",
                "sampled_best_4096_gap_closure": "1.0",
            }
        ]
        result = analyze_finite_shot_cohort(run_rows, selected_rows, 7, 100)
        self.assertEqual(result["window_count"], 1)
        self.assertAlmostEqual(
            result["analytical_improving_windows"][
                "expected_best_256_gap_closure"
            ]["mean"],
            0.5,
        )
        self.assertEqual(result["selected_best_of_4096"]["exact_hit_count"], 1)

    def test_routing_case_records_forwarded_slurm_identity(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "QEDA_SLURM_JOB_ID": "123_7",
                "QEDA_SLURM_ARRAY_JOB_ID": "123",
                "QEDA_SLURM_ARRAY_TASK_ID": "7",
                "QEDA_SLURM_JOB_PARTITION": "general",
                "QEDA_SLURM_CPUS_PER_TASK": "2",
                "QEDA_SLURM_MEM_PER_NODE": "16384",
            },
            clear=False,
        ):
            metadata = slurm_metadata()
        self.assertEqual(metadata["job_id"], "123_7")
        self.assertEqual(metadata["array_task_id"], "7")
        self.assertEqual(metadata["memory_per_node_mb"], "16384")

    def test_timing_bundle_applies_fixed_assignment_and_restores_baseline(self) -> None:
        window = self.window()
        assignment = initial_assignment(window)
        cost = assignment_cost(window, assignment)
        row = {
            "window_id": window["window_id"],
            "solution_assignment": json.dumps(assignment),
            "best_hpwl_dbu": str(cost),
            "initial_hpwl_dbu": str(window["initial_hpwl_dbu"]),
            "selection_stratum": str(window["selection_stratum"]),
            "seed": "11",
        }
        rendered = render_timing_tcl(
            "fixture", {str(window["window_id"]): window}, [row]
        )
        self.assertIn("qeda_measure $output fixture __baseline_before__", rendered)
        self.assertIn("qeda_measure $output fixture __baseline_after__", rendered)
        self.assertIn("check_placement -verbose", rendered)
        self.assertIn("QEDA_TIMING_CASES fixture 1", rendered)

    def test_downstream_bundle_preserves_the_fixed_assignment(self) -> None:
        window = self.window()
        assignment = initial_assignment(window)
        expected = expected_cells(window, assignment)
        rendered = render_routing_apply_tcl(window, assignment)
        self.assertEqual(len(expected), 4)
        self.assertIn("check_placement -verbose", rendered)
        self.assertIn("write_db $::env(QEDA_OUTPUT_ODB)", rendered)

    def test_congestion_parser_uses_the_final_total(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "5_1_grt.log"
            path.write_text(
                "Total 100 20 20.00% 1 / 2 / 3\n"
                "Total 200 30 15.00% 0 / 0 / 0\n"
            )
            parsed = parse_congestion(path)
        self.assertEqual(parsed["resource"], 200)
        self.assertEqual(parsed["total_overflow"], 0)

    def test_classical_comparator_selection_uses_frozen_best_annealing_seed(self) -> None:
        rows = []
        for index in range(36):
            window_id = f"fixture_w{index:04d}"
            common = {
                "window_id": window_id,
                "best_hpwl_dbu": "10",
                "solution_assignment": "[0,1,2,3]",
            }
            rows.extend(
                [
                    common | {"method": "exact", "seed": ""},
                    common | {"method": "greedy", "seed": ""},
                    common | {
                        "method": "simulated_annealing",
                        "seed": "11",
                        "best_hpwl_dbu": "9",
                    },
                    common | {
                        "method": "simulated_annealing",
                        "seed": "17",
                        "best_hpwl_dbu": "8",
                    },
                ]
            )
        selected = select_comparator_rows(rows)
        self.assertEqual(len(selected), 108)
        annealing = [row for row in selected if row["method"] == "simulated_annealing"]
        self.assertEqual({row["seed"] for row in annealing}, {"17"})

    def test_classical_pairing_uses_qaoa_minus_comparator_direction(self) -> None:
        qaoa = []
        classical = []
        for index in range(36):
            common = {
                "window_id": f"fixture_w{index:04d}",
                "design": ("aes", "gcd", "ibex")[index % 3],
            }
            qaoa.append(common | {"routed_wirelength": 9.0})
            classical.append(common | {"routed_wirelength": 10.0})
        result = paired_summary(
            qaoa,
            classical,
            "routed_wirelength",
            "lower",
            20260827,
            100,
        )
        self.assertEqual(result["mean_difference"], -1.0)
        self.assertEqual(result["qaoa_wins"], 36)
        self.assertEqual(result["qaoa_losses"], 0)

    def test_pooled_downstream_analysis_preserves_design_hierarchy(self) -> None:
        rows = []
        for design, offset in (("alpha", 0.0), ("beta", 10.0)):
            window_id = f"{design}_w0000"
            for method, delta in (
                ("qaoa", -2.0),
                ("exact", -2.0),
                ("greedy", -1.0),
                ("simulated_annealing", -3.0),
            ):
                row = {
                    "window_id": window_id,
                    "design": design,
                    "method": method,
                    "total_overflow": 0,
                    "retained_cell_count": 4,
                    "all_cells_retained": True,
                    "routed_wirelength": 100.0 + offset + delta,
                    "routed_wirelength_delta": delta,
                    "via_count": 20.0,
                    "via_count_delta": 0.0,
                    "congestion_demand": 30.0,
                    "congestion_demand_delta": 0.0,
                    "setup_wns_ns": 0.0,
                    "setup_wns_ns_delta": 0.0,
                    "setup_tns_ns": 0.0,
                    "setup_tns_ns_delta": 0.0,
                    "hold_wns_ns": 0.0,
                    "hold_wns_ns_delta": 0.0,
                    "hold_tns_ns": 0.0,
                    "hold_tns_ns_delta": 0.0,
                }
                rows.append(row)
        source = {
            "assignment_identity_with_qaoa": {
                method: {
                    "identical_assignment_count": 2,
                    "different_assignment_same_routed_wirelength_windows": [],
                }
                for method in ("exact", "greedy", "simulated_annealing")
            }
        }
        result = analyze_downstream_cohorts(rows, [source], 7, 100)
        self.assertEqual(result["design_count"], 2)
        self.assertEqual(result["window_count"], 2)
        self.assertEqual(result["routing_zero_overflow_count"], 8)
        exact = result["qaoa_pairwise"]["exact"]["routed_wirelength"]
        self.assertEqual(exact["ties"], 2)
        greedy = result["qaoa_pairwise"]["greedy"]["routed_wirelength"]
        self.assertEqual(greedy["mean_difference"], -1.0)


if __name__ == "__main__":
    unittest.main()
