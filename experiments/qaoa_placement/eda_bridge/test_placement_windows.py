from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.qaoa_placement.eda_bridge.placement_windows import (
    SCHEMA_VERSION,
    build_manifest,
    canonical_manifest_hash,
    parse_def,
    parse_lef,
    reinsert_window,
    write_gate_outputs,
)


HERE = Path(__file__).resolve().parent
TINY_DEF = HERE / "testdata/tiny.def"
TINY_LEF = HERE / "testdata/tiny.lef"


class PlacementWindowTests(unittest.TestCase):
    def test_parse_fixture(self) -> None:
        design = parse_def(TINY_DEF)
        macros = parse_lef(TINY_LEF, design.dbu_per_micron)
        self.assertEqual(design.name, "tiny")
        self.assertEqual(design.dbu_per_micron, 1000)
        self.assertEqual(len(design.components), 6)
        self.assertEqual(len(design.nets), 5)
        self.assertEqual(macros["INVX1"].width, 1000)

    def test_manifest_is_exact_and_hash_stable(self) -> None:
        source = {"repository": "fixture", "commit": "fixture", "platform": "test", "design": "tiny"}
        manifest = build_manifest(
            TINY_DEF, TINY_LEF, source, count=1, cells_per_window=4, sites_per_window=6, selection_seed=0
        )
        self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)
        self.assertEqual(manifest["manifest_hash"], canonical_manifest_hash(manifest))
        window = manifest["windows"][0]
        self.assertEqual(window["feasible_assignment_count"], 360)
        self.assertLessEqual(window["exact_hpwl_dbu"], window["initial_hpwl_dbu"])
        self.assertEqual(len(set(window["exact_best_assignment"].values())), 4)

    def test_reinsertion_and_outputs(self) -> None:
        manifest = build_manifest(
            TINY_DEF, TINY_LEF, {"repository": "fixture"}, count=1, cells_per_window=4, sites_per_window=6,
            selection_seed=0,
        )
        updated = reinsert_window(TINY_DEF.read_text(), manifest["windows"][0])
        reparsed = parse_def_from_text(updated)
        sites = {int(site["site_id"]): site for site in manifest["windows"][0]["candidate_sites"]}
        for cell_name, site_id in manifest["windows"][0]["exact_best_assignment"].items():
            component = reparsed.components[cell_name]
            self.assertEqual((component.x, component.y), (sites[int(site_id)]["x"], sites[int(site_id)]["y"]))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_gate_outputs(manifest, TINY_DEF, output, TINY_LEF)
            summary = json.loads((output / "gate_summary.json").read_text())
            self.assertEqual(summary["window_count"], 1)
            self.assertTrue(summary["all_reinserted_objectives_match"])
            self.assertTrue(summary["all_full_design_deltas_match_local"])
            self.assertTrue((output / "reinserted/tiny_w0000.def").is_file())


def parse_def_from_text(text: str):
    with tempfile.NamedTemporaryFile("w", suffix=".def", delete=False) as handle:
        handle.write(text)
        path = Path(handle.name)
    try:
        return parse_def(path)
    finally:
        path.unlink()


if __name__ == "__main__":
    unittest.main()
