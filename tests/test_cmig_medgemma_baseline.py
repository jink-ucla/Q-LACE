from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/cmig_medgemma_baseline.py"
SPEC = importlib.util.spec_from_file_location("cmig_medgemma_baseline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def protocol():
    return {"protocol_id": "synthetic", "system_prompt": "",
            "question_template": "Change in {finding}? A improved B stable C worsened",
            "option_mapping": {"A": "improving", "B": "stable", "C": "worsening"},
            "content_layout": ["prior_annotation", "prior_image", "current_annotation", "current_image", "question"],
            "prior_annotation": "Prior:", "current_annotation": "Current:",
            "max_new_tokens": 16, "provenance_status": "synthetic-test-only",
            "source_reference": "none", "frozen_before_inference": True}


class BaselineTests(unittest.TestCase):
    def make_manifest(self, directory):
        source = Path(directory) / "source.jsonl"
        rows = []
        for index, label in enumerate(("improving", "stable", "worsening")):
            rows.append({"source_id": "same-source-id", "pair_id": "same-pair-id",
                         "patient_id": str(index // 2), "prior_image_path": f"/fake/prior-{index}.jpg",
                         "current_image_path": "/fake/current.jpg", "query_finding": "edema",
                         "query_change": label, "split": "test"})
        source.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return MODULE.build_manifest(source, expected_rows=3, check_images=False)

    def test_preserves_duplicate_source_id_and_distinct_priors(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, meta = self.make_manifest(directory)
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({r["row_id"] for r in rows}), 3)
            self.assertEqual(meta["duplicate_source_id_extra_rows"], 2)
            self.assertEqual(meta["n_physical_image_pairs"], 3)
            self.assertEqual([r["row_index"] for r in rows], [0, 1, 2])

    def test_exact_physical_duplicates_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            row = {"source_id": "x", "patient_id": "p", "prior_image_path": "/fake/p.jpg",
                   "current_image_path": "/fake/c.jpg", "query_finding": "edema", "query_change": "stable"}
            source.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            rows, meta = MODULE.build_manifest(source, expected_rows=2, check_images=False)
            self.assertNotEqual(rows[0]["row_id"], rows[1]["row_id"])
            self.assertEqual(meta["duplicate_physical_finding_extra_rows"], 1)

    def test_count_and_missing_images_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_manifest(directory)
            source = Path(directory) / "source.jsonl"
            with self.assertRaises(ValueError):
                MODULE.build_manifest(source, expected_rows=1326, check_images=False)
            with self.assertRaises(ValueError):
                MODULE.build_manifest(source, expected_rows=3)

    def test_strict_parser_and_configurable_mapping(self):
        mapping = protocol()["option_mapping"]
        for text, label in ((" A ", "improving"), ("(B).", "stable"), ("worsened", "worsening"), ("Improving.", "improving"), ("Final Answer: C", "worsening")):
            self.assertEqual(MODULE.parse_answer(text, mapping), (label, None))
        for text in ("A because edema decreased", "stable model stable", "A or B", "no_supported_evidence", "", '{"label":"stable"}', "improved/worsened"):
            self.assertIsNone(MODULE.parse_answer(text, mapping)[0])
        swapped = {"A": "worsening", "B": "stable", "C": "improving"}
        self.assertEqual(MODULE.parse_answer("A", swapped)[0], "worsening")

    def test_target_is_absent_from_messages(self):
        row = {"query_finding": "edema", "target_label": "secret-answer"}
        messages = MODULE.make_messages(protocol(), row, "prior sentinel", "current sentinel")
        self.assertEqual([v["type"] for v in messages[0]["content"]], ["text", "image", "text", "image", "text"])
        self.assertEqual(messages[0]["content"][1]["image"], "prior sentinel")
        self.assertNotIn("secret-answer", json.dumps(messages))

    def test_invalid_predictions_count_in_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _ = self.make_manifest(directory)
            predictions = [{**r, "pred_label": label} for r, label in zip(rows, ("improving", "stable", None))]
            score = MODULE.score_predictions(rows, list(reversed(predictions)))
            self.assertAlmostEqual(score["accuracy"], 2 / 3)
            self.assertAlmostEqual(score["supported_macro_f1"], 2 / 3)
            self.assertAlmostEqual(score["class_balanced_accuracy"], 2 / 3)
            self.assertEqual(score["invalid_count"], 1)

    def test_wrong_resume_rows_and_partial_results_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _ = self.make_manifest(directory)
            predictions = [{**r, "pred_label": "stable"} for r in rows]
            with self.assertRaises(ValueError):
                MODULE.score_predictions(rows, predictions[:2])
            with self.assertRaises(ValueError):
                MODULE.validate_predictions(rows, predictions + [predictions[0]])
            predictions[0]["physical_key"] = ["wrong", "wrong", "edema"]
            with self.assertRaises(ValueError):
                MODULE.validate_predictions(rows, predictions, allow_partial=True)

    def test_atomic_manifest_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _ = self.make_manifest(directory)
            path = Path(directory) / "manifest.jsonl"
            MODULE.write_jsonl(path, rows)
            self.assertEqual(MODULE.read_manifest(path, 3), rows)
            rows[0]["prior_image_path"] = "/wrong.jpg"
            MODULE.write_jsonl(path, rows)
            with self.assertRaises(ValueError):
                MODULE.read_manifest(path, 3)

    def test_protocol_must_be_frozen_and_bijective(self):
        self.assertEqual(MODULE.validate_protocol(protocol())["protocol_id"], "synthetic")
        invalid = protocol()
        invalid["frozen_before_inference"] = False
        with self.assertRaises(ValueError):
            MODULE.validate_protocol(invalid)
        invalid = protocol()
        invalid["option_mapping"]["A"] = "stable"
        with self.assertRaises(ValueError):
            MODULE.validate_protocol(invalid)


if __name__ == "__main__":
    unittest.main()
