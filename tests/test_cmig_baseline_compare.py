from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/cmig_baseline_compare.py"
SPEC = importlib.util.spec_from_file_location("cmig_baseline_compare", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
H = MODULE.HARNESS


def synthetic_rows():
    rows = []
    for finding in ("edema", "pneumonia"):
        for index, label in enumerate(MODULE.LABELS):
            key = [f"/fake/prior-{index}.jpg", "/fake/current.jpg", finding]
            number = len(rows)
            rows.append({"row_id": f"{number:06d}-{H.digest_json(key)[:24]}", "row_index": number,
                         "physical_key": key, "physical_key_sha256": H.digest_json(key),
                         "prior_image_path": key[0], "current_image_path": key[1],
                         "query_finding": finding, "target_label": label,
                         "patient_id": f"p{index}", "source_id": f"source-{index}", "pair_id": f"pair-{index}"})
    return rows


def synthetic_records(rows, predictions):
    result = []
    for row, prediction in zip(rows, predictions):
        text = prediction if prediction else "bad explanation"
        result.append({**row, "pred_label": prediction, "raw_text": text,
                       "parse_error": None if prediction else "invalid", "eos_observed": True,
                       "truncated": False, "generated_token_ids": [1], "generated_tokens": 1,
                       "protocol_sha256": "replaced", "prior_image_sha256": "a" * 64,
                       "current_image_sha256": "b" * 64, "historical_parse_success": 1})
    return result


def synthetic_protocol():
    return {"protocol_id": "test", "option_mapping": {"A": "improving", "B": "stable", "C": "worsening"},
            "max_new_tokens": 128, "question_template": "{finding}", "system_prompt": "",
            "content_layout": ["prior_image", "current_image", "question"],
            "provenance_status": "test", "source_reference": "test", "frozen_before_inference": True}


def fingerprint():
    files = [{"name": "model.safetensors", "bytes": 1, "sha256": "c" * 64}]
    return {"path": "/fake/model", "files": files, "files_sha256": H.digest_json(files)}


class CompareTests(unittest.TestCase):
    def test_vectorized_metrics_match_frozen_scorer(self):
        rows = synthetic_rows()
        records = synthetic_records(rows, ["improving", "stable", None, "stable", "worsening", "worsening"])
        counts, _, _, _ = MODULE.patient_confusions(rows, {"test": records})
        actual = MODULE.metrics_from_counts(counts.sum(axis=0))[0]
        expected = H.score_predictions(rows, records)
        self.assertAlmostEqual(actual[0], expected["accuracy"])
        self.assertAlmostEqual(actual[1], expected["supported_macro_f1"])
        self.assertAlmostEqual(actual[2], expected["class_balanced_accuracy"])
        self.assertAlmostEqual(actual[3], expected["mean_per_finding_class_balanced_accuracy"])

    def test_nse_is_error_not_extra_macro_f1_class(self):
        rows = synthetic_rows()
        records = synthetic_records(rows, ["improving", "stable", "no_supported_evidence"] * 2)
        counts, _, _, _ = MODULE.patient_confusions(rows, {"test": records})
        actual = MODULE.metrics_from_counts(counts.sum(axis=0))[0]
        np.testing.assert_allclose(actual, [2 / 3] * 4)

    def test_pooled_and_finding_averaged_balanced_accuracy_differ(self):
        counts = np.zeros((2, 3, 5), dtype=np.int64)
        counts[0, 0, 0], counts[0, 1, 1], counts[0, 2, 2] = 90, 90, 90
        counts[1, 0, 1], counts[1, 1, 2], counts[1, 2, 0] = 1, 1, 1
        actual = MODULE.metrics_from_counts(counts)
        self.assertAlmostEqual(actual[2], 90 / 91)
        self.assertAlmostEqual(actual[3], 0.5)

    def test_pairing_makes_identical_arm_contrast_exactly_zero(self):
        rows = synthetic_rows()
        records = synthetic_records(rows, ["stable"] * len(rows))
        counts, _, _, _ = MODULE.patient_confusions(rows, {"a": records, "b": records})
        samples = MODULE.paired_bootstrap(counts, replicates=75, seed=123, batch_size=17)
        np.testing.assert_array_equal(samples[:, 0] - samples[:, 1], 0)
        np.testing.assert_array_equal(samples, MODULE.paired_bootstrap(counts, replicates=75, seed=123, batch_size=17))

    def test_three_run_mean_is_mean_metric_not_prediction_ensemble(self):
        rows = synthetic_rows()
        gold = [r["target_label"] for r in rows]
        predictions = [gold, ["stable"] * len(rows), ["worsening"] * len(rows)]
        arms = {"base_native": synthetic_records(rows, ["improving"] * len(rows))}
        for name, pred in zip(MODULE.NEW_ARMS[1:4], predictions):
            arms[name] = synthetic_records(rows, pred)
        summary = MODULE.build_summary(rows, arms, replicates=50, seed=1)
        per_run = [summary["arms"][name]["metrics"]["accuracy"]["estimate"] for name in MODULE.NEW_ARMS[1:4]]
        aggregate = summary["aggregates"]["qlace_native_three_runs"]
        self.assertAlmostEqual(aggregate["metrics"]["accuracy"]["estimate"], np.mean(per_run))
        self.assertAlmostEqual(aggregate["sample_sd"]["accuracy"], np.std(per_run, ddof=1))
        self.assertIn("conditional", aggregate["ci_scope"])

    def test_historical_alignment_uses_occurrences_not_source_lookup(self):
        rows = synthetic_rows()
        old = [{"source_id": r["source_id"], "pair_id": r["pair_id"], "patient_id": r["patient_id"],
                "query_finding": r["query_finding"], "target_label": r["target_label"], "pred_label": r["target_label"]} for r in rows]
        aligned = MODULE.align_historical(rows, old)
        self.assertEqual(len(aligned), 6)
        self.assertEqual([r["row_id"] for r in aligned], [r["row_id"] for r in rows])
        old[0]["target_label"] = "worsening"
        with self.assertRaises(ValueError):
            MODULE.align_historical(rows, old)

    def test_historical_missing_eos_is_unknown(self):
        rows = synthetic_rows()
        records = [{**r, "pred_label": "stable", "historical_parse_success": None} for r in rows]
        diagnostics = MODULE.diagnostic_counts(records, historical=True)
        self.assertIsNone(diagnostics["truncated_count"])
        self.assertIsNone(diagnostics["eos_observed_count"])
        self.assertIsNone(diagnostics["parse_failure_count"])

    def test_blinded_query_is_verified_before_canonical_grouping(self):
        rows = synthetic_rows()
        old = [{**row, "query_finding": "the unspecified finding", "pred_label": "stable"} for row in rows]
        aligned = MODULE.align_historical(rows, old, finding_blind=True)
        self.assertEqual([r["query_finding"] for r in aligned], [r["query_finding"] for r in rows])
        old[0]["query_finding"] = "edema"
        with self.assertRaises(ValueError):
            MODULE.align_historical(rows, old, finding_blind=True)

    def test_reduced_classifier_metadata_still_checks_order_and_gold(self):
        rows = synthetic_rows()
        old = [{"source_id": row["source_id"], "query_finding": row["query_finding"],
                "gold_label": row["target_label"], "pred_label": "stable"} for row in rows]
        self.assertEqual(len(MODULE.align_historical(rows, old, reduced_metadata=True)), len(rows))
        old[0]["source_id"] = "wrong"
        with self.assertRaises(ValueError):
            MODULE.align_historical(rows, old, reduced_metadata=True)

    def write_full_run(self, directory):
        rows, protocol = synthetic_rows(), synthetic_protocol()
        predictions = synthetic_records(rows, [r["target_label"] for r in rows])
        for row in predictions:
            row["protocol_sha256"] = H.digest_json(protocol)
        contract = {"smoke_only": False, "manifest_sha256": H.digest_json(rows),
                    "selected_row_ids": [r["row_id"] for r in rows], "protocol": protocol,
                    "protocol_sha256": H.digest_json(protocol), "script_sha256": "frozen",
                    "dtype": "bfloat16", "do_sample": False, "cpu_offload": False,
                    "attention_implementation": "eager", "use_cache": True,
                    "model": fingerprint(), "adapter": None}
        H.write_json(directory / "run_manifest.json", {"contract": contract})
        H.write_jsonl(directory / "predictions.jsonl", predictions)
        score = H.score_predictions(rows, predictions)
        score.update({"smoke_only": False, "n_full_manifest_rows": len(rows)})
        H.write_json(directory / "metrics.json", score)
        return rows, protocol, contract, predictions

    def test_complete_run_integrity_and_forged_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            rows, protocol, contract, _ = self.write_full_run(path)
            records, _ = MODULE.load_new_run(rows, path, protocol, None, "frozen")
            self.assertEqual(len(records), len(rows))
            contract["protocol_sha256"] = "wrong"
            H.write_json(path / "run_manifest.json", {"contract": contract})
            with self.assertRaises(ValueError):
                MODULE.load_new_run(rows, path, protocol, None, "frozen")

    def test_smoke_partial_and_inconsistent_metrics_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            rows, protocol, contract, predictions = self.write_full_run(path)
            contract["smoke_only"] = True
            H.write_json(path / "run_manifest.json", {"contract": contract})
            with self.assertRaises(ValueError):
                MODULE.load_new_run(rows, path, protocol, None, "frozen")
            contract["smoke_only"] = False
            H.write_json(path / "run_manifest.json", {"contract": contract})
            H.write_jsonl(path / "predictions.jsonl", predictions[:-1])
            with self.assertRaises(ValueError):
                MODULE.load_new_run(rows, path, protocol, None, "frozen")
            H.write_jsonl(path / "predictions.jsonl", predictions)
            metrics = json.loads((path / "metrics.json").read_text())
            metrics["accuracy"] = 0.1
            H.write_json(path / "metrics.json", metrics)
            with self.assertRaises(ValueError):
                MODULE.load_new_run(rows, path, protocol, None, "frozen")

    def test_fingerprint_integrity_rejected(self):
        forged = fingerprint()
        forged["files"][0]["sha256"] = "d" * 64
        with self.assertRaises(ValueError):
            MODULE.check_fingerprint(forged)


if __name__ == "__main__":
    unittest.main()
