"""Synthetic table/figure fixtures; no manuscript or private records required."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import verify_manuscript_results as results


def figure():
    rows = [{"key": "qlace", "label": "Synthetic reference", "group": "Synthetic",
             "no_acc": 0.4, "no_f1": 0.3, "pc_acc": 0.7, "pc_f1": 0.6}]
    rows += [{"key": f"synthetic_{i}", "label": f"Synthetic comparator {i}", "group": "Synthetic",
              "no_acc": 0.1, "no_f1": 0.2, "pc_acc": 0.2 + i * 0.01, "pc_f1": 0.3 + i * 0.01}
             for i in range(12)]
    return {"schema": "qlace-published-figure-data-v1", "label": "fig:synthetic", "number": "A1", "page": "3",
            "primary": {"accuracy": 0.7, "macro_f1": 0.6}, "panel_b": rows,
            "panel_c": {"reference_key": "qlace", "estimates": [
                {"key": row["key"], "label": row["label"], "accuracy_delta": 0.7 - row["pc_acc"],
                 "macro_f1_delta": 0.6 - row["pc_f1"]} for row in rows[1:]]},
            "verified_panel_counts": {"B": 13, "C": 12}}


def bundle():
    labels = {"tab:protocol-counts": {"number": "A1"}}
    tables = []
    for number, label, text, aliases in (
        ("1", "tab:label-semantics", r"Label & Meaning \\ stable & unchanged \\", ["tab:synthetic-alias"]),
        ("A1", "tab:protocol-counts", r"Population & Rows \\ Synthetic & 12 \\", []),
    ):
        cap = r"Synthetic table; compare Table~\ref{tab:protocol-counts}."
        tables.append({"label": label, "aliases": aliases, "number": number, "page": "2",
                       "caption_latex": cap, "caption_text": results.latex_text(cap, labels),
                       "tabulars": results.tabulars(r"\begin{tabular}{ll}" + text + r"\end{tabular}", labels)})
    context = {"model_figure": {"caption_text": "Synthetic model"}, "protocol_counts_label": "tab:protocol-counts"}
    snapshot = {"schema": "qlace-published-tables-v1", "table_count": 2, "tables": tables,
                "context": context, "reference_numbers": {"tab:protocol-counts": "A1"}}
    index = {"schema": "qlace-manuscript-results-index-v1", "table_count": 2,
             "snapshot": "aggregates/manuscript_tables.json", "context": context,
             "coverage_from_compiled_labels": [
                 {key: table[key] for key in ("label", "aliases", "number", "page")} for table in tables],
             "tables": [], "figures": [{"label": "fig:synthetic", "number": "A1", "page": "3",
                                        "data": "aggregates/figure_A1_data.json",
                                        "verified_panel_counts": {"B": 13, "C": 12}}],
             "prose_result_families": [{"topic": "Synthetic runtime context",
                                        "canonical_aggregate_families": ["aggregates/context.json"]}]}
    for table in tables:
        entry = {key: table[key] for key in ("label", "aliases", "number", "page")}
        entry.update({"csv": f"aggregates/manuscript_tables/table_{table['number']}.csv", "tabular_count": 1,
                      "classification": "specification" if table["number"] == "1" else "published_results_or_context_counts",
                      "canonical_aggregate_families": [] if table["number"] == "1" else ["aggregates/context.json"]})
        index["tables"].append(entry)
    return snapshot, index, figure()


def write_bundle(root, snapshot=None, index=None, graph=None):
    if snapshot is None:
        snapshot, index, graph = bundle()
    (root / "aggregates/manuscript_tables").mkdir(parents=True)
    source = results.encoded(snapshot)
    (root / "aggregates/manuscript_tables.json").write_bytes(source)
    index["snapshot_sha256"] = results.sha(source)
    for table, entry in zip(snapshot["tables"], index["tables"]):
        data = results.table_csv(table)
        (root / entry["csv"]).write_bytes(data)
        entry["csv_sha256"] = results.sha(data)
    data = results.encoded(graph)
    (root / "aggregates/figure_A1_data.json").write_bytes(data)
    index["figures"][0]["sha256"] = results.sha(data)
    (root / "aggregates/context.json").write_text("{}\n")
    (root / "aggregates/manuscript_results_index.json").write_bytes(results.encoded(index))
    return snapshot, index, graph


def test_brace_depth_preserves_multiline_cells_and_escaped_separators():
    rows = results.table_rows(r"\hline Head & Value \\ \shortstack{first\\second} & \parbox[t]{3cm}{a\&b\\c} \\[.2em] \hline")
    assert len(rows) == 2 and [len(row) for row in rows] == [2, 2]
    assert rows[1] == [r"\shortstack{first\\second}", r"\parbox[t]{3cm}{a\&b\\c}"]
    assert [results.latex_text(cell) for cell in rows[1]] == ["first\nsecond", "a&b\nc"]


def test_text_projection_preserves_numeric_strings_and_multicolumn_content():
    value = r"\multicolumn{3}{c}{\textbf{2{,}048} $\pm$ 0.014; 95\%}"
    assert results.latex_text(value) == "2,048 ± 0.014; 95%"
    assert results.latex_text(r"\texttt{no\_supported\_evidence}") == "no_supported_evidence"
    assert results.latex_text(r"$-.0021$ / $+0.1195$") == "-.0021 / +0.1195"


def test_multiple_tabulars_and_tabularx_width_not_exported_as_cells():
    body = r"\begin{tabularx}{\textwidth}{Xc}A & B\\\end{tabularx}" + r"\begin{tabular}{rr}1 & 2\\\end{tabular}"
    panels = results.tabulars(body, {})
    assert [panel["tabular_number"] for panel in panels] == [1, 2]
    assert panels[0]["rows"][0]["cells_latex"] == ["A", "B"]
    assert panels[1]["rows"][0]["cells_text"] == ["1", "2"]


def test_compiled_labels_preserve_aliases_and_nested_caption_groups():
    text = r"\newlabel{tab:one}{{A2}{9}{Text \textbf{nested}}{table.caption.2}{}}" + "\n" + r"\newlabel{tab:alias}{{A2}{9}{Text}{table.caption.2}{}}"
    labels = results.aux_labels(text)
    assert labels["tab:one"] == labels["tab:alias"] == {"number": "A2", "page": "9", "anchor": "table.caption.2"}


@pytest.mark.parametrize("body", [r"A & \textbf{unclosed", r"A} & B\\"])
def test_parser_rejects_unbalanced_cell_groups(body):
    with pytest.raises(ValueError):
        results.table_rows(body)


def test_parser_rejects_mismatched_tabulars_and_duplicate_aux_labels():
    with pytest.raises(ValueError, match="Mismatched"):
        results.tabulars(r"\begin{tabular}{l}x\\\end{tabularx}", {})
    with pytest.raises(ValueError, match="Duplicate"):
        results.aux_labels(r"\newlabel{a}{{1}{2}}\newlabel{a}{{1}{2}}")


def test_offline_bundle_verification_covers_tables_aliases_cells_and_figure(tmp_path):
    write_bundle(tmp_path)
    assert results.verify(tmp_path) == {
        "status": "passed", "table_count": 2, "tabular_count": 2, "cell_count": 8,
        "canonical_families_present": 1, "panel_b_rows": 13, "panel_c_comparisons": 12,
        "point_deltas_checked": 24,
        "scope": "Published snapshot consistency, not independent experiment or bootstrap reproduction"}


@pytest.mark.parametrize("mutation", ["missing_table", "lost_alias", "wrong_page", "duplicate_number", "no_family", "no_prose"])
def test_index_coverage_failures_are_rejected(tmp_path, mutation):
    snapshot, index, graph = bundle()
    if mutation == "missing_table":
        index["tables"].pop()
    elif mutation == "lost_alias":
        index["coverage_from_compiled_labels"][0]["aliases"] = []
    elif mutation == "wrong_page":
        index["tables"][0]["page"] = "999"
    elif mutation == "duplicate_number":
        snapshot["tables"][1]["number"] = "1"
    elif mutation == "no_family":
        index["tables"][1]["canonical_aggregate_families"] = []
    else:
        index["prose_result_families"] = []
    write_bundle(tmp_path, snapshot, index, graph)
    with pytest.raises(ValueError):
        results.verify(tmp_path)


def test_canonical_family_missing_or_path_escape_is_rejected(tmp_path):
    _, index, _ = write_bundle(tmp_path)
    (tmp_path / "aggregates/context.json").unlink()
    with pytest.raises(ValueError, match="family is missing"):
        results.verify(tmp_path)
    with pytest.raises(ValueError, match="Unsafe"):
        results.safe_path(tmp_path, "../outside")


def test_csv_hash_cannot_mask_a_changed_published_cell(tmp_path):
    _, index, _ = write_bundle(tmp_path)
    path = tmp_path / index["tables"][0]["csv"]
    data = path.read_bytes().replace(b"unchanged", b"different")
    path.write_bytes(data)
    index["tables"][0]["csv_sha256"] = results.sha(data)
    (tmp_path / "aggregates/manuscript_results_index.json").write_bytes(results.encoded(index))
    with pytest.raises(ValueError, match="CSV/JSON"):
        results.verify(tmp_path)


def test_self_consistent_hashes_cannot_mask_bad_text_projection(tmp_path):
    snapshot, index, graph = bundle()
    snapshot["tables"][0]["tabulars"][0]["rows"][1]["cells_text"][0] = "not the source cell"
    write_bundle(tmp_path, snapshot, index, graph)
    with pytest.raises(ValueError, match="Cell text"):
        results.verify(tmp_path)


@pytest.mark.parametrize("mutation", ["delta", "excluded", "counts", "raw_provenance"])
def test_figure_points_and_public_scope_are_checked(mutation):
    payload = figure()
    if mutation == "delta":
        payload["panel_c"]["estimates"][0]["accuracy_delta"] += 0.001
    elif mutation == "excluded":
        payload["panel_b"][1]["key"] = "chexagent"
    elif mutation == "counts":
        payload["verified_panel_counts"]["B"] = 99
    else:
        payload["panel_b"][0]["source_refs"] = ["not-public"]
    with pytest.raises(ValueError):
        results.verify_figure(payload)


def test_portable_cli_needs_no_manuscript_or_private_files(tmp_path):
    write_bundle(tmp_path)
    result = subprocess.run([sys.executable, str(Path(results.__file__)), "--root", str(tmp_path)],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["table_count"] == 2
