#!/usr/bin/env python3
"""Verify published table snapshots and graph arithmetic offline.

This checks bundled coverage/projections and figure point deltas, not original
experiments, private source data, bootstrap resampling or TeX rendering.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re


class ExportError(ValueError):
    pass


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def escaped(text: str, at: int) -> bool:
    start = at
    while start and text[start - 1] == "\\":
        start -= 1
    return (at - start) % 2 == 1


def group(text: str, at: int, opening="{", closing="}") -> tuple[str, int]:
    while at < len(text) and text[at].isspace():
        at += 1
    if at >= len(text) or text[at] != opening:
        raise ExportError(f"Expected {opening!r} at offset {at}")
    start, depth = at + 1, 1
    for end in range(start, len(text)):
        if escaped(text, end):
            continue
        if text[end] == opening:
            depth += 1
        elif text[end] == closing:
            depth -= 1
            if not depth:
                return text[start:end], end + 1
    raise ExportError("Unclosed LaTeX group")


def arguments(text: str, at: int, count: int) -> tuple[list[str], int]:
    result = []
    for _ in range(count):
        while at < len(text) and text[at].isspace():
            at += 1
        while at < len(text) and text[at] == "[":
            _, at = group(text, at, "[", "]")
            while at < len(text) and text[at].isspace():
                at += 1
        value, at = group(text, at)
        result.append(value)
    return result, at


def aux_labels(text: str) -> dict[str, dict]:
    labels = {}
    for match in re.finditer(r"\\newlabel\s*(?=\{)", text):
        values, end = arguments(text, match.end(), 2)
        fields, at = [], 0
        while at < len(values[1]):
            if not values[1][at:].strip():
                break
            field, at = group(values[1], at)
            fields.append(field)
        if len(fields) >= 2:
            record = {"number": fields[0], "page": fields[1],
                      "anchor": fields[3] if len(fields) > 3 else ""}
            if values[0] in labels:
                raise ExportError(f"Duplicate compiled label: {values[0]}")
            labels[values[0]] = record
    return labels


def latex_text(text: str, labels: dict | None = None) -> str:
    """Readable projection; never evaluate numeric strings or infer metrics."""
    labels = labels or {}
    selected = {"parbox": (2, -1), "multicolumn": (3, -1), "multirow": (3, -1),
                "shortstack": (1, 0)}
    pattern = re.compile(r"\\(parbox|multicolumn|multirow|shortstack|ref|cite[a-z]*)\b")
    while match := pattern.search(text):
        name = match.group(1)
        count, select = selected.get(name, (1, 0))
        values, end = arguments(text, match.end(), count)
        value = values[select]
        if name == "ref":
            value = labels.get(value, {}).get("number", value)
        elif name.startswith("cite"):
            value = "[" + value + "]"
        text = text[:match.start()] + value + text[end:]
    symbols = {"pm": "±", "rightarrow": "→", "leftrightarrow": "↔",
               "Delta": "Δ", "times": "×", "dagger": "†", "leq": "≤",
               "geq": "≥", "neq": "≠", "cmark": "✓", "nomark": "—"}
    text = re.sub(r"\\\\(?:\[[^]]*\])?", "\n", text)
    text = re.sub(r"\\par\b", "\n", text)
    text = re.sub(r"\\([%&#_{}$])", lambda m: {"{": "\uFFF0", "}": "\uFFF1"}.get(m[1], m[1]), text)
    text = re.sub(r"\\([A-Za-z]+)\*?", lambda m: symbols.get(m[1], ""), text)
    text = re.sub(r"\\[,;:! ]", " ", text)
    text = text.replace("{", "").replace("}", "").replace("$", "")
    text = text.replace("\uFFF0", "{").replace("\uFFF1", "}")
    text = text.replace("``", '"').replace("''", '"').replace("~", " ")
    return "\n".join(" ".join(line.split()) for line in text.splitlines()).strip()


RULE = re.compile(r"^\s*\\(?:hline|toprule|midrule|bottomrule)\b|"
                  r"^\s*\\(?:cline|cmidrule)(?:\([^)]*\))?\{[^}]*\}|"
                  r"^\s*\\addlinespace(?:\[[^]]*\])?")


def table_rows(body: str) -> list[list[str]]:
    """Split only unescaped depth-zero ampersands/row terminators."""
    rows, cells, start, depth, at = [], [], 0, 0, 0

    def finish(end):
        nonlocal cells, start
        cells.append(body[start:end].strip())
        while match := RULE.match(cells[0]):
            cells[0] = cells[0][match.end():].strip()
        if any(cells):
            rows.append(cells)
        cells = []

    while at < len(body):
        char = body[at]
        if char in "{}" and not escaped(body, at):
            depth += 1 if char == "{" else -1
            if depth < 0:
                raise ExportError("Unbalanced cell braces")
        if not depth and not escaped(body, at):
            if char == "&":
                cells.append(body[start:at].strip())
                start = at + 1
            elif body.startswith("\\\\", at) or body.startswith("\\tabularnewline", at):
                finish(at)
                at += 2 if body.startswith("\\\\", at) else len("\\tabularnewline")
                if at < len(body) and body[at] == "*":
                    at += 1
                if at < len(body) and body[at] == "[":
                    _, at = group(body, at, "[", "]")
                start = at
                continue
        at += 1
    if depth:
        raise ExportError("Unclosed cell braces")
    finish(len(body))
    return rows


TABULAR = re.compile(r"\\(begin|end)\{(tabularx|tabular\*|tabular)\}")


def tabulars(body: str, labels: dict) -> list[dict]:
    result, cursor = [], 0
    while opening := TABULAR.search(body, cursor):
        if opening[1] != "begin":
            raise ExportError("Unexpected tabular closing environment")
        stack = [opening[2]]
        for token in TABULAR.finditer(body, opening.end()):
            if token[1] == "begin":
                stack.append(token[2])
            elif not stack or stack.pop() != token[2]:
                raise ExportError("Mismatched tabular environments")
            if not stack:
                break
        else:
            raise ExportError("Unclosed tabular environment")
        count = 1 if opening[2] == "tabular" else 2
        specs, start = arguments(body, opening.end(), count)
        rows = table_rows(body[start:token.start()])
        result.append({"tabular_number": len(result) + 1, "environment": opening[2],
                       "column_spec_latex": specs[-1],
                       "rows": [{"row_number": i + 1, "cells_latex": row,
                                 "cells_text": [latex_text(cell, labels) for cell in row]}
                                for i, row in enumerate(rows)]})
        cursor = token.end()
    if not result:
        raise ExportError("Rendered table has no literal tabular environment")
    return result

CSV_HEADER = ("tabular_number", "row_number", "cell_number", "cell_latex", "cell_text")


def table_csv(table: dict) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(CSV_HEADER)
    for panel in table["tabulars"]:
        for row in panel["rows"]:
            for cell, (latex, text) in enumerate(zip(row["cells_latex"], row["cells_text"]), 1):
                writer.writerow((panel["tabular_number"], row["row_number"], cell, latex, text))
    return stream.getvalue().encode()

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual, expected, name: str) -> None:
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-12, rel_tol=0),
            f"Figure arithmetic mismatch: {name}")


def verify_figure(payload: dict) -> dict:
    require(payload["schema"] == "qlace-published-figure-data-v1", "Unknown figure-data schema")
    rows, estimates = payload["panel_b"], payload["panel_c"]["estimates"]
    require((len(rows), len(estimates)) == (13, 12), "Active figure requires 13 B rows and 12 C comparisons")
    require(payload["verified_panel_counts"] == {"B": 13, "C": 12}, "Stored figure counts disagree")
    keyed = {row["key"]: row for row in rows}
    require(len(keyed) == 13 and "chexagent" not in keyed, "Duplicate or excluded figure row")
    require(len({row["key"] for row in estimates}) == 12, "Duplicate C comparison")
    require({r["key"] for r in estimates} == set(keyed) - {"qlace"}, "C/B comparison populations differ")
    require(payload["panel_c"]["reference_key"] == "qlace", "Unexpected figure reference")
    for row in rows:
        for metric in ("no_acc", "no_f1", "pc_acc", "pc_f1"):
            value = row[metric]
            require(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and 0 <= value <= 1, "Invalid graph point")
    for metric, suffix in (("accuracy", "acc"), ("macro_f1", "f1")):
        close(payload["primary"][metric], keyed["qlace"]["pc_" + suffix], "primary/" + metric)
        for row in estimates:
            close(row[metric + "_delta"], keyed["qlace"]["pc_" + suffix] - keyed[row["key"]]["pc_" + suffix],
                  row["key"] + "/" + metric)
    def no_raw_references(value):
        if isinstance(value, dict):
            require(not ({"sources", "source_refs"} & set(value)), "Raw figure provenance must not be bundled")
            for item in value.values():
                no_raw_references(item)
        elif isinstance(value, list):
            for item in value:
                no_raw_references(item)
    no_raw_references(payload)
    return {"panel_b_rows": 13, "panel_c_comparisons": 12, "point_deltas_checked": 24}


def safe_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe result-export path")
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root.resolve()), "Result-export path escapes bundle")
    return resolved


def verify(root: Path) -> dict:
    """Check bundled snapshots without TeX, source manuscripts or private data."""
    index = json.loads((root / "aggregates/manuscript_results_index.json").read_text())
    require(index["schema"] == "qlace-manuscript-results-index-v1", "Unknown result-index schema")
    source = safe_path(root, index["snapshot"]).read_bytes()
    require(sha(source) == index["snapshot_sha256"], "Table snapshot hash mismatch")
    snapshot = json.loads(source)
    require(snapshot["schema"] == "qlace-published-tables-v1", "Unknown table-snapshot schema")
    tables = snapshot["tables"]
    require(len(tables) == snapshot["table_count"] == index["table_count"] == len(index["tables"]),
            "Published table count/coverage mismatch")
    identity = lambda table: {key: table[key] for key in ("label", "aliases", "number", "page")}
    require([identity(t) for t in tables] == index["coverage_from_compiled_labels"],
            "Snapshot does not cover the recorded compiled tables")
    require([identity(t) for t in tables] == [identity(t) for t in index["tables"]],
            "Result index does not cover every table")
    numbers = [t["number"] for t in tables]
    all_labels = [name for t in tables for name in [t["label"], *t["aliases"]]]
    require(len(set(numbers)) == len(numbers) and len(set(all_labels)) == len(all_labels),
            "Duplicate table numbers or aliases")
    require(snapshot["context"] == index["context"], "Context/index mismatch")
    require(snapshot["context"]["protocol_counts_label"] in all_labels, "Protocol context is missing")
    references = {key: {"number": value} for key, value in snapshot.get("reference_numbers", {}).items()}
    cells = panels = 0
    families = set()
    for table, entry in zip(tables, index["tables"]):
        require(len(table["tabulars"]) == entry["tabular_count"] > 0, "Tabular coverage mismatch")
        require(table["caption_text"] == latex_text(table["caption_latex"], references), "Caption text projection mismatch")
        for p, panel in enumerate(table["tabulars"], 1):
            panels += 1
            require(panel["tabular_number"] == p and panel["rows"], "Invalid tabular numbering or empty table")
            for r, row in enumerate(panel["rows"], 1):
                require(row["row_number"] == r and row["cells_latex"], "Invalid table row")
                require(len(row["cells_latex"]) == len(row["cells_text"]), "Cell population mismatch")
                require(row["cells_text"] == [latex_text(cell, references) for cell in row["cells_latex"]],
                        "Cell text projection mismatch")
                cells += len(row["cells_latex"])
        actual = safe_path(root, entry["csv"]).read_bytes()
        require(sha(actual) == entry["csv_sha256"], "Published CSV hash mismatch")
        require(actual == table_csv(table), "Published CSV/JSON projection mismatch")
        require(entry["classification"] == "specification" or entry["canonical_aggregate_families"],
                "Measured/context table has no canonical aggregate family")
        for name in entry["canonical_aggregate_families"]:
            require(safe_path(root, name).is_file(), "Referenced canonical aggregate family is missing")
            families.add(name)
    require(index.get("prose_result_families"), "Prose result families are missing")
    for record in index["prose_result_families"]:
        require(record["topic"] and record["canonical_aggregate_families"], "Empty prose-result mapping")
        for name in record["canonical_aggregate_families"]:
            require(safe_path(root, name).is_file(), "Referenced prose aggregate family is missing")
            families.add(name)
    require(len(index["figures"]) == 1, "Expected one numerical diagnostic figure")
    graph = index["figures"][0]
    data = safe_path(root, graph["data"]).read_bytes()
    require(sha(data) == graph["sha256"], "Figure-data hash mismatch")
    figure = json.loads(data)
    require(all(figure[key] == graph[key] for key in ("label", "number", "page", "verified_panel_counts")),
            "Figure/index identity mismatch")
    checked = verify_figure(figure)
    return {"status": "passed", "table_count": len(tables), "tabular_count": panels,
            "cell_count": cells, "canonical_families_present": len(families), **checked,
            "scope": "Published snapshot consistency, not independent experiment or bootstrap reproduction"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    print(json.dumps(verify(parser.parse_args().root), indent=2))


if __name__ == "__main__":
    main()
