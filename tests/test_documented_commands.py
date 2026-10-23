"""Parse documented commands without importing scripts or executing workloads.

Only argparse declarations are reconstructed from AST. Defaults/help text are
irrelevant to syntax validation; runtime input checks remain the user's task.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import shlex

import pytest

from tools.run_paper_training import command

ROOT = Path(__file__).resolve().parents[1]


def declared_parser(path: Path) -> argparse.ArgumentParser:
    tree = ast.parse(path.read_text())
    objects = {}
    root = None
    assignments = [node for node in ast.walk(tree)
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)]
    # Resolve parser/subparser assignments in source order; no script imports.
    for node in sorted(assignments, key=lambda item: item.lineno):
        call, target = node.value, node.targets[0].id
        if not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr == "ArgumentParser":
            parser = argparse.ArgumentParser(prog=path.name)
            objects[target] = parser
            root = parser
        elif isinstance(call.func.value, ast.Name) and call.func.value.id in objects:
            owner = objects[call.func.value.id]
            if call.func.attr == "add_subparsers":
                options = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords
                           if kw.arg in ("dest", "required")}
                objects[target] = owner.add_subparsers(**options)
            elif call.func.attr == "add_parser":
                objects[target] = owner.add_parser(ast.literal_eval(call.args[0]))
    assert root is not None, f"No argparse declaration in {path}"
    for node in sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call)), key=lambda item: item.lineno):
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                and isinstance(node.func.value, ast.Name) and node.func.value.id in objects):
            continue
        flags = [ast.literal_eval(arg) for arg in node.args]
        options = {}
        for kw in node.keywords:
            if kw.arg in ("action", "nargs", "required", "choices", "dest"):
                options[kw.arg] = ast.literal_eval(kw.value)
            elif kw.arg == "type":
                assert isinstance(kw.value, ast.Name), f"Unsupported declared CLI type in {path}"
                options["type"] = {"int": int, "float": float, "str": str, "Path": Path}[kw.value.id]
        objects[node.func.value.id].add_argument(*flags, **options)
    return root


def local_examples():
    examples = []
    for document in (ROOT / "README.md", ROOT / "RECONSTRUCTION.md", ROOT / "docs/CONDITIONAL_RELIABILITY.md"):
        for number, line in enumerate(document.read_text().splitlines(), start=1):
            tokens = shlex.split(line) if line.startswith("python ") else []
            if len(tokens) >= 2 and tokens[1].endswith(".py"):
                script = tokens[1]
                integer_examples = {"$QL_RUN_SEED", "$QL_CONTROL_SEED", "$QL_SWAP_ROWS"}
                arguments = ["123" if token in integer_examples else token for token in tokens[2:]]
                examples.append((str(document.relative_to(ROOT)), number, script, arguments))
    return examples


@pytest.mark.parametrize("document,line,script,arguments", local_examples())
def test_documented_local_command_parses(document, line, script, arguments):
    path = ROOT / script
    assert path.is_file(), f"Missing documented entrypoint at {document}:{line}"
    parser = declared_parser(path)
    if "--help" in arguments:
        # A help flag may short-circuit an otherwise invalid argparse invocation.
        # Documentation help examples must contain no hidden extra arguments.
        assert arguments == ["--help"]
    else:
        parser.parse_args(arguments)


@pytest.mark.parametrize("no_eval", [False, True])
def test_printed_training_command_parses_against_actual_trainer(no_eval):
    cmd = command(Path("synthetic-model"), Path("synthetic-data"), Path("synthetic-output"), 123,
                  no_eval=no_eval)
    parser = declared_parser(ROOT / "scripts/train_medgemma_longitudinal_sft.py")
    args = parser.parse_args(cmd[2:])
    assert args.max_train_samples == 2048 and args.max_eval_samples == 512
    assert args.no_eval == no_eval


def test_guard_rejects_wrong_subcommand_unknown_flags_and_bad_choices():
    baseline = declared_parser(ROOT / "scripts/cmig_medgemma_baseline.py")
    with pytest.raises(SystemExit):
        baseline.parse_args(["infer", "--protocol", "synthetic.json", "--output-dir", "fake"])
    with pytest.raises(SystemExit):
        baseline.parse_args(["prepare", "--not-an-option", "fake"])
    builder = declared_parser(ROOT / "dataset/build_ms_cxr_t_longitudinal_stage1.py")
    with pytest.raises(SystemExit):
        builder.parse_args(["--split-policy", "nonexistent-policy"])
