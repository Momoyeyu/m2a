#!/usr/bin/env python
"""
Batch pipeline: run every (model x method x attack_type) cell of a table
configuration, then aggregate mirage/mute means and print the 2-D python
array whose dimensions match the paper tables.

Each cell is executed as a subprocess (``python run_attack.py``) so that the
per-run RNG state matches the original one-process-per-experiment behaviour.
Cells can run in parallel across GPUs (``execution.jobs`` / ``execution.gpus``
in the batch YAML, or ``--jobs`` on the CLI).

Example
-------
python run_table.py --config configs/tables/table1_single.yaml --jobs 2
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml

from m2a.config import deep_update, parse_set_value
from m2a.report import build_table, collect_results, print_table, save_table

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_cell_cfg(table_cfg, cfg_dir, model, method, attack_type):
    """Merge model defaults with the table-level overrides for one cell."""
    model_yaml = os.path.join(cfg_dir, table_cfg.get("model_dir", "../models"),
                              f"{model}.yaml")
    raw = load_yaml(model_yaml)
    # ``base_dir`` in the model yaml is relative to the yaml's own directory;
    # make it absolute so the generated cell config resolves correctly.
    model_dir = os.path.dirname(os.path.abspath(model_yaml))
    raw["base_dir"] = os.path.normpath(
        os.path.join(model_dir, raw.get("base_dir", ".")))

    cell = deep_update(raw, {
        "mode": table_cfg.get("mode", "single"),
        "attack_type": attack_type,
        "method": method,
    })
    # global overrides -> per-model overrides -> per-method overrides
    cell = deep_update(cell, table_cfg.get("overrides"))
    cell = deep_update(cell, table_cfg.get("model_overrides", {}).get(model))
    cell = deep_update(cell, table_cfg.get("method_overrides", {}).get(method))
    cell = deep_update(cell,
                       table_cfg.get("cell_overrides", {}).get(model, {}).get(method))

    results_dir = os.path.join(table_cfg["results_dir"], "cells")
    cell["results_dir"] = results_dir
    cell["log_dir"] = os.path.join(table_cfg["results_dir"], "logs")
    return cell


def run_cell(name, cell_dict, python, workdir, gpu, dry_run):
    cmd = [python, os.path.join(REPO_ROOT, "run_attack.py"), "--config", name]
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[run_table] start cell {os.path.basename(name)[:-5]} "
          f"(gpu={gpu}) -> {' '.join(cmd)}")
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=workdir or REPO_ROOT, env=env)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="batch YAML describing the table")
    parser.add_argument("--jobs", type=int, default=None,
                        help="number of cells to run in parallel (default: from config or 1)")
    parser.add_argument("--python", dest="pythons", action="append", default=[],
                        metavar="MODEL=PATH",
                        help="interpreter used for a model's cells, e.g. "
                             "--python crnn=.venv/bin/python (repeatable)")
    parser.add_argument("--set", dest="sets", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="extra override applied to every cell (repeatable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="only generate cell configs and print the commands")
    args = parser.parse_args()

    cfg_path = os.path.abspath(args.config)
    cfg_dir = os.path.dirname(cfg_path)
    table_cfg = load_yaml(cfg_path)
    table_cfg.setdefault("name", os.path.splitext(os.path.basename(cfg_path))[0])

    results_dir = table_cfg.get("results_dir",
                                os.path.join(REPO_ROOT, "results", table_cfg["name"]))
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(cfg_dir, results_dir)
    table_cfg["results_dir"] = results_dir
    cells_dir = os.path.join(results_dir, "cells")
    os.makedirs(cells_dir, exist_ok=True)

    models = table_cfg["models"]
    methods = table_cfg["methods"]
    attack_types = table_cfg.get("attack_types", ["mirage", "mute"])

    # CLI --set values apply as global cell overrides
    for item in args.sets:
        key, _, value = item.partition("=")
        table_cfg.setdefault("overrides", {})[key.strip()] = parse_set_value(value)

    # materialize every cell config
    jobs = []
    for model in models:
        for method in methods:
            for attack_type in attack_types:
                cell = build_cell_cfg(table_cfg, cfg_dir, model, method, attack_type)
                name = f"{model}_{method}_{attack_type}"
                cell_path = os.path.join(cells_dir, f"{name}.yaml")
                with open(cell_path, "w") as f:
                    yaml.safe_dump(cell, f, sort_keys=False)
                jobs.append((cell_path, cell))

    execution = table_cfg.get("execution", {})
    n_jobs = args.jobs or execution.get("jobs", 1)
    gpus = execution.get("gpus")  # e.g. [0, 1]; None -> inherit
    pythons = dict(table_cfg.get("python") or {})
    for item in args.pythons:
        model, _, path = item.partition("=")
        pythons[model.strip()] = path.strip()
    workdirs = table_cfg.get("workdir", {})

    def submit(idx_job):
        idx, (cell_path, cell) = idx_job
        gpu = gpus[idx % len(gpus)] if gpus else None
        python = pythons.get(cell["model"], sys.executable)
        workdir = workdirs.get(cell["model"], cell.get("workdir"))
        rc = run_cell(cell_path, cell, python, workdir, gpu, args.dry_run)
        return os.path.basename(cell_path), rc

    with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as pool:
        futures = [pool.submit(submit, (i, j)) for i, j in enumerate(jobs)]
        failed = [name for name, rc in
                  (f.result() for f in as_completed(futures)) if rc != 0]
    if failed:
        print(f"[run_table] WARNING: {len(failed)} cell(s) failed: {failed}")

    # ---- aggregate: mean over attack types ------------------------------
    results = collect_results(os.path.join(results_dir, "cells"))
    table = build_table(results, methods, models, attack_types)
    title = f"{table_cfg['name']} ({table_cfg.get('mode', 'single')}-target)"
    print_table(table, methods, models, title=title)
    save_table(table, methods, models,
               os.path.join(results_dir, "table.json"), title=title)


if __name__ == "__main__":
    main()
