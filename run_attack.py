#!/usr/bin/env python
"""
Run a single attack-experiment cell (one model x one method x one attack type).

Examples
--------
# reproduce a cell of Table 1 (single-target, mirage) on CRNN
python run_attack.py --config configs/models/crnn.yaml --set method=m2a --set attack_type=mirage

# quick smoke test
python run_attack.py --config configs/models/atst_sed.yaml \
    --set attack_iters=20 --set num_samples=2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m2a.config import AttackConfig, parse_set_value


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to a YAML config file")
    parser.add_argument("--set", dest="sets", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override a config field (repeatable)")
    parser.add_argument("--dump", action="store_true",
                        help="print the resolved config and exit")
    args = parser.parse_args()

    overrides = {}
    for item in args.sets:
        key, _, value = item.partition("=")
        overrides[key.strip()] = parse_set_value(value)
    cfg = AttackConfig.from_yaml(args.config, overrides=overrides)

    if args.dump:
        import yaml
        print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        return

    from m2a.runner import run_experiment  # deferred: needs torch
    run_experiment(cfg)


if __name__ == "__main__":
    main()
