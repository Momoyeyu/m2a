"""
Experiment runner — the generic version of the ``main()`` functions that used
to live in ``sed-crnn/attack.py`` / ``sed-crnn/arbitrary_attack.py`` /
``ATST-SED/attack.py`` / ``ATST-SED/arbitrary_attack.py``.

Per-sample execution order is preserved exactly:

1. pick the (random) target label / edit set   (python ``random``)
2. construct the editor (model init consumes torch RNG)
3. ``editor.attack()``                          (delta init consumes torch RNG)
"""

import json
import logging
import os
import random
import traceback

import numpy as np
import torch

from .editors import get_editor_class
from .utils import Event, get_rand_label, rand_edit_set, read_data, setup_logging


def seed_all(seed, torch_seed):
    """Reproduce the seeding behaviour of the original per-model scripts."""
    random.seed(seed)
    if torch_seed:
        # sed-crnn/crnn.py already seeds torch/numpy at import time; doing it
        # again here keeps semantics identical when several cells share one
        # process (e.g. the in-process pipeline mode).
        torch.manual_seed(seed)
        np.random.seed(seed)


def build_edit_set(cfg, class_labels):
    """Build the per-sample edit set (None for single-target mode)."""
    if cfg.mode == "single":
        return None
    if cfg.edit_set:
        return [Event(target_label=e["target_label"],
                      attack_type=e.get("attack_type", cfg.attack_type),
                      start_time=e.get("start_time"),
                      end_time=e.get("end_time"))
                for e in cfg.edit_set]
    return rand_edit_set(cfg.edit_num, cfg.attack_type, class_labels)


def run_experiment(cfg):
    """Run one experiment cell (model x method x attack_type) over a wav list.

    Returns a result dict that is also written to ``results_dir`` as JSON.
    """
    editor_cls = get_editor_class(cfg.model)
    seed_all(cfg.seed, getattr(editor_cls, "seed_torch", True))

    if cfg.wav_input:
        wav_list = [cfg.wav_input]
    elif cfg.wav_dir:
        wav_list = read_data(cfg.wav_dir, cfg.num_samples)
    else:
        raise ValueError("either 'wav_input' or 'wav_dir' must be set")

    setup_logging(cfg.log_dir, cfg.log_prefix())
    logging.info(f"Starting Attack on {editor_cls.model_display_name} Model "
                 f"for Sound Event Detection")
    logging.info(f"Total samples: {len(wav_list)}")
    if cfg.use_cw:
        logging.info("Testing attack performance with C&W settings")
    elif cfg.use_preservation_loss:
        logging.info(f"Using preservation loss with alpha: {cfg.alpha}")

    if len(wav_list) == 0:
        logging.warning("No wav files found")
        return None

    eval_matrix = {
        "avg SNR": 0.0,  # Signal to Noise Ratio
        "avg EP": 0.0,   # Average Edit Precision
        "avg ASR": 0.0,
        "avg UER": 0.0,
        "avg Time": 0.0,
    }
    samples = []

    for i, wav_input in enumerate(wav_list):
        logging.info(
            f"\n\n===============================================================================================================================\n"
            f"=====                                                      No.{i}                                                           =====\n"
            f"===============================================================================================================================\n\n")
        # RNG order matches the original scripts: single-target draws one
        # label via get_rand_label(); multi-target draws edit_num labels
        # inside rand_edit_set() (no extra draw).
        if cfg.mode == "single":
            target_label = cfg.target_label or get_rand_label(editor_cls.class_labels)
            edit_set = None
        else:
            target_label = cfg.target_label
            edit_set = build_edit_set(cfg, editor_cls.class_labels)
        sample_cfg = cfg.merged(wav_input=wav_input,
                                target_label=target_label,
                                edit_set=edit_set)
        try:
            attacker = editor_cls(sample_cfg)
            result = attacker.attack()
        except Exception:
            logging.error(f"Attack failed on {wav_input}:\n{traceback.format_exc()}")
            result = None
        if result is None:
            logging.error(f"Sample {wav_input} skipped (attack did not finish).")
            continue
        # normalise numpy scalars so the JSON summary stays numeric
        result = {k: (v.item() if isinstance(v, np.generic) else v)
                  for k, v in result.items()}

        eval_matrix["avg SNR"] += result["SNR"]
        eval_matrix["avg EP"] += result["EP"]
        eval_matrix["avg ASR"] += result["ASR"]
        eval_matrix["avg UER"] += result["UER"]
        eval_matrix["avg Time"] += result["Time"]

        samples.append({"wav": wav_input, "target_label": target_label, **result})
        logging.info("Attack completed.")

    n = len(samples)
    if n == 0:
        logging.error("All attacks failed.")
        return None

    aggregate = {
        "SNR": eval_matrix["avg SNR"] / n,
        "EP": eval_matrix["avg EP"] / n,
        "ASR": eval_matrix["avg ASR"] / n,
        "UER": eval_matrix["avg UER"] / n,
        "Time": eval_matrix["avg Time"] / n,
    }

    logging.info(f"Average SNR: {aggregate['SNR']} dB")
    logging.info(f"Average EP: {aggregate['EP'] * 100:.2f}%")
    logging.info(f"Average ASR: {aggregate['ASR'] * 100:.2f}%")
    logging.info(f"Average UER: {aggregate['UER'] * 100:.2f}%")
    logging.info(f"Average Time: {aggregate['Time']:.2f}")

    summary = {
        "model": cfg.model,
        "method": cfg.method,
        "mode": cfg.mode,
        "attack_type": cfg.attack_type,
        "num_samples": len(wav_list),
        "num_finished": n,
        "aggregate": aggregate,
        "samples": samples,
        "config": cfg.to_dict(),
    }

    os.makedirs(cfg.results_dir, exist_ok=True)
    out_name = f"{cfg.model}_{cfg.method}_{cfg.mode}_{cfg.attack_type}.json"
    out_path = os.path.join(cfg.results_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logging.info(f"Results written to {out_path}")
    return summary
