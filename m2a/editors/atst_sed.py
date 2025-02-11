"""ATST-SED victim-model adapter.

Merges ``ATST-SED/attack.py`` and ``ATST-SED/arbitrary_attack.py``.
"""

import logging
import os
import random
import sys

import torch

# ``ATST-SED`` is the vendored copy of the official ATST-SED repo; its modules
# are written for a flat sys.path, so expose them directly.
_ATST_SED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "ATST-SED")
if _ATST_SED_DIR not in sys.path:
    sys.path.insert(0, _ATST_SED_DIR)

from inference import ATSTSEDInferencer  # noqa: E402
from desed_task.dataio.datasets_atst_sed import read_audio  # noqa: E402

from .base import BaseEditor  # noqa: E402


class AtstSedEditor(BaseEditor):
    class_labels = {
        "Alarm_bell_ringing": 0,
        "Blender": 1,
        "Cat": 2,
        "Dishes": 3,
        "Dog": 4,
        "Electric_shaver_toothbrush": 5,
        "Frying": 6,
        "Running_water": 7,
        "Speech": 8,
        "Vacuum_cleaner": 9,
    }
    model_display_name = "ATST-SED"
    seed_torch = False  # the original ATST-SED scripts only seed python's random
    reject_full_coverage = True  # single-target mode aborts on full coverage
    frame_margin = 1
    cw_prefix = "cw_"

    def load_model(self, args):
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.config_path = args.config_path

        # Load the trained ATST-SED model with fixed parameters
        self.inferencer = ATSTSEDInferencer(
            self.model_path,
            self.config_path,
            overlap_dur=args.overlap_dur)
        self.inferencer.to(self.device)
        # Set model to training mode to enable backward pass
        self.inferencer.model.train()

        # Define spectrogram transformer (must match training)
        self.mel_spectrogram = self.inferencer.feature_extractor.sed_feat_extractor

    def load_audio(self, path=None):
        """
        Load the input audio (same pipeline as the original scripts).
        """
        if path is None:
            path = self.wav_input_path
        try:
            y, _, _, _ = read_audio(path, False, False, None)
            return y
        except Exception as e:
            logging.error(f"Error loading audio {self.wav_input_path}: {e}")
            return None

    def select_random_segment(self, total_samples):
        # Check if the audio is shorter than the perturbation duration
        if total_samples <= self.delta_length:
            logging.error("Audio is shorter than the perturbation duration.")
            # Adjust the start and end indices to ensure they are divisible by hop_len
            start_idx = 0
            end_idx = (total_samples // self.hop_len) * self.hop_len
            return start_idx, end_idx

        max_start = ((total_samples - self.delta_length) // self.hop_len) * self.hop_len
        start_idx = random.randint(0, max_start // self.hop_len) * self.hop_len
        end_idx = start_idx + self.delta_length

        # Adjust index
        end_idx = (end_idx // self.hop_len) * self.hop_len
        if end_idx > total_samples:
            end_idx = (total_samples // self.hop_len) * self.hop_len
            start_idx = end_idx - self.delta_length
            if start_idx < 0:
                start_idx = 0

        return start_idx, end_idx

    def forward_logits(self, y):
        _, logits = self.inferencer(y)  # [batch, frame, num_classes]
        return torch.cat(logits).transpose(-1, -2)

    def forward_segment_logits(self, y_adv, event):
        _, logits = self.inferencer(y_adv[event.start_idx:event.end_idx])
        return torch.cat(logits).transpose(-1, -2)

    def forward_eval(self, y):
        self.inferencer.eval()
        with torch.no_grad():
            logits = self.forward_logits(y)
        self.inferencer.model.train()
        return logits

    def acoustic_repr(self, y):
        sed_feats, _ = self.inferencer.feature_extractor(y)
        return sed_feats
