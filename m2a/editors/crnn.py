"""CRNN (SEDnet) victim-model adapter.

Merges ``sed-crnn/attack.py`` and ``sed-crnn/arbitrary_attack.py``.
"""

import logging
import os
import random
import sys

import torch
import torchaudio

# ``sed-crnn`` is the vendored copy of the original (pytorch) CRNN repo;
# its modules are written for a flat sys.path, so expose them directly.
_SED_CRNN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "sed-crnn")
if _SED_CRNN_DIR not in sys.path:
    sys.path.insert(0, _SED_CRNN_DIR)

from crnn import CRNNForest  # noqa: E402

from .base import BaseEditor  # noqa: E402


class CrnnEditor(BaseEditor):
    class_labels = {
        'brakes squeaking': 0,
        'car': 1,
        'children': 2,
        'large vehicle': 3,
        'people speaking': 4,
        'people walking': 5
    }
    model_display_name = "CRNN"
    seed_torch = True  # sed-crnn/crnn.py seeds torch & numpy at import time

    def __init__(self, args):
        if args.mode == "single":
            self.cw_prefix = "sec_"
            self.early_stop_loss = 1e-4
            self.frame_margin = 1
        else:
            self.cw_prefix = "cw_"
            self.frame_margin = 0
        super().__init__(args)

    def load_model(self, args):
        # Load the trained CRNN model with fixed parameters
        self.crnn = CRNNForest(self.model_path, self.device, args, args.num_classes)
        self.crnn.train()  # Set model to training mode to enable backward pass

        # Define spectrogram transformer (must match training)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_len,
            n_mels=args.nb_mel_bands,
            center=True,
            power=2.0
        ).to(self.device)

    def load_audio(self, path=None):
        """
        Load and preprocess the input audio.
        """
        if path is None:
            path = self.wav_input_path
        try:
            y, sr = torchaudio.load(path)
            y = y.to(torch.float32)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                y = resampler(y)
            if y.shape[0] > 1:
                y = torch.mean(y, dim=0, keepdim=True)  # Convert to mono

            # Normalize to [-1, 1]
            y = y / torch.max(torch.abs(y))
            target_samples = self.sample_rate * 25
            current_samples = y.shape[1]
            if current_samples > target_samples:
                start_index = random.randint(0, current_samples - target_samples)
                y = y[:, start_index:start_index + target_samples]
            # Ensure the tensor is 1D to match delta's shape
            return y.squeeze(0)
        except Exception as e:
            logging.error(f"Error loading audio {self.wav_input_path}: {e}")
            return None

    def select_random_segment(self, total_samples):
        if total_samples <= self.delta_length:
            logging.error("Audio is shorter than the perturbation duration.")
            return 0, total_samples  # Return integer indices
        max_start = total_samples - self.delta_length
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + self.delta_length
        return start_idx, end_idx

    def calc_spectrogram(self, y):
        """
        Calculate the Mel spectrogram of the audio.
        """
        spectrogram = self.mel_spectrogram(y)
        spectrogram = torch.log1p(spectrogram)  # Log scaling
        # Normalize
        mean = spectrogram.mean()
        std = spectrogram.std()
        spectrogram = (spectrogram - mean) / (std + 1e-10)
        return spectrogram.unsqueeze(0)  # [batch, n_mels, time]

    def forward_logits(self, y):
        spec = self.calc_spectrogram(y)  # [time] -> [1, n_mels, frame]
        spec = spec.unsqueeze(1)  # Add channel dimension -> [1, 1, n_mels, frame]
        return self.crnn(spec)  # [batch, frame, num_classes]

    def forward_segment_logits(self, y_adv, event):
        spec = self.calc_spectrogram(y_adv).unsqueeze(1)  # [1, 1, n_mels, frame]
        # [batch, channel, feature, frame] -> [batch, frame, num_classes]
        return self.crnn(spec[:, :, :, event.start_frame:event.end_frame])

    def forward_eval(self, y):
        self.crnn.eval()
        with torch.no_grad():
            return self.forward_logits(y)  # [batch, frame, num_classes]

    def acoustic_repr(self, y):
        return self.calc_spectrogram(y)
