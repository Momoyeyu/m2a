"""
Shared utilities for the M2A attack framework.

All functions in this module are shared by every model / attack-mode
combination. They are kept byte-for-byte equivalent to the helpers that used
to be duplicated across ``sed-crnn/attack.py``, ``sed-crnn/arbitrary_attack.py``,
``ATST-SED/attack.py`` and ``ATST-SED/arbitrary_attack.py``.
"""

import datetime
import logging
import math
import os
import random

import numpy as np
import torch


class Event:
    """A single editing target used by the (multi-)target attack."""

    def __init__(self,
                 target_label,  # target label
                 attack_type="mirage",  # attack type ("mirage" or "mute)
                 start_time=None,  # ptb start time
                 end_time=None):  # ptb end time
        assert target_label is not None
        assert attack_type == "mirage" or attack_type == "mute"
        self.target_label = target_label
        self.attack_type = attack_type
        self.start_time = start_time
        self.end_time = end_time
        self.start_idx = None
        self.end_idx = None
        self.start_frame = None
        self.end_frame = None


def calc_matrix(se, fe, ue, ne):
    EP = (se + ne) / (se + fe + ue + ne)
    ASR = se / (se + fe)
    UER = ue / (ue + ne)
    return EP, ASR, UER


def read_data(data_dir, max_length):
    wav_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_path = os.path.join(root, file)
                wav_list.append(wav_path)
    repeat_times = math.ceil(max_length / len(wav_list))
    repeated_list = wav_list * repeat_times
    result = repeated_list[:max_length]
    return result


def calculate_snr(original, adversarial):
    """
    Calculate the Signal-to-Noise Ratio (SNR) between the original and adversarial audio.

    Args:
        original (torch.Tensor or np.ndarray): Original audio tensor or numpy array.
        adversarial (torch.Tensor or np.ndarray): Adversarial audio tensor or numpy array.

    Returns:
        float: The calculated SNR in decibels (dB).
    """
    if isinstance(original, torch.Tensor):
        original = original.cpu().numpy()
    if isinstance(adversarial, torch.Tensor):
        adversarial = adversarial.cpu().numpy()

    # Ensure both arrays are of the same shape
    if original.shape != adversarial.shape:
        raise ValueError("Original and adversarial audio must have the same shape for SNR calculation.")

    # Calculate the power of the original signal and the noise
    power_signal = np.sum(original ** 2)
    power_noise = np.sum((original - adversarial) ** 2)

    # Handle the case where power_noise is zero (no noise)
    if power_noise == 0:
        return float('inf')

    snr = 10 * np.log10(power_signal / power_noise)
    return snr


def setup_logging(output_dir, log_prefix):
    """
    Set up logging to console and file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Create file handler
    log_filename = os.path.join(output_dir,
                                f'{log_prefix}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def get_rand_label(class_labels):
    return random.choice(list(class_labels.keys()))


def rand_edit_set(num, attack_type, class_labels):
    edit_set = []
    for i in range(num):
        edit_set.append(Event(target_label=get_rand_label(class_labels), attack_type=attack_type))
    return edit_set


def bce_preservation_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute BCE loss only on masked region (non-target region).

    :param pred: Predictions (batch_size, N, C)
    :param target: Ground truth labels (batch_size, N, C)
    :param mask: Mask tensor (batch_size, N, C), indicating non-target region
    :return: BCE loss computed on masked region (non-target region)
    """
    # Compute BCE loss manually
    bce_loss = - (target * torch.log(torch.sigmoid(pred) + 1e-8) + (1 - target) * torch.log(
        1 - torch.sigmoid(pred) + 1e-8))

    # Apply mask
    masked_loss = bce_loss * mask

    # Normalize by valid elements to avoid division by zero
    return masked_loss.sum() / (mask.sum() + 1e-8)


def aro_acoustic_loss(feat_adv: torch.Tensor, feat_ref: torch.Tensor) -> torch.Tensor:
    """
    Acoustic representation loss used by the ARO baseline
    (Jin et al., ICME 2025):  L_AR = 1 - Cos_Sim(E(x_adv), E(x_ref)).

    :param feat_adv: Acoustic representation of the adversarial audio.
    :param feat_ref: Acoustic representation of the reference audio.
    :return: 1 - cosine similarity between the two flattened representations.
    """
    adv = feat_adv.reshape(-1)
    ref = feat_ref.reshape(-1).to(adv.device)
    return 1.0 - torch.nn.functional.cosine_similarity(adv, ref, dim=0)
