"""
Base ``Editor`` that drives the adversarial optimization loop.

The class merges the four historical scripts

* ``sed-crnn/attack.py``            (CRNN,  single target)
* ``sed-crnn/arbitrary_attack.py``  (CRNN,  multi target)
* ``ATST-SED/attack.py``            (ATST,  single target)
* ``ATST-SED/arbitrary_attack.py``  (ATST,  multi target)

into a single implementation. A single-target attack is internally treated
as a one-element ``edit_set``; the multi-target code path with a single
event is numerically identical to the original single-target script.

Model-specific behaviour lives in the subclasses (``CrnnEditor`` /
``AtstSedEditor``) through the hooks documented below, so the control flow
is written once while every numeric quirk of the original code is kept:

* ``frame_margin``        - upper clamp bound is ``num_frames - margin``
                            (CRNN single: 1, CRNN multi: 0, ATST: 1)
* ``early_stop_loss``     - CRNN single stops when loss < 1e-4
* ``reject_full_coverage``- ATST single aborts when the segment covers the
                            whole clip
* ``cw_prefix``           - adv file prefix for the C&W variant
                            ("sec_" for CRNN single, "cw_" otherwise)
"""

import logging
import os
import random
import time

import soundfile as sf
import torch
from torch import nn, optim

from ..utils import Event, aro_acoustic_loss, bce_preservation_loss, calc_matrix, calculate_snr


class BaseEditor:
    # -- to be provided by subclasses --------------------------------------
    class_labels = {}          # label name -> class index
    model_display_name = "?"

    # -- behavioural switches (defaults; subclasses may override) ----------
    frame_margin = 1           # frame clamp bound = num_frames - margin
    early_stop_loss = None     # loss threshold for early stopping (None = off)
    reject_full_coverage = False   # abort when segment covers whole audio
    cw_prefix = "cw_"          # filename prefix when use_cw is set

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.wav_input_path = args.wav_input
        self.save_output = args.save_output
        self.output_dir = args.output_dir
        self.attack_iters = args.attack_iters
        self.delta_duration = args.delta_duration  # in seconds
        self.tau = args.tau  # Maximum perturbation amplitude
        self.model_path = args.model_path
        self.posterior_thresh = args.posterior_thresh
        self.seq_len = args.seq_len
        self.mode = args.mode
        self.attack_type = args.attack_type
        self.target_label = args.target_label
        self.use_preservation_loss = args.use_preservation_loss
        self.alpha = args.alpha
        self.use_cw = args.use_cw
        self.use_faag = args.use_faag
        self.use_aro = args.use_aro
        self.aro_ref_wav = args.aro_ref_wav
        self.aro_beta = args.aro_beta

        # Audio parameters
        self.sample_rate = args.sample_rate
        self.n_fft = args.n_fft
        self.hop_len = args.hop_len
        self.nb_mel_bands = args.nb_mel_bands

        if args.early_stop is not None:
            self.early_stop_loss = args.early_stop

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.load_model(args)  # hook: build the victim model

        # Determine delta_length based on start_time and end_time
        start_time, end_time = args.start_time, args.end_time
        if self.mode == "single" and start_time is not None and end_time is not None:
            if end_time <= start_time:
                logging.error("end_time must be greater than start_time. Falling back to random segment.")
                start_time = None
                end_time = None
                self.delta_length = int(self.delta_duration * self.sample_rate)
            else:
                self.delta_length = int((end_time - start_time) * self.sample_rate)
        else:
            self.delta_length = int(self.delta_duration * self.sample_rate)  # Number of samples in delta

        # Build the edit set: a single-target attack is a one-element set.
        if self.mode == "single":
            self.edit_set = [Event(target_label=args.target_label,
                                   attack_type=args.attack_type,
                                   start_time=start_time,
                                   end_time=end_time)]
        else:
            self.edit_set = list(args.edit_set)

        self.aro_ref_feat = None

    # ======================================================================
    # hooks implemented by the subclasses
    # ======================================================================
    def load_model(self, args):
        """Instantiate the victim model (kept in train mode for gradients)."""
        raise NotImplementedError

    def load_audio(self, path=None):
        """Load the input waveform as a 1-D float tensor."""
        raise NotImplementedError

    def select_random_segment(self, total_samples):
        """Pick a random (start_idx, end_idx) pair of ``delta_length`` samples."""
        raise NotImplementedError

    def forward_logits(self, y):
        """Return frame-level predictions of shape [batch, frames, classes]."""
        raise NotImplementedError

    def forward_segment_logits(self, y_adv, event):
        """FAAG-style forward restricted to the perturbed segment."""
        raise NotImplementedError

    def forward_eval(self, y):
        """Inference-mode forward pass used inside ``evaluate_attack``."""
        raise NotImplementedError

    def acoustic_repr(self, y):
        """Acoustic representation E(y) used by the ARO loss."""
        raise NotImplementedError

    # ======================================================================
    # shared implementation
    # ======================================================================
    def init_delta(self, total_samples):
        """
        Resolve the sample indices of every edit and build the perturbation
        tensors. Returns ``None`` when the indices are invalid.
        """
        if not self.edit_set:
            logging.error("Event list is empty.")
            return None
        # calculate (start_idx, end_idx)
        for event in self.edit_set:
            if event.start_time is not None and event.end_time is not None:
                start_idx = int(event.start_time * self.sample_rate)
                end_idx = int(event.end_time * self.sample_rate)
                if start_idx < 0 or end_idx > total_samples:
                    logging.error("Specified start_time and/or end_time are out of bounds. Exiting attack.")
                    return None
                event.start_idx = start_idx
                event.end_idx = end_idx
                if self.mode == "single":
                    expected_length = self.delta_length
                    actual_length = end_idx - start_idx
                    if actual_length != expected_length:
                        logging.warning(f"Specified perturbation length ({actual_length} samples) "
                                        f"does not match expected delta_length ({expected_length} samples).")
                    logging.info(f"Using specified segment from {start_idx} to {end_idx} for perturbation.")
            else:
                start_idx, end_idx = self.select_random_segment(total_samples)
                event.start_idx = start_idx
                event.end_idx = end_idx
                event.start_time = event.start_idx / self.sample_rate
                event.end_time = event.end_idx / self.sample_rate
        # sort event_idx by start_idx
        self.edit_set.sort(key=lambda x: x.start_idx)

        if self.mode == "single":
            event = self.edit_set[0]
            if self.reject_full_coverage and event.start_idx == 0 and event.end_idx == total_samples:
                logging.error("Perturbation segment covers the entire audio. Exiting attack.")
                return None
            logging.info(
                f"Perturbation segment: start_time={event.start_idx / self.sample_rate:.3f}s, "
                f"end_time={event.end_idx / self.sample_rate:.3f}s")

        if self.use_cw:  # full delta settings used. only used when attacking task like SEC
            delta = {
                "ptb": torch.zeros(total_samples, requires_grad=True, device=self.device),
                "start_idx": 0,
                "end_idx": total_samples,
            }
            # Initialize delta within [-tau, tau]
            delta["ptb"].data = delta["ptb"].data.uniform_(-self.tau, self.tau)
            return [delta]

        # generate delta_list for the editor
        event = self.edit_set[0]
        delta_list = [{
            "start_idx": event.start_idx,
            "end_idx": event.end_idx,
        }]
        for event in self.edit_set[1:]:
            # get last delta
            last_delta = delta_list[-1]
            # if time crosses
            if event.start_idx <= last_delta["end_idx"]:
                # merge two event's deltas into a single delta
                last_delta["end_idx"] = max(last_delta["end_idx"], event.end_idx)
            else:  # else, generate new delta
                delta_list.append({
                    "start_idx": event.start_idx,
                    "end_idx": event.end_idx,
                })
        # instantiate deltas
        for delta in delta_list:
            delta["ptb"] = torch.zeros(delta["end_idx"] - delta["start_idx"], requires_grad=True,
                                       device=self.device)
            delta["ptb"].data = delta["ptb"].data.uniform_(-self.tau, self.tau)
        return delta_list

    def apply_deltas(self, y, delta_list):
        y_adv = y.clone().detach()
        if self.use_cw:  # full delta settings used. only used when attacking task like SEC
            y_adv += delta_list[0]["ptb"]
        else:
            for delta in delta_list:
                y_adv[delta["start_idx"]:delta["end_idx"]] += delta["ptb"]
        # Ensure audio stays in [-1, 1]
        return torch.clamp(y_adv, -1.0, 1.0)

    def clamp_frames(self, event, num_frames):
        """Convert the sample indices of an event into frame indices."""
        bound = num_frames - self.frame_margin
        event.start_frame = max(0, min(int(event.start_idx / self.hop_len), bound))
        event.end_frame = max(0, min(int(event.end_idx / self.hop_len), bound))

    def load_aro_reference(self, total_samples):
        """Load the reference audio used by the ARO loss and extract E(x_ref)."""
        ref = self.load_audio(self.aro_ref_wav)
        if ref is None:
            logging.error(f"Failed to load aro_ref_wav {self.aro_ref_wav}. Exiting attack.")
            return False
        ref = ref.to(self.device)
        if ref.shape[0] < total_samples:
            ref = torch.nn.functional.pad(ref, (0, total_samples - ref.shape[0]))
        else:
            ref = ref[:total_samples]
        with torch.no_grad():
            self.aro_ref_feat = self.acoustic_repr(ref).detach()
        return True

    def attack(self):
        y = self.load_audio()
        if y is None:
            logging.error("Failed to load audio. Exiting attack.")
            return None
        y = y.to(self.device)

        if self.use_aro and not self.load_aro_reference(y.shape[0]):
            return None

        # Step 1: Initialize perturbation delta on device
        delta_list = self.init_delta(y.shape[0])
        if not delta_list:
            return None
        optimizer = optim.Adam([delta["ptb"] for delta in delta_list], lr=1e-3)
        criterion = nn.BCELoss()

        # Step 2: Generate output of origin input
        y_org = y.clone().detach()
        logits_org = self.forward_logits(y_org)  # [batch, time, num_classes]

        # Determine which time frames correspond to the perturbed segment
        num_frames = logits_org.shape[1]
        for event in self.edit_set:
            self.clamp_frames(event, num_frames)

        # Create target labels based on attack_type and target_label
        target = logits_org.clone().detach()  # Ensure 'logits' is defined before this line

        # Edit target output
        for event in self.edit_set:
            if event.attack_type is not None and event.target_label is not None:
                if event.attack_type not in ["mirage", "mute"]:
                    logging.error("Invalid attack_type. It must be either 'mirage' or 'mute'. Exiting attack.")
                    return None
                if event.target_label not in self.class_labels:
                    logging.error(f"Invalid target_label '{event.target_label}'. Exiting attack.")
                    return None

                target_class_idx = self.class_labels[event.target_label]

                if event.attack_type == "mirage":
                    # Set the target_label's output to 1 in the perturbed segment
                    target[:, event.start_frame:event.end_frame, target_class_idx] = 1.0
                    logging.info(
                        f"Attack type: 'mirage' - setting label '{event.target_label}' to 1 "
                        f"from {event.start_frame} to {event.end_frame} frame.")
                elif event.attack_type == "mute":
                    # Set the target_label's output to 0 in the perturbed segment
                    target[:, event.start_frame:event.end_frame, target_class_idx] = 0.0
                    logging.info(
                        f"Attack type: 'mute' - setting label '{event.target_label}' to 0 "
                        f"from {event.start_frame} to {event.end_frame} frame.")
            else:
                logging.error("No attack_type or target_label specified.")
                return None

        attack_start_time = time.time()

        # Step 3: Enter the optimization loop
        for iteration in range(1, self.attack_iters + 1):
            optimizer.zero_grad()

            # Step 3.1: Apply perturbation
            y_adv = self.apply_deltas(y, delta_list)

            # Step 3.2: Pass through the model and calculate optimization loss
            if self.use_faag:
                adv_loss = 0
                for event in self.edit_set:
                    target_class_idx = self.class_labels[event.target_label]
                    logits = self.forward_segment_logits(y_adv, event)
                    adv_loss = adv_loss + criterion(
                        logits[:, :, target_class_idx],
                        target[:, event.start_frame:event.end_frame, target_class_idx])
            else:
                logits = self.forward_logits(y_adv)
                # Compute loss only on the perturbed segment
                adv_loss = 0
                preservation_mask = torch.ones_like(logits, dtype=torch.float)
                for event in self.edit_set:  # edit_set has been sorted at init_delta
                    target_class_idx = self.class_labels[event.target_label]
                    adv_loss = adv_loss + criterion(
                        logits[:, event.start_frame:event.end_frame, target_class_idx],
                        target[:, event.start_frame:event.end_frame, target_class_idx])
                    preservation_mask[:, event.start_frame:event.end_frame, target_class_idx] = 0
                # preservation consider global context: which may decease ASR but also decease UER
                preservation_loss = bce_preservation_loss(logits, target, preservation_mask)

            if self.use_aro:
                aro_loss = aro_acoustic_loss(self.acoustic_repr(y_adv), self.aro_ref_feat)

            loss = adv_loss
            if self.use_preservation_loss:
                loss = loss + preservation_loss * self.alpha
            if self.use_aro:
                loss = loss + aro_loss * self.aro_beta

            # Step 3.3: Backpropagate and step
            loss.backward()
            optimizer.step()

            # Step 3.4: Clip delta to stay within [-tau, tau]
            for delta in delta_list:
                delta["ptb"].data = torch.clamp(delta["ptb"].data, -self.tau, self.tau)

            # Logging
            if iteration % 100 == 0 or iteration == 1:
                log_msg = f"Iteration {iteration}/{self.attack_iters}, adv loss: {adv_loss.item():.6f}"
                if self.use_preservation_loss:
                    log_msg += f", preservation loss: {preservation_loss.item():.6f}"
                if self.use_aro:
                    log_msg += f", aro loss: {aro_loss.item():.6f}"
                logging.info(log_msg)

            # Early stopping if loss is sufficiently low
            if self.early_stop_loss is not None and loss.item() < self.early_stop_loss:
                logging.info(f"Attack succeeded at iteration {iteration}")
                break

        attack_end_time = time.time()

        # Step 4: Apply the final perturbation
        with torch.no_grad():
            y_adv = self.apply_deltas(y, delta_list)

        # Step 5: Save the adversarial audio
        adv_audio = y_adv.cpu().numpy()
        adv_filename = self.adv_filename()
        adv_path = os.path.join(self.output_dir, adv_filename)
        if self.save_output:
            sf.write(adv_path, adv_audio, self.sample_rate)
            logging.info(f"Adversarial audio saved to {adv_path}")

        # Step 6: Evaluate the attack
        original = y.clone().detach()  # y is already on device
        return self.evaluate_attack(original, y_adv, attack_end_time - attack_start_time)

    def adv_filename(self):
        adv_filename = f"adv_{os.path.splitext(os.path.basename(self.wav_input_path))[0]}"
        if self.mode == "single":
            event = self.edit_set[0]
            adv_filename += (f"_{event.attack_type}_{event.target_label}_"
                             f"{event.start_time:.3f}-{event.end_time:.3f}")
        else:
            for event in self.edit_set:
                adv_filename += (f"_{event.attack_type}_{event.target_label}_"
                                 f"{event.start_time:.3f}_{event.end_time:.3f}")
        adv_filename += ".wav"
        if self.use_cw:
            adv_filename = self.cw_prefix + adv_filename
        return adv_filename

    def evaluate_attack(self, original, y_adv, attack_time):
        """
        Evaluate the success of the attack based on attack_type and target_label,
        and calculate the Signal-to-Noise Ratio (SNR) between original and adversarial audio.

        Returns:
            dict: A dictionary containing 'EP', 'ASR', 'UER', 'SNR', 'Time'
                  and the raw 'SE'/'FE'/'UE'/'NE' counts.
        """
        logits_org = self.forward_eval(original)
        logits_adv = self.forward_eval(y_adv)  # [batch, time, num_classes]

        num_frames = logits_adv.shape[1]
        for event in self.edit_set:
            self.clamp_frames(event, num_frames)

        evaluation_result = {
            "EP": 0,
            "ASR": 0,
            "UER": 0,
            "SNR": 0.0,
            "Time": attack_time,
            "SE": 0,
            "FE": 0,
            "UE": 0,
            "NE": 0,
        }

        pred_org = (logits_org > self.posterior_thresh)
        pred_adv = (logits_adv > self.posterior_thresh)
        se = 0
        fe = 0
        ne = (pred_org == pred_adv).sum().item()
        ue = (pred_org != pred_adv).sum().item()

        for event in self.edit_set:
            if event.attack_type is not None and event.target_label is not None:
                if event.target_label not in self.class_labels:
                    logging.error(f"Invalid target_label '{event.target_label}'. Evaluation skipped.")
                    return evaluation_result

                target_class_idx = self.class_labels[event.target_label]

                target_pred_org = pred_org[0, event.start_frame:event.end_frame, target_class_idx]
                target_pred_adv = pred_adv[0, event.start_frame:event.end_frame, target_class_idx]
                ue -= (target_pred_org != target_pred_adv).sum().item()
                ne -= (target_pred_org == target_pred_adv).sum().item()
                if event.attack_type == "mute":
                    se += (target_pred_adv == False).sum().item()
                    fe += (target_pred_adv == True).sum().item()
                elif event.attack_type == "mirage":
                    se += (target_pred_adv == True).sum().item()
                    fe += (target_pred_adv == False).sum().item()
            else:
                logging.warning("attack_type or target_label not specified. Evaluation skipped.")

        evaluation_result["EP"], evaluation_result["ASR"], evaluation_result["UER"] = calc_matrix(se, fe, ue, ne)
        evaluation_result["SE"], evaluation_result["FE"] = se, fe
        evaluation_result["UE"], evaluation_result["NE"] = ue, ne
        logging.info(f"Edit Precision (EP): {evaluation_result['EP'] * 100:.2f}%")
        logging.info(f"Attack Success Rate (ASR): {evaluation_result['ASR'] * 100:.2f}%")
        logging.info(f"Unintended Edit Rate (UER): {evaluation_result['UER'] * 100:.2f}%")
        logging.info(f"Attack time consumption: {evaluation_result['Time']:.2f}")
        # Compute SNR
        try:
            snr_value = calculate_snr(original, y_adv)
            evaluation_result["SNR"] = snr_value
            logging.info(f"Signal-to-Noise Ratio (SNR): {snr_value:.2f} dB")
        except ValueError as ve:
            logging.error(f"SNR Calculation Error: {ve}")
            evaluation_result["SNR"] = None
        return evaluation_result
