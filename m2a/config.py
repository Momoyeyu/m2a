"""
Configuration handling for the M2A attack framework.

A single :class:`AttackConfig` replaces the two duplicated ``ArgsNamespace``
classes that used to live in ``sed-crnn/attack.py`` (CRNN) and
``ATST-SED/attack.py`` (ATST-SED). Every experiment is described by a YAML
file; fields that are irrelevant for the selected model are simply ignored.

Method presets reproduce the flag combinations used in the paper:

===========  =======  =========  ========================  ========
method       use_cw   use_faag   use_preservation_loss     use_aro
===========  =======  =========  ========================  ========
m2a (ours)   False    False      True (weighted by alpha)  False
cw           True     False      False                     False
faag         False    True       False                     False
aro          False    False      False                     True
custom       whatever the YAML/CLI says
===========  =======  =========  ========================  ========
"""

import copy
import os

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_NAMES = ("crnn", "atst_sed")
ATTACK_TYPES = ("mirage", "mute")
ATTACK_MODES = ("single", "multi")

# Preset flag combinations for the compared methods.  Explicit ``use_*``
# entries in the YAML file or --set overrides win over the preset.
METHOD_PRESETS = {
    "m2a": dict(use_cw=False, use_faag=False, use_preservation_loss=True, use_aro=False),
    "cw": dict(use_cw=True, use_faag=False, use_preservation_loss=False, use_aro=False),
    "faag": dict(use_cw=False, use_faag=True, use_preservation_loss=False, use_aro=False),
    "aro": dict(use_cw=False, use_faag=False, use_preservation_loss=False, use_aro=True),
    "custom": {},
}

# Config fields whose values are filesystem paths resolved against ``base_dir``.
_PATH_FIELDS = ("wav_input", "wav_dir", "output_dir", "log_dir", "results_dir",
                "config_path", "aro_ref_wav")
_PATH_LIST_FIELDS = ("model_path",)


class AttackConfig:
    """Union of the parameters used by the CRNN and ATST-SED attacks."""

    FIELDS = dict(
        # ----- experiment selection -------------------------------------
        model=None,                  # "crnn" | "atst_sed"
        mode="single",               # "single" | "multi"
        attack_type="mirage",        # "mirage" | "mute"
        method="m2a",                # method preset (see METHOD_PRESETS)
        # ----- inputs -----------------------------------------------------
        wav_input=None,              # single file (optional; else batch over wav_dir)
        wav_dir=None,                # directory with .wav files for batch runs
        num_samples=50,              # max samples drawn (with repetition) from wav_dir
        # ----- attack hyper-parameters ------------------------------------
        posterior_thresh=0.5,
        save_output=False,
        output_dir="AEs/",
        log_dir="logs/",
        results_dir="results/",
        attack_iters=1000,
        delta_duration=3.0,
        tau=0.02,
        alpha=10,
        start_time=None,
        end_time=None,
        target_label=None,           # None -> random label per sample
        edit_num=3,                  # number of edits for mode="multi"
        edit_set=None,               # optional explicit list of event dicts
        # ----- method switches --------------------------------------------
        use_preservation_loss=True,
        use_cw=False,
        use_faag=False,
        use_aro=False,
        aro_ref_wav=None,            # reference audio for the ARO loss
        aro_beta=1.0,                # weight of the ARO loss term
        # ----- audio front-end --------------------------------------------
        sample_rate=44100,
        n_fft=2048,
        hop_len=1024,
        nb_mel_bands=40,
        seq_len=256,
        # ----- CRNN architecture ------------------------------------------
        model_path=None,             # list of fold checkpoints (crnn) / ckpt (atst)
        nb_ch=1,
        num_classes=6,
        cnn_nb_filt=128,
        cnn_pool_size=[1, 1, 1],
        rnn_nb=[32, 32],
        fc_nb=[32],
        dropout_rate=0.5,
        # ----- ATST-SED ---------------------------------------------------
        config_path=None,            # stage2.yaml
        overlap_dur=3,
        # ----- misc ---------------------------------------------------------
        device=None,                 # e.g. "cuda:0"; None -> auto
        seed=42,
        base_dir=None,               # relative paths resolve against this dir
        early_stop=None,             # None -> model default, or loss threshold
        python=None,                 # interpreter used for subprocess cells
        workdir=None,                # cwd used for subprocess cells
    )

    def __init__(self, **kwargs):
        values = dict(self.FIELDS)
        values.update(kwargs)
        for key, value in values.items():
            setattr(self, key, value)
        self._explicit = set(kwargs)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, raw, base_dir=None):
        """Build a config from a plain dict (already YAML-parsed)."""
        raw = dict(raw or {})
        unknown = sorted(set(raw) - set(cls.FIELDS))
        if unknown:
            raise KeyError(f"Unknown config keys: {unknown}")
        cfg = cls(**raw)
        cfg.apply_method_preset()
        cfg.validate()
        if base_dir and cfg.base_dir is None:
            cfg.base_dir = base_dir
        cfg.resolve_paths()
        return cfg

    @classmethod
    def from_yaml(cls, path, overrides=None):
        path = os.path.abspath(path)
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        raw.update(overrides or {})
        cfg_dir = os.path.dirname(path)
        # ``base_dir`` in the file is resolved relative to the file itself.
        base_dir = raw.get("base_dir")
        if base_dir and not os.path.isabs(base_dir):
            raw["base_dir"] = os.path.normpath(os.path.join(cfg_dir, base_dir))
        return cls.from_dict(raw, base_dir=cfg_dir)

    def apply_method_preset(self):
        """Replicates the flag interactions of the original scripts."""
        preset = METHOD_PRESETS.get(self.method)
        if preset is None:
            raise ValueError(f"Unknown method '{self.method}'. "
                             f"Choose one of {sorted(METHOD_PRESETS)}")
        for key, value in preset.items():
            if key not in self._explicit:
                setattr(self, key, value)
        # original behaviour: C&W disables every other variant switch
        if self.use_cw:
            self.use_preservation_loss = False
            self.use_faag = False
            self.use_aro = False

    def validate(self):
        if self.model not in MODEL_NAMES:
            raise ValueError(f"model must be one of {MODEL_NAMES}, got {self.model!r}")
        if self.mode not in ATTACK_MODES:
            raise ValueError(f"mode must be one of {ATTACK_MODES}, got {self.mode!r}")
        if self.attack_type not in ATTACK_TYPES:
            raise ValueError(f"attack_type must be one of {ATTACK_TYPES}, "
                             f"got {self.attack_type!r}")
        if self.mode == "multi" and self.edit_set is None and self.edit_num is None:
            raise ValueError("multi-target mode needs 'edit_num' or 'edit_set'")
        if self.use_aro and not self.aro_ref_wav:
            raise ValueError("method 'aro' requires 'aro_ref_wav' "
                             "(a reference audio used to compute the ARO loss)")

    def resolve_paths(self):
        if not self.base_dir:
            return
        for field in _PATH_FIELDS:
            value = getattr(self, field)
            if value and not os.path.isabs(value):
                setattr(self, field, os.path.normpath(os.path.join(self.base_dir, value)))
        for field in _PATH_LIST_FIELDS:
            value = getattr(self, field)
            if value is None:
                continue
            if isinstance(value, str):
                setattr(self, field,
                        value if os.path.isabs(value)
                        else os.path.normpath(os.path.join(self.base_dir, value)))
            else:
                setattr(self, field, [
                    p if os.path.isabs(p) else os.path.normpath(os.path.join(self.base_dir, p))
                    for p in value
                ])

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    def to_dict(self):
        return {k: getattr(self, k) for k in self.FIELDS}

    def merged(self, **overrides):
        """Return a new config with ``overrides`` applied on top."""
        raw = self.to_dict()
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return AttackConfig.from_dict(raw, base_dir=None)

    def log_prefix(self):
        """Reproduce the log-prefix naming scheme of the original scripts."""
        prefix = "main_" if self.mode == "single" else "main_arb_"
        prefix += self.attack_type
        if self.use_cw:
            prefix += "_cw"
        else:
            if self.use_faag:
                prefix += "_faag"
            if self.use_aro:
                prefix += f"_aro({self.aro_beta})"
            if self.use_preservation_loss:
                prefix += f"_alpha({self.alpha})"
        prefix += f"_tau({self.tau})"
        return prefix


def deep_update(base, extra):
    """Recursively merge dict ``extra`` into dict ``base``."""
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def parse_set_value(text):
    """Parse a ``--set key=value`` value with YAML scalar semantics."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text
