"""
Shared utilities for all analysis scripts.

model loading, HF model extraction, test data loaders — used by every script.
"""
from __future__ import annotations
import json, sys, logging, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class _Tee:
    """Writes to multiple streams at once, flushing after every write so
    the file stays current even if the process crashes mid-run — a
    diagnostic log that's empty exactly when something goes wrong defeats
    the purpose. Used to mirror print() output into a log file without
    losing the live terminal view.

    Implements isatty() and a generic __getattr__ passthrough because
    replacing sys.stdout isn't just about write()/flush() — libraries
    (transformers' loading report, tqdm, rich, etc.) introspect stdout for
    things like isatty() (to decide whether to emit ANSI color codes) or
    fileno(). Without forwarding those to the real terminal stream, this
    wrapper crashes any code that checks them.
    """
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        # Report the REAL terminal's tty-ness (first stream = sys.__stdout__),
        # not the log file's — this is what callers actually want to know.
        return self._streams[0].isatty()

    def __getattr__(self, name):
        # Forward anything else (fileno, encoding, mode, ...) to the real
        # terminal stream so libraries that inspect stdout beyond
        # write/flush/isatty keep working transparently.
        return getattr(self._streams[0], name)


def setup_run_log(script_name: str, log_dir: str | Path = "results/analysis/logs") -> Path:
    """
    Capture this run's full output — both logging.*() calls AND bare
    print() — to a timestamped file under log_dir, in addition to the
    terminal. Call once near the top of a script's main().

    Why this exists: several analysis scripts print diagnostics that reveal
    silent computation failures (e.g. build_paft_cache.py's "writing null,
    not 0.0" warnings, or build_cache.py's per-entry fallback-to-null
    warnings). Grepping those after the fact is how several real bugs got
    confirmed fixed in this codebase — but only if the output was actually
    captured, which requires remembering to `tee` manually. This makes that
    automatic so it can't be forgotten.

    Implementation note: this adds a logging.FileHandler to the root logger
    rather than reassigning sys.stderr, because logging.StreamHandler (the
    one logging.basicConfig() installs by default) captures a reference to
    sys.stderr at construction time — reassigning sys.stderr afterward does
    NOT redirect a handler that already grabbed the old reference. Adding a
    second handler works regardless of when basicConfig() ran. Bare print()
    calls, by contrast, look up sys.stdout freshly on every call, so
    reassigning sys.stdout for those does work.

    Returns the log file path (also printed to confirm where it went).
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{script_name}_{timestamp}.log"

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    f = open(log_path, "a")
    sys.stdout = _Tee(sys.__stdout__, f)

    print(f"[setup_run_log] Full output for this run is also being saved to {log_path}")
    return log_path

def get_hf_model(method):
    """
    Extract the underlying HuggingFace-compatible model from any method.

    For lm_eval and generation, we need a model that responds to HF's
    forward(input_ids, attention_mask, ...) → logits interface.

    PAFT/SVF:   method.model is PAFTModel/SVFModel; .base is the GPT2LMHeadModel
                with adapted attention layers — this IS HF-compatible.
    LoRA:       method.model is a PeftModel; .base_model.model is GPT2LMHeadModel.
    Others:     method.model IS a GPT2LMHeadModel.
    """
    from paft.model.paft_model import PAFTModel
    from paft.model.svf_model  import SVFModel

    m = method.model
    if isinstance(m, (PAFTModel, SVFModel)):
        return m.base                        # GPT2LMHeadModel with adapted attn
    if hasattr(m, "base_model"):             # PeftModel (LoRA, PoLAR via PEFT)
        return m.base_model.model
    return m                                 # frozen, full_finetune, bitfit


def get_tokenizer(hf_name: str):
    tok = AutoTokenizer.from_pretrained(hf_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    return tok


# ──────────────────────────────────────────────────────────────────────────────
# Run discovery
# ──────────────────────────────────────────────────────────────────────────────

def discover_complete_runs(
    checkpoint_root: Path,
    model:    str,
    domains:  List[str],
    methods:  List[str],
) -> List[Tuple[str, str, Path]]:
    """Return (domain, method, run_dir) for every run with a sentinel file."""
    runs = []
    for domain in domains:
        for method in methods:
            run_dir = checkpoint_root / model / domain / method
            if (run_dir / "final" / "training_complete").exists():
                runs.append((domain, method, run_dir))
    return runs


def load_init_config(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "init" / "config.json"
    with path.open() as f:
        return json.load(f)

# Long-form metric keys matching values written to metrics.json.
# Must match build_cache.py's TASK_PRIMARY exactly.
TASK_PRIMARY = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "f1",
    "qnli": "accuracy",
    "qqp":  "f1",
    "rte":  "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson",
}

def load_adapted_weights(run_dir: Path | str, tag: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Loads adapted_weights.pt from a specific tag checkpoint.
    Flattens each layer's head dimensions [H, n, d] -> [H*n, d] for 2D matrix analysis.
    Returns: (W_V_layers, W_O_layers) as lists of 2D tensors.
    """
    path = Path(run_dir) / tag / "adapted_weights.pt"
    if not path.exists():
        return [], []

    ckpt = torch.load(path, map_location="cpu")

    W_V_layers = []
    if "W_V" in ckpt:
        for W in ckpt["W_V"]:
            # Flatten [H, n, d] -> [H*n, d]
            W_V_layers.append(W.reshape(-1, W.shape[-1]))

    W_O_layers = []
    if "W_O" in ckpt:
        for W in ckpt["W_O"]:
            # Flatten [H, d, n] -> [H*d, n]
            W_O_layers.append(W.reshape(-1, W.shape[-1]))

    return W_V_layers, W_O_layers