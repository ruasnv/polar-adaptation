"""
analysis/plot_style.py
Shared matplotlib style for all PAFT paper figures.
Import at the top of every plotting script:
    from analysis.plot_style import apply_style, COLORS, STYLES, METHOD_LABELS
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Core style ────────────────────────────────────────────────────────────────

def apply_style():
    mpl.rcParams.update({
        # Font — matches Computer Modern used in LaTeX papers
        "font.family":          "serif",
        "font.serif":           ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset":     "cm",
        "font.size":            9,

        # Axes
        "axes.labelsize":       9,
        "axes.titlesize":       9,
        "axes.titleweight":     "normal",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       0.8,
        "axes.labelpad":        4,

        # Ticks
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "xtick.major.size":     3,
        "ytick.major.size":     3,
        "xtick.major.width":    0.8,
        "ytick.major.width":    0.8,
        "xtick.direction":      "out",
        "ytick.direction":      "out",

        # Grid
        "axes.grid":            True,
        "grid.alpha":           0.25,
        "grid.linewidth":       0.5,
        "grid.color":           "#cccccc",

        # Lines
        "lines.linewidth":      1.5,
        "lines.markersize":     4.5,
        "lines.markeredgewidth":0.6,

        # Legend
        "legend.fontsize":      7.5,
        "legend.framealpha":    0.92,
        "legend.edgecolor":     "#cccccc",
        "legend.borderpad":     0.4,
        "legend.labelspacing":  0.3,
        "legend.handlelength":  1.5,
        "legend.handletextpad": 0.4,

        # Figure
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.02,

        # Misc
        "patch.linewidth":      0.6,
        "hatch.linewidth":      0.6,
    })


# ── Color palette ─────────────────────────────────────────────────────────────
# Muted, distinguishable in color and greyscale, consistent across all figures

COLORS = {
    # PAFT family — blues
    "pure_paft":          "#3182bd",   # medium blue
    "hybrid_paft":        "#08519c",   # dark blue
    "safe_pure_paft":     "#6baed6",   # light blue
    "safe_hybrid_paft":   "#08306b",   # navy

    # Additive baselines — warm
    "lora_r8":            "#e6550d",   # orange-red
    "lora_r64":           "#a63603",   # dark orange
    "polar_r8":           "#fd8d3c",   # light orange

    # Non-additive baselines — neutral
    "bitfit":             "#31a354",   # green
    "svf":                "#756bb1",   # purple
    "full_ft":            "#252525",   # near black
    "frozen":             "#969696",   # grey

    # Reference lines
    "pretrained":         "#bdbdbd",   # light grey dashed
}


# ── Marker styles ─────────────────────────────────────────────────────────────
# Every method gets a unique marker so figures work in greyscale print

MARKERS = {
    "pure_paft":          "o",
    "hybrid_paft":        "s",
    "safe_pure_paft":     "^",
    "safe_hybrid_paft":   "D",
    "lora_r8":            "v",
    "lora_r64":           "<",
    "polar_r8":           ">",
    "bitfit":             "P",
    "svf":                "X",
    "full_ft":            "*",
    "frozen":             "h",
}


# ── Line styles ───────────────────────────────────────────────────────────────

LINESTYLES = {
    "pure_paft":          "-",
    "hybrid_paft":        "-",
    "safe_pure_paft":     "-",
    "safe_hybrid_paft":   "-",
    "lora_r8":            "--",
    "lora_r64":           "--",
    "polar_r8":           "--",
    "bitfit":             ":",
    "svf":                ":",
    "full_ft":            (0, (3, 1, 1, 1)),  # dash-dot
    "frozen":             ":",
}


# ── Clean method labels for legends and axes ──────────────────────────────────

METHOD_LABELS = {
    "frozen":             "Frozen",
    "pure_paft":          "pure-PAFT (Ours)",
    "hybrid_paft":        "hybrid-PAFT (Ours)",
    "safe_pure_paft":     "safe-pure-PAFT (Ours)",
    "safe_hybrid_paft":   "safe-hybrid-PAFT (Ours)",
    "lora_r8":            "LoRA $r{=}8$",
    "lora_r64":           "LoRA $r{=}64$",
    "polar_r8":           "PoLAR $r{=}8$",
    "bitfit":             "BitFit",
    "svf":                "SVF",
    "full_ft":            "Full FT",
}

# Short labels for space-constrained plots
METHOD_LABELS_SHORT = {
    "frozen":             "Frozen",
    "pure_paft":          "pure-PAFT",
    "hybrid_paft":        "hybrid-PAFT",
    "safe_pure_paft":     "safe-pure-PAFT",
    "safe_hybrid_paft":   "safe-hybrid-PAFT",
    "lora_r8":            "LoRA $r{=}8$",
    "lora_r64":           "LoRA $r{=}64$",
    "polar_r8":           "PoLAR $r{=}8$",
    "bitfit":             "BitFit",
    "svf":                "SVF",
    "full_ft":            "Full FT",
}


# ── Figure size helpers ───────────────────────────────────────────────────────
# JMLR uses 6.5" text width. Single column = half. Double = full.

def fig_single(height_ratio=0.75):
    """Single-column figure: 3.25" wide."""
    return plt.subplots(figsize=(3.25, 3.25 * height_ratio))

def fig_double(height_ratio=0.45):
    """Double-column figure: 6.5" wide."""
    return plt.subplots(figsize=(6.5, 6.5 * height_ratio))

def fig_double_subplots(ncols=2, height_ratio=0.45):
    """Double-column figure with subplots."""
    w = 6.5
    return plt.subplots(1, ncols, figsize=(w, w * height_ratio))


# ── Convenience: plot one method line ────────────────────────────────────────

def plot_method(ax, x, y, method, label=None, **kwargs):
    """Plot a single method with consistent color, marker, linestyle."""
    defaults = dict(
        color     = COLORS.get(method, "#333333"),
        marker    = MARKERS.get(method, "o"),
        linestyle = LINESTYLES.get(method, "-"),
        label     = label or METHOD_LABELS.get(method, method),
        markevery = max(1, len(x) // 8),  # don't crowd markers
        zorder    = 3 if "paft" in method else 2,
    )
    defaults.update(kwargs)
    return ax.plot(x, y, **defaults)
