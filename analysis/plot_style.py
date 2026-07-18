"""
analysis/plot_style.py
Shared matplotlib style for all PAFT paper figures.
Import at the top of every plotting script:
    from analysis.plot_style import apply_style, COLORS, STYLES, METHOD_LABELS

Design language: one consistent ink color for all chrome (spines, ticks,
labels) instead of several different greys competing with each other;
sans-serif figure text against the serif paper body; a single accent color
for the paper's headline method with everything else in clean, clearly-
named hues (real blue, real green, real purple) rather than desaturated
in-between colors that read as muddy or accidental. Reference/baseline
lines (Frozen, Full FT) stay neutral on purpose — that's a deliberate
"floor/ceiling" signal, not a leftover color.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import logging

# Matplotlib's font-manager warns loudly every time a requested font isn't
# installed and it falls back to a bundled default (DejaVu Sans). That
# fallback is expected and harmless — figures still render correctly — so
# don't let it spam every pipeline run's log. If you want to confirm which
# font actually rendered, check a saved PDF's embedded font metadata instead.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ── Single source of truth for chrome ──────────────────────────────────────────
# One ink color for everything that isn't data (spines, ticks, axis labels),
# one light neutral for gridlines. Previously these were four unrelated
# greys (#888888, #666666, #333333, #cccccc) that never quite matched each
# other — that mismatch is what reads as "weird greys."

INK       = "#3C4043"   # charcoal, not flat black — used for all chrome
GRID_LINE = "#EAEAEA"   # single light neutral, gridlines only


def apply_style():
    mpl.rcParams.update({
        # Font — clean sans-serif for figures, contrasts with serif paper body.
        # Falls back down this list to whatever's installed; on most
        # machines this resolves to a Helvetica/Arial-equivalent.
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Helvetica Neue", "Arial", "Liberation Sans",
                                  "DejaVu Sans"],
        "mathtext.fontset":     "custom",
        "mathtext.rm":          "Liberation Sans",
        "mathtext.it":          "Liberation Sans:italic",
        "mathtext.bf":          "Liberation Sans:bold",
        "mathtext.cal":         "Liberation Sans:italic",   # was unset — matplotlib's
                                                              # "custom" fontset defaults
                                                              # any unset sub-family to
                                                              # 'cursive', which isn't
                                                              # installed anywhere and
                                                              # fires a warning on every
                                                              # figure. Not currently used
                                                              # by any \mathcal{} in these
                                                              # plots, but must still be
                                                              # set to something real.
        "font.size":            9,

        # Axes
        "axes.labelsize":       9,
        "axes.labelcolor":      INK,
        "axes.titlesize":       9,
        "axes.titleweight":     "normal",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.edgecolor":       INK,
        "axes.linewidth":       0.7,
        "axes.labelpad":        6,

        # Ticks — same ink as everything else, not a separate grey
        "xtick.color":          INK,
        "ytick.color":          INK,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "xtick.major.size":     3,
        "ytick.major.size":     3,
        "xtick.major.width":    0.7,
        "ytick.major.width":    0.7,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.top":            False,
        "ytick.right":          False,

        # Grid — horizontal only, single light neutral, no separate alpha
        # blending needed since the color itself is already light
        "axes.grid":            True,
        "axes.grid.axis":       "y",
        "grid.alpha":           1.0,
        "grid.linewidth":       0.6,
        "grid.color":           GRID_LINE,

        # Lines — rounded caps/joins read less "default matplotlib"
        "lines.linewidth":      1.5,
        "lines.markersize":     4.5,
        "lines.markeredgewidth":0.6,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle":"round",
        "lines.dash_capstyle":  "round",

        # Legend — frameless by default; pass frameon=True on the rare plot
        # where a boxed legend genuinely helps contrast against busy data
        "legend.fontsize":      7.5,
        "legend.frameon":       False,
        "legend.borderpad":     0.3,
        "legend.labelspacing":  0.4,
        "legend.handlelength":  1.6,
        "legend.handletextpad": 0.5,

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
# One accent color for the headline method; everything else is a clean,
# clearly-named hue (real blue, real green, real purple) at a muted
# saturation — NOT a desaturated blend toward grey. A "blue-grey" or
# "green-grey" looks like an accident; a muted blue or muted green still
# reads as intentional. Frozen and Full FT are the two exceptions, kept
# genuinely neutral on purpose since they're floor/ceiling reference
# lines, not "just another method."

ACCENT = "#CC785C"   # warm clay/terracotta — the one saturated color in
                      # the whole palette, reserved for the headline result

COLORS = {
    # Headline method
    "safe_hybrid_paft":   ACCENT,

    # Other PAFT variants — tints of the accent hue, so they read as
    # "part of the same family" without competing with the headline color
    "hybrid_paft":         "#E3A98D",
    "safe_pure_paft":      "#A85A3F",
    "pure_paft":           "#F0CFBE",

    # Additive baselines — clean blue family (not blue-grey)
    "lora_r8":             "#4C72B0",
    "lora_r64":             "#2E4F73",
    "polar_r8":             "#8FB3DE",

    # Non-additive baselines — clean, distinct hues, not grey-blended
    "bitfit":              "#59A14F",   # clean green
    "svf":                 "#8C6BB1",   # clean purple

    # Deliberately neutral — floor/ceiling reference methods, not "leftover"
    # colors. Full FT = upper bound (trains everything); Frozen = lower
    # bound (trains nothing). Neutral ink is the correct signal here.
    "full_ft":             "#2B2B2B",
    "frozen":              "#ADABA5",   # warm light neutral, not cold grey

    # Reference lines (e.g. pretrained baseline)
    "pretrained":          "#D6D3CC",
}


# ── Marker styles ─────────────────────────────────────────────────────────────
# Unchanged — already solid: every method has a unique marker so figures
# survive greyscale printing and colorblind readers.

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
# Unchanged.

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
# Unchanged.

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
# Unchanged — sizing wasn't the problem.

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
        color     = COLORS.get(method, INK),
        marker    = MARKERS.get(method, "o"),
        linestyle = LINESTYLES.get(method, "-"),
        label     = label or METHOD_LABELS.get(method, method),
        markevery = max(1, len(x) // 8),  # don't crowd markers
        zorder    = 3 if method == "safe_hybrid_paft" else 2,
    )
    defaults.update(kwargs)
    return ax.plot(x, y, **defaults)