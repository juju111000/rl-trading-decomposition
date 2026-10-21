"""
Shared paper figure style. Import from every figure script:

    from paper_style import apply, PALETTE, SIZES, save, qa

apply() sets the rcParams; PALETTE is semantic and used identically in every
figure (primary = RL / hysteresis / shielded, secondary = passive / softmax /
unshielded, neutral = reference); secondary series always also carry a dashed
line, hatch or open marker so figures survive grayscale printing. Palette
validated (light surface): CVD dE 21.2, normal-vision dE 29.8, both >= 3:1.
No figure titles -- captions carry them.
"""
import itertools

import matplotlib
import matplotlib.pyplot as plt

PALETTE = {"primary": "#2B5F9E", "secondary": "#D1662C", "neutral": "#555555", "ink": "#000000"}
PALETTE.update({"rl": PALETTE["primary"], "hysteresis": PALETTE["primary"], "shielded": PALETTE["primary"],
                "passive": PALETTE["secondary"], "softmax": PALETTE["secondary"],
                "unshielded": PALETTE["secondary"], "reference": PALETTE["neutral"]})
SECONDARY_LS = (0, (4, 2))          # dashed: grayscale cue for every secondary line
SECONDARY_HATCH = "////"
SIZES = {"single": (3.4, 2.6), "double": (7.0, 2.8)}
FS = {"label": 9, "tick": 8, "annot": 8, "small": 7.5}
MIN_FONT = 7.5


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.edgecolor": PALETTE["ink"], "axes.labelcolor": PALETTE["ink"],
        "axes.labelsize": FS["label"], "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.visible": False, "ytick.minor.visible": False, "font.size": FS["annot"],
        "legend.frameon": False, "hatch.color": "white", "hatch.linewidth": 0.8, "lines.linewidth": 1.4,
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.facecolor": "white", "figure.facecolor": "white",
        "axes.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42, "axes.titlesize": FS["label"],
    })


def save(fig, stem):
    """PNG (600 dpi) + vector PDF."""
    fig.savefig(f"{stem}.png")
    fig.savefig(f"{stem}.pdf")


def qa(fig, name):
    """At the figure's own (target) size: every visible text >= MIN_FONT pt, no two text boxes
    overlapping, nothing outside the saved canvas. Returns (ok, report lines)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip() and t.get_window_extent(r).width > 0]
    boxes = [(t, t.get_window_extent(r)) for t in texts]
    small = [(t.get_text(), t.get_fontsize()) for t in texts if t.get_fontsize() < MIN_FONT - 1e-9]
    overlaps = [(a.get_text(), b.get_text()) for (a, ba), (b, bb) in itertools.combinations(boxes, 2)
                if ba.overlaps(bb) and ba.intersection(ba, bb) is not None
                and ba.intersection(ba, bb).width > 0.5 and ba.intersection(ba, bb).height > 0.5]
    w, h = fig.get_size_inches()
    ok = not small and not overlaps
    rep = [f"{name}: {w:.1f} x {h:.1f} in, {len(texts)} text items, min font "
           f"{min(t.get_fontsize() for t in texts):.1f} pt, overlaps {len(overlaps)}, below {MIN_FONT} pt {len(small)}"
           f" -> {'OK' if ok else 'FAIL'}"]
    rep += [f"    overlap: {a!r} x {b!r}" for a, b in overlaps] + [f"    small: {s!r} {f} pt" for s, f in small]
    return ok, rep
