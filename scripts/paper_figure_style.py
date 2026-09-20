"""Shared appearance for the current paper figures."""
PALETTE = {
    "background": "#ffffff",
    "chart_background": "#f6f2e8",
    "grid": "#ded7c8",
    "ink": "#274753",
    "teal": "#299d8f",
    "green": "#8ab07c",
    "deep_teal": "#287779",
}
MODEL_COLORS = dict(zip(
    ["diffusion", "autoregressive", "cvae", "flow_matching"],
    [PALETTE["ink"], PALETTE["green"], PALETTE["teal"], PALETTE["deep_teal"]],
))
def apply_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": PALETTE["chart_background"],
        "grid.color": PALETTE["grid"],
        "text.color": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "axes.edgecolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 450,
    })
