"""
combine_heatmaps.py

Combines all heatmap PNGs into publication-worthy figures.
Layout: 4 configs (rows) x 2 seeds (cols) per page.
If more than 4 configs exist, spills into page_2, page_3, etc.

Usage:
    python combine_heatmaps.py --root /path/to/your/lamda_heatmaps/folder
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ── publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

ROWS_PER_PAGE = 4   # configs per page
CELL_W        = 7.0 # inches per seed column
CELL_H        = 4.0 # inches per config row

# ── filename pattern ──────────────────────────────────────────────────────────
PATTERN = re.compile(
    r"(?P<model>.+?)_kv(?P<kv>\d+)_seq(?P<seq>\d+)_seed(?P<seed>\d+)_(?P<suffix>.+)\.png"
)

def parse_name(path: Path):
    m = PATTERN.match(path.name)
    return m.groupdict() if m else None

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", s)]

def collect_images(root: Path):
    """Groups PNGs by: suffix -> (model, kv, seq) -> seed -> Path"""
    groups = defaultdict(lambda: defaultdict(dict))
    for png in sorted(root.rglob("*.png")):
        # skip previously generated combined figures
        if "combined_figures" in str(png):
            continue
        info = parse_name(png)
        if info is None:
            continue
        key    = (info["model"], info["kv"], info["seq"])
        seed   = info["seed"]
        suffix = info["suffix"]
        groups[suffix][key][seed] = png
    return groups


def make_page(configs, seeds, data, suffix, page_num, total_pages, out_dir, pretty_suffix):
    """Render one page: up to ROWS_PER_PAGE configs x len(seeds) seeds."""
    n_rows = len(configs)
    n_cols = len(seeds)

    fig_w = CELL_W * n_cols + 1.2   # +1.2 for row labels
    fig_h = CELL_H * n_rows + 0.9   # +0.9 for suptitle

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={"hspace": 0.08, "wspace": 0.04},
    )

    # column headers (seed labels) on first row only
    for c, seed in enumerate(seeds):
        axes[0][c].set_title(f"seed = {seed}", fontsize=13,
                             fontweight="bold", pad=8)

    for r, cfg in enumerate(configs):
        model, kv, seq = cfg
        seed_map = data[cfg]

        for c, seed in enumerate(seeds):
            ax = axes[r][c]
            ax.axis("off")

            if seed in seed_map:
                img = mpimg.imread(seed_map[seed])
                ax.imshow(img, aspect="auto")
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=10,
                        style="italic")

            # row label on the very left
            if c == 0:
                label = f"{model}\nkv={kv}  seq={seq}"
                ax.text(-0.14, 0.5, label,
                        transform=ax.transAxes,
                        fontsize=11, fontweight="bold",
                        va="center", ha="right",
                        rotation=0,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="#f0f0f0",
                                  edgecolor="#cccccc",
                                  linewidth=0.8))

    # page title
    page_info = f"  (page {page_num}/{total_pages})" if total_pages > 1 else ""
    fig.suptitle(f"{pretty_suffix}{page_info}",
                 fontsize=15, fontweight="bold", y=1.005)

    plt.tight_layout(rect=[0.13, 0, 1, 1])

    page_tag  = f"_page{page_num}" if total_pages > 1 else ""
    out_path  = out_dir / f"combined_{suffix}{page_tag}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"    ✓  saved → {out_path.name}")


def make_combined_figure(suffix: str, data: dict, out_dir: Path):
    configs = sorted(data.keys(), key=lambda x: natural_sort_key("_".join(x)))
    seeds   = sorted({s for cfg in data.values() for s in cfg}, key=natural_sort_key)

    if not configs or not seeds:
        print(f"  [skip] no data for '{suffix}'")
        return

    pretty_suffix = suffix.replace("_", " ").title()

    # paginate: chunks of ROWS_PER_PAGE
    pages = [configs[i:i + ROWS_PER_PAGE] for i in range(0, len(configs), ROWS_PER_PAGE)]
    total = len(pages)

    print(f"  → '{suffix}'  |  {len(configs)} configs  |  {total} page(s)")
    for p_idx, page_configs in enumerate(pages, start=1):
        make_page(page_configs, seeds, data, suffix,
                  p_idx, total, out_dir, pretty_suffix)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".",
                        help="Root folder containing all config sub-folders.")
    args   = parser.parse_args()

    root    = Path(args.root).resolve()
    out_dir = root / "combined_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂  Scanning: {root}")
    groups = collect_images(root)

    if not groups:
        print("⚠️  No matching PNG files found.")
        return

    print(f"📊  Found {len(groups)} heatmap type(s): {list(groups.keys())}\n")

    for suffix, data in groups.items():
        make_combined_figure(suffix, data, out_dir)

    print(f"\n🎉  Done! Figures saved in: {out_dir}")


if __name__ == "__main__":
    main()