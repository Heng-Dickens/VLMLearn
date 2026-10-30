#!/usr/bin/env python3
"""Rebuild three exact teaching diagrams, without dataset imagery."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "docs/assets"
INK = "#18324a"
BLUE = "#e7f0ff"
GREEN = "#e5f6ef"
ORANGE = "#fff0dc"


def canvas(title, subtitle, width=16, height=8):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("#ffffff")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.text(.3, 7.55, title, fontsize=22, weight="bold", color=INK)
    ax.text(.3, 7.08, subtitle, fontsize=11.5, color="#526579")
    fig.subplots_adjust(left=.015, right=.985, bottom=.03, top=.97)
    return fig, ax


def box(ax, x, y, text, width=3.2, height=1.15, color=BLUE, size=12):
    patch = FancyBboxPatch((x-width/2, y-height/2), width, height,
        boxstyle="round,pad=0.10,rounding_size=0.12", linewidth=1.4,
        edgecolor="#7191aa", facecolor=color, zorder=3)
    ax.add_patch(patch)
    ax.text(x, y, text, va="center", ha="center", fontsize=size, color=INK,
            linespacing=1.45, zorder=4)


def arrow(ax, a, b, label=None, rad=0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16,
        linewidth=1.7, color="#597b98", connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+.18, label, ha="center", fontsize=10, color=INK)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=140, facecolor="white")
    fig.savefig(OUT / f"{name}.svg", facecolor="white")
    plt.close(fig)


def technical_map():
    fig, ax = canvas("From vision-language learning to reliable boxes",
        "Model family, detection task and prompt modality are different axes.")
    box(ax, 2, 4.2, "Image-text representations\nCLIP / ALBEF / BLIP", height=1.6)
    box(ax, 7, 5.6, "Region-level grounding\nGrounding DINO / WeDetect", width=4)
    box(ax, 7, 2.5, "Visual tokens + LLM\nBLIP-2 / MiniGPT-4 / LLaVA", width=4, color=GREEN)
    box(ax, 13, 5.6, "Visual-prompt detectors\nYOLOE / PET-DINO", width=4, color=ORANGE)
    box(ax, 13, 2.5, "Direct boxes or review\nQwen experiment workflow", width=4, color=GREEN)
    arrow(ax, (3.7, 4.7), (4.85, 5.6))
    arrow(ax, (3.7, 3.6), (4.85, 2.5))
    arrow(ax, (9.15, 5.6), (10.85, 5.6), "visual cues")
    arrow(ax, (9.15, 2.5), (10.85, 2.5), "instructions")
    arrow(ax, (13, 4.9), (13, 3.2), "candidates")
    ax.text(.3, .5, "Learning map, not a strict model genealogy.  Goal: few labels, high precision, accurate box boundaries.",
            fontsize=12, color=INK)
    save(fig, "technical_map")


def savpe_flow():
    fig, ax = canvas("SAVPE: semantic features + spatial activation",
        "Pinned YOLOE-v8/11 implementation; D = 512, A = 16. Batch and reference-instance axes omitted.")
    box(ax, 2, 5.6, "Reference image features\nP3 / P4 / P5", width=3.2)
    box(ax, 6, 5.9, "Semantic branch\nS: 512 x H x W\ncv1 + cv3", height=1.45)
    box(ax, 6, 3.8, "Image activation FI\n16 x H x W\ncv2 + cv4", height=1.4)
    box(ax, 2, 1.8, "Box-derived binary mask\n1 x H x W\nsame image coordinates", height=1.45, color=GREEN)
    box(ax, 6, 1.8, "Mask features FV\n16 x H x W\ncv5", height=1.4, color=GREEN)
    box(ax, 10.3, 5.9, "Split S into 16 groups\n32 channels per group", width=3.3)
    box(ax, 10.3, 3.5, "Concat FI + FV: 32 ch\ncv6 -> 16 ch\nmasked spatial softmax", width=3.3, height=1.6, color=ORANGE)
    box(ax, 14, 4.75, "Weighted sums\n16 x 32 -> 512\nL2-normalized p", width=2.9, height=1.6, color=GREEN)
    arrow(ax, (3.7, 5.7), (4.25, 5.9))
    arrow(ax, (3.65, 5.2), (4.25, 3.9))
    arrow(ax, (3.7, 1.8), (4.25, 1.8))
    arrow(ax, (7.75, 5.9), (8.5, 5.9))
    arrow(ax, (7.75, 3.8), (8.5, 3.8))
    arrow(ax, (7.75, 1.8), (9.6, 2.6))
    arrow(ax, (12.05, 5.9), (12.5, 5.3))
    arrow(ax, (12.05, 3.5), (12.5, 4.1))
    ax.text(.3, .35, "Two different 32s: activation concat = 16 + 16; semantic group width = 512 / 16. Output stays 512-D.",
            fontsize=11.5, color=INK)
    save(fig, "savpe_flow")


def pet_dino_flow():
    fig, ax = canvas("PET-DINO: visual cues enter selection and decoding",
        "Swin-T configuration: 256-D features, 900 queries, 6 decoder layers. N depends on image size.")
    box(ax, 2.2, 5.7, "Reference image + boxes\nimage features + positions", width=3.7, color=GREEN)
    box(ax, 7.5, 5.7, "AFVPG\nvisual prompt vectors V", width=3.6, color=GREEN)
    box(ax, 2.2, 2.8, "Target image\nmulti-scale encoder\nfeatures: N x 256", width=3.7, height=1.5)
    box(ax, 7.5, 2.8, "Prompt scores + proposals\nTop-K selection\nR0: 900 x 4", width=3.6, height=1.5, color=ORANGE)
    box(ax, 13, 3.6, "Decoder\nimage + prompt attention\niterative box refinement", width=4, height=1.6)
    box(ax, 13, 1.3, "Boxes + class scores", width=4, height=.8, color=GREEN)
    arrow(ax, (4.15, 5.7), (5.6, 5.7))
    arrow(ax, (4.15, 2.8), (5.6, 2.8))
    arrow(ax, (7.5, 5.05), (7.5, 3.65))
    arrow(ax, (9.4, 5.7), (12.8, 4.5))
    arrow(ax, (9.4, 2.8), (10.85, 3.3))
    arrow(ax, (13, 2.7), (13, 1.8))
    ax.text(.3, .3, "R0 (box coordinates) and content queries are separate. IBP / DMD are training strategies, omitted here.",
            fontsize=11.5, color=INK)
    save(fig, "pet_dino_flow")


if __name__ == "__main__":
    technical_map()
    savpe_flow()
    pet_dino_flow()
