"""Shared chart style so every README graph reads as one system."""

import platform
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e6e5e0", "#fcfcfb"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "font.size": 10,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        }
    )


def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        import subprocess

        if platform.system() == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            cpu = out.stdout.strip() or cpu
    except Exception:
        pass
    return cpu


def footer(model: str, dtype: str, runs: str) -> str:
    return f"{hardware()} · {model} · {dtype} · {runs} · {date.today().isoformat()}"
