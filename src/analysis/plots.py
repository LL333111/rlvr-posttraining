from __future__ import annotations

from pathlib import Path
from typing import Any


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def line_plot(
    series: list[dict[str, Any]],
    *,
    x: str,
    y: str,
    group: str | None,
    xlabel: str,
    ylabel: str,
    title: str,
    output: str | Path,
) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    if group:
        labels = sorted({str(row[group]) for row in series})
        for label in labels:
            rows = [row for row in series if str(row[group]) == label]
            rows.sort(key=lambda row: float(row[x]))
            axis.plot([row[x] for row in rows], [row[y] for row in rows], marker="o", label=label)
        axis.legend(frameon=False)
    else:
        rows = sorted(series, key=lambda row: float(row[x]))
        axis.plot([row[x] for row in rows], [row[y] for row in rows], linewidth=2)
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def grouped_bar(
    labels: list[str],
    values: list[float],
    *,
    ylabel: str,
    title: str,
    output: str | Path,
) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.bar(labels, values, color=["#697386", "#4c78a8", "#59a14f"][: len(labels)])
    axis.set(ylabel=ylabel, title=title)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
