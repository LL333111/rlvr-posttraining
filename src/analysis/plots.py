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
    rolling_window: int | None = None,
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
        x_values = [row[x] for row in rows]
        y_values = [float(row[y]) for row in rows]
        if rolling_window and rolling_window > 1:
            axis.plot(x_values, y_values, linewidth=1, alpha=0.3, label="Per-step reward")
            rolling = [
                sum(y_values[max(0, index - rolling_window + 1) : index + 1])
                / min(index + 1, rolling_window)
                for index in range(len(y_values))
            ]
            axis.plot(
                x_values,
                rolling,
                linewidth=2.5,
                label=f"{rolling_window}-step rolling mean",
            )
            axis.legend(frameon=False)
        else:
            axis.plot(x_values, y_values, linewidth=2)
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
