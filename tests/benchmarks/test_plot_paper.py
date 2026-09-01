from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.text import Text

from scripts.benchmarks.plot_paper import (
    exclude_tagged_cases,
    plot_outcome_overview,
)


def test_exclude_tagged_cases_removes_every_backend_row_for_a_case() -> None:
    rows = [
        {"case_id": "linear", "backend": "wfomc", "tags": "set;sequence"},
        {"case_id": "linear", "backend": "asp", "tags": "set;sequence"},
        {"case_id": "round", "backend": "wfomc", "tags": "set;circle"},
        {"case_id": "round", "backend": "asp", "tags": "set;circle"},
    ]

    filtered = exclude_tagged_cases(rows, {"circle"})

    assert filtered == rows[:2]


def test_exclude_tagged_cases_requires_an_exact_tag_match() -> None:
    rows = [
        {"case_id": "shape", "backend": "wfomc", "tags": "set;circle_shape"},
    ]

    assert exclude_tagged_cases(rows, {"circle"}) == rows


def test_outcome_overview_legend_does_not_overlap_panel_titles() -> None:
    rows = [
        {
            "case_id": "case",
            "backend": backend,
            "status": "solved",
            "elapsed_sec": "1.0",
        }
        for backend in ("wfomc", "propositionalwfomc", "coso", "asp", "essence")
    ]

    figure = plot_outcome_overview(rows)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legend_box = figure.legends[0].get_window_extent(renderer)
    title_texts = {
        axis.get_title(loc="left")
        for axis in figure.axes
        if axis.get_title(loc="left")
    }
    panel_titles = [
        child
        for axis in figure.axes
        for child in axis.get_children()
        if isinstance(child, Text) and child.get_text() in title_texts
    ]

    assert panel_titles
    assert all(
        not legend_box.overlaps(title.get_window_extent(renderer))
        for title in panel_titles
    )
    plt.close(figure)
