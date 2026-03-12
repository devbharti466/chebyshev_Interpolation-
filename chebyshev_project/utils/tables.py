"""CSV and tabulated output utilities."""

from __future__ import annotations

import csv
import os
from typing import Any


def results_to_csv(
    headers: list[str],
    data: list[list[Any]],
    filename: str,
) -> None:
    """Write results to a CSV file.

    Parameters
    ----------
    headers : list of str
        Column headers.
    data : list of list
        Rows of data; each row must have the same length as headers.
    filename : str
        Output file path. Parent directory must exist.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(data)


def print_summary_table(
    headers: list[str],
    data: list[list[Any]],
    title: str = "",
) -> None:
    """Print a formatted summary table to stdout.

    Parameters
    ----------
    headers : list of str
        Column headers.
    data : list of list
        Rows of data.
    title : str
        Optional title printed above the table.
    """
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    # Compute column widths
    col_widths = [len(str(h)) for h in headers]
    for row in data:
        for j, val in enumerate(row):
            col_widths[j] = max(col_widths[j], len(str(val)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"

    print(sep)
    print(fmt.format(*[str(h) for h in headers]))
    print(sep)
    for row in data:
        print(fmt.format(*[str(v) for v in row]))
    print(sep)
