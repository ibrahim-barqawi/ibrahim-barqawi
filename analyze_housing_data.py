"""Exploratory analysis for the provided housing price data set.

The script works with a tab-separated values file that contains the columns
listed in the user prompt (price, area, bedrooms, ... furnishingstatus).

Because the execution environment does not have internet access and cannot
install external dependencies, the implementation relies solely on the Python
standard library.  You can point the tool at the full data set supplied in the
prompt, or use the bundled ``data/sample_housing.tsv`` file for a quick smoke
check.

Example
-------
Run the module directly to print a compact report summarising the numeric
features, the prevalence of boolean amenities, grouped pricing statistics, and
a simple correlation matrix::

    python analyze_housing_data.py                   # uses the sample file
    python analyze_housing_data.py path/to/data.tsv  # analyse a custom file
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, quantiles, stdev
from typing import Dict, Iterable, List, Sequence

DEFAULT_DATA_PATH = Path("data/sample_housing.tsv")

BOOL_COLUMNS = (
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
)

CATEGORY_COLUMNS = ("furnishingstatus",)

NUMERIC_COLUMNS = (
    "price",
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking",
)


@dataclass
class NumericSummary:
    count: int
    mean: float
    std: float
    min: float
    q1: float
    median: float
    q3: float
    max: float


def load_housing_data(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return list(reader)


def _to_float(value: str) -> float:
    return float(value.strip())


def normalise_records(records: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    cleaned: List[Dict[str, object]] = []
    for row in records:
        normalised: Dict[str, object] = {}
        for column, value in row.items():
            text = value.strip()
            if column in NUMERIC_COLUMNS:
                normalised[column] = _to_float(text)
            elif column in BOOL_COLUMNS:
                normalised[column] = text.lower() == "yes"
            else:
                normalised[column] = text
        cleaned.append(normalised)
    return cleaned


def summarise_numeric(values: Sequence[float]) -> NumericSummary:
    ordered = sorted(values)
    q1, q2, q3 = quantiles(ordered, n=4, method="inclusive")
    return NumericSummary(
        count=len(ordered),
        mean=mean(ordered),
        std=stdev(ordered) if len(ordered) > 1 else 0.0,
        min=ordered[0],
        q1=q1,
        median=q2,
        q3=q3,
        max=ordered[-1],
    )


def boolean_feature_share(rows: Sequence[Dict[str, object]], column: str) -> float:
    true_count = sum(1 for row in rows if row[column])
    return true_count / len(rows)


def price_by_category(rows: Sequence[Dict[str, object]], column: str) -> List[Dict[str, float]]:
    groups: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        key = str(row[column])
        groups[key].append(row["price"])

    summary: List[Dict[str, float]] = []
    for key, prices in groups.items():
        ordered = sorted(prices)
        summary.append(
            {
                column: key,
                "count": len(prices),
                "mean": mean(prices),
                "median": median(prices),
                "min": ordered[0],
                "max": ordered[-1],
            }
        )
    summary.sort(key=lambda entry: entry["mean"], reverse=True)
    return summary


def correlation(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        raise ValueError("Input sequences must have the same length")
    if len(x) < 2:
        return 0.0

    mean_x = mean(x)
    mean_y = mean(y)
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return numerator / (denom_x * denom_y)


def correlation_matrix(rows: Sequence[Dict[str, object]], columns: Iterable[str]) -> Dict[str, Dict[str, float]]:
    numeric_lists = {col: [row[col] for row in rows] for col in columns}
    matrix: Dict[str, Dict[str, float]] = {}
    for col_x in columns:
        matrix[col_x] = {}
        for col_y in columns:
            matrix[col_x][col_y] = correlation(numeric_lists[col_x], numeric_lists[col_y])
    return matrix


def print_numeric_summary(rows: Sequence[Dict[str, object]]) -> None:
    print("Numeric feature summary:")
    headers = (
        "count",
        "mean",
        "std",
        "min",
        "q1",
        "median",
        "q3",
        "max",
    )
    print("\t".join(["feature", *headers]))
    for column in NUMERIC_COLUMNS:
        summary = summarise_numeric([row[column] for row in rows])
        values = [
            f"{summary.count}",
            f"{summary.mean:.2f}",
            f"{summary.std:.2f}",
            f"{summary.min:.2f}",
            f"{summary.q1:.2f}",
            f"{summary.median:.2f}",
            f"{summary.q3:.2f}",
            f"{summary.max:.2f}",
        ]
        print("\t".join([column, *values]))
    print()


def print_boolean_summary(rows: Sequence[Dict[str, object]]) -> None:
    print("Amenity prevalence (share of listings with value=True):")
    for column in BOOL_COLUMNS:
        share = boolean_feature_share(rows, column)
        print(f"  {column:17s}: {share * 100:5.1f}%")
    print()


def print_price_by_category(rows: Sequence[Dict[str, object]]) -> None:
    for column in CATEGORY_COLUMNS:
        print(f"Price distribution by {column}:")
        stats = price_by_category(rows, column)
        print("\t".join([column, "count", "mean", "median", "min", "max"]))
        for entry in stats:
            print(
                "\t".join(
                    [
                        entry[column],
                        f"{entry['count']}",
                        f"{entry['mean']:.2f}",
                        f"{entry['median']:.2f}",
                        f"{entry['min']:.2f}",
                        f"{entry['max']:.2f}",
                    ]
                )
            )
        print()


def print_correlation_matrix(rows: Sequence[Dict[str, object]]) -> None:
    print("Correlation matrix (Pearson, numeric features):")
    matrix = correlation_matrix(rows, NUMERIC_COLUMNS)
    header = "\t".join([" ", *NUMERIC_COLUMNS])
    print(header)
    for col_x in NUMERIC_COLUMNS:
        row_values = [f"{matrix[col_x][col_y]:.3f}" for col_y in NUMERIC_COLUMNS]
        print("\t".join([col_x, *row_values]))


def main(argv: List[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_DATA_PATH
    if not path.exists():
        raise SystemExit(f"Data file not found: {path}")

    raw_records = load_housing_data(path)
    rows = normalise_records(raw_records)

    print(f"Dataset shape (rows, columns): {len(rows)} x {len(raw_records[0])}")
    print()

    print_numeric_summary(rows)
    print_boolean_summary(rows)
    print_price_by_category(rows)
    print_correlation_matrix(rows)
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main(sys.argv))
