"""De-identified monthly PEA invoice summary used for reconciliation.

The source PDFs and account identifiers are retained privately and are not part
of the public code package. ``INVOICES`` retains the tuple interface used by
the analysis scripts: ``(month, billed_kWh, billed_peak_kW, total_THB)``.
"""
from pathlib import Path

import pandas as pd


_PATH = Path(__file__).resolve().parent / "data" / "invoice_summary_deidentified.csv"
_REQUIRED = ["month", "billed_kwh", "billed_peak_kw", "invoice_total_thb"]


def load_invoices(path: Path = _PATH) -> list[tuple[str, float, float, float]]:
    """Load and validate the de-identified invoice series used in this study."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing de-identified invoice summary: {path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != _REQUIRED:
        raise ValueError(f"Invoice columns must be {_REQUIRED}; found {list(frame.columns)}")
    if frame.isna().any().any() or frame.duplicated("month").any():
        raise ValueError("Invoice summary contains missing values or duplicate months")
    months = pd.PeriodIndex(frame["month"], freq="M")
    expected = pd.period_range("2023-01", "2026-09", freq="M")
    if len(frame) != len(expected) or not months.equals(expected):
        raise ValueError("Invoice summary must contain monthly observations from 2023-01 through 2026-09")
    numeric = frame[_REQUIRED[1:]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or (numeric < 0).any().any():
        raise ValueError("Invoice quantities must be finite, non-negative numbers")
    return [
        (str(month), float(kwh), float(kw), float(thb))
        for month, kwh, kw, thb in zip(
            frame["month"], numeric["billed_kwh"], numeric["billed_peak_kw"], numeric["invoice_total_thb"]
        )
    ]


INVOICES = load_invoices()
