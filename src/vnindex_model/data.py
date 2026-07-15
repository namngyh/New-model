"""Data discovery, defensive parsing, and quality reporting."""

from __future__ import annotations

import csv
import itertools
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .persistence import sha256_file, write_json

ALIASES = {
    "date": {"date", "datetime", "time", "tradingdate"},
    "open": {"open", "openprice"},
    "high": {"high", "highprice"},
    "low": {"low", "lowprice"},
    "close": {"close", "adjustedclose", "adjclose", "closeprice"},
    "volume": {"volume", "vol", "tradingvolume"},
}
SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".parquet"}


def normalize_column(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def canonical_columns(columns: Iterable[object]) -> dict[object, str]:
    mapping: dict[object, str] = {}
    for column in columns:
        normalized = normalize_column(column)
        for canonical, aliases in ALIASES.items():
            if normalized in aliases and canonical not in mapping.values():
                mapping[column] = canonical
    return mapping


def discover_data_file(root: str | Path = ".") -> Path:
    """Find the most plausible local price file without assuming its name."""
    root = Path(root)
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and "reports" not in path.parts
        and "artifacts" not in path.parts
        and "processed" not in path.parts
    ]
    if not candidates:
        raise FileNotFoundError("Không tìm thấy CSV/XLSX/XLS/Parquet trong workspace")
    candidates.sort(key=lambda p: ("vnindex" not in p.name.lower(), "raw" not in p.parts, -p.stat().st_size))
    return candidates[0]


def _number(group: list[str]) -> float:
    parts = [token.strip().replace(" ", "") for token in group if token.strip()]
    if not parts:
        return np.nan
    if len(parts) == 1:
        return float(parts[0].replace(",", ""))
    first, *rest = parts
    rebuilt = first
    for token in rest:
        if "." in token:
            whole, decimal = token.split(".", 1)
            rebuilt += whole.zfill(3) + "." + decimal
        else:
            rebuilt += token.zfill(3)
    return float(rebuilt.replace(",", ""))


def _parse_split_ohlcv(tokens: list[str]) -> dict[str, object] | None:
    """Recover five unquoted comma-separated numbers using OHLC constraints."""
    date = tokens[0].strip()
    parts = [token.strip() for token in tokens[1:] if token.strip()]
    if len(parts) < 5:
        return None
    # The local vendor format writes 1,840.69 as two unquoted tokens. A short
    # integer prefix followed by a decimal token is therefore structural, not
    # something to infer from OHLC validity. This matters because a few source
    # rows genuinely violate High/Close by a small amount and must be reported,
    # not silently "repaired" by choosing a different grouping.
    position = 0
    price_groups: list[list[str]] = []
    for _ in range(4):
        if position >= len(parts):
            break
        current = parts[position]
        if (
            "." not in current
            and len(current.lstrip("+-")) <= 2
            and position + 1 < len(parts)
            and ("." in parts[position + 1] or len(parts[position + 1].lstrip("+-")) <= 3)
        ):
            price_groups.append(parts[position : position + 2])
            position += 2
        else:
            price_groups.append([current])
            position += 1
    if len(price_groups) == 4 and position < len(parts):
        try:
            open_, high, low, close = [_number(group) for group in price_groups]
            volume = _number(parts[position:])
            values = np.array([open_, high, low, close, volume], dtype=float)
            if np.isfinite(values).all() and min(values[:4]) > 0 and volume >= 0:
                return {
                    "date": date,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
        except (TypeError, ValueError):
            pass
    # Generic fallback for less regular files. It is only reached when the
    # deterministic vendor rule above cannot recover five finite fields.
    best: tuple[float, dict[str, object]] | None = None
    for lengths in itertools.product(range(1, 4), repeat=4):
        cuts = np.cumsum(lengths).tolist()
        if cuts[-1] >= len(parts):
            continue
        groups = [parts[: cuts[0]]]
        groups.extend(parts[cuts[i] : cuts[i + 1]] for i in range(3))
        groups.append(parts[cuts[-1] :])
        try:
            open_, high, low, close, volume = [_number(group) for group in groups]
        except (TypeError, ValueError):
            continue
        values = [open_, high, low, close, volume]
        if not np.isfinite(values).all() or min(values[:4]) <= 0 or volume < 0:
            continue
        score = 0.0
        score += 8.0 if high >= max(open_, close) - 0.02 else -10.0
        score += 8.0 if low <= min(open_, close) + 0.02 else -10.0
        score += 5.0 if high >= low else -10.0
        score += 4.0 if high / max(low, 1e-12) < 1.6 else -8.0
        score += 3.0 if all("." not in x for x in groups[-1]) else -5.0
        score -= 0.2 * (sum(len(group) for group in groups[:4]) - 4)
        record = {"date": date, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
        if best is None or score > best[0]:
            best = (score, record)
    return None if best is None else best[1]


def _load_csv(path: Path) -> tuple[pd.DataFrame, list[int]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError("Tệp dữ liệu rỗng")
    header_mapping = canonical_columns(rows[0])
    malformed: list[int] = []
    records: list[dict[str, object]] = []
    split_format = len(rows[0]) > 6 or any(len(row) > 7 for row in rows[1 : min(100, len(rows))])
    if split_format and {"date", "open", "high", "low", "close", "volume"}.issubset(header_mapping.values()):
        for line_number, row in enumerate(rows[1:], start=2):
            if not any(token.strip() for token in row):
                continue
            record = _parse_split_ohlcv(row)
            if record is None:
                malformed.append(line_number)
            else:
                records.append(record)
        return pd.DataFrame(records), malformed
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame.rename(columns=canonical_columns(frame.columns))
    return frame, malformed


def load_price_data(path: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame, malformed = _load_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path).rename(columns=canonical_columns(pd.read_excel(path, nrows=0).columns))
        malformed = []
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
        frame = frame.rename(columns=canonical_columns(frame.columns))
        malformed = []
    else:
        raise ValueError(f"Định dạng chưa hỗ trợ: {suffix}")
    if "date" not in frame or "close" not in frame:
        raise ValueError("Dữ liệu phải có date và close")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", dayfirst=True)
    for column in ["open", "high", "low", "close", "volume"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column].astype(str).str.replace(",", "", regex=False), errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    duplicate_rows = int(frame.duplicated().sum())
    duplicate_dates = int(frame["date"].duplicated(keep=False).sum())
    frame = frame.drop_duplicates()
    if frame["date"].duplicated().any():
        aggregations = {column: "last" for column in frame.columns if column != "date"}
        for column, operation in {"open": "first", "high": "max", "low": "min", "volume": "sum"}.items():
            if column in aggregations:
                aggregations[column] = operation
        frame = frame.groupby("date", as_index=False).agg(aggregations)
    frame = frame.sort_values("date").reset_index(drop=True)
    metadata = {
        "source_file": str(path),
        "rows_loaded": int(len(frame)),
        "start_date": frame["date"].min().strftime("%Y-%m-%d"),
        "end_date": frame["date"].max().strftime("%Y-%m-%d"),
        "columns": frame.columns.tolist(),
        "duplicate_rows_removed": duplicate_rows,
        "duplicate_date_records": duplicate_dates,
        "malformed_rows": malformed,
        "sha256": sha256_file(path),
    }
    return frame, metadata


def validate_price_data(frame: pd.DataFrame, metadata: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    missing = frame.isna().sum()
    nonpositive = {
        column: int((frame[column] <= 0).sum()) for column in ["open", "high", "low", "close"] if column in frame
    }
    ohlc_invalid = pd.Series(False, index=frame.index)
    if {"open", "high", "low", "close"}.issubset(frame.columns):
        ohlc_invalid = (frame["high"] < frame[["open", "close"]].max(axis=1) - 1e-8) | (
            frame["low"] > frame[["open", "close"]].min(axis=1) + 1e-8
        )
    gaps = frame["date"].diff().dt.days
    gap_rows = frame.loc[gaps > 10, ["date"]].copy()
    gap_rows["calendar_gap_days"] = gaps[gaps > 10].astype(int)
    quality = {
        **metadata,
        "missing_counts": {key: int(value) for key, value in missing.items()},
        "nonpositive_prices": nonpositive,
        "negative_volume": int((frame.get("volume", pd.Series(dtype=float)) < 0).sum()),
        "ohlc_constraint_violations": int(ohlc_invalid.sum()),
        "calendar_gaps_over_10_days": int(len(gap_rows)),
        "close_interpolation_applied": False,
    }
    rows = [
        {"check": "rows_loaded", "value": len(frame), "status": "info"},
        {"check": "duplicate_rows_removed", "value": metadata["duplicate_rows_removed"], "status": "cleaned"},
        {
            "check": "ohlc_constraint_violations",
            "value": int(ohlc_invalid.sum()),
            "status": "pass" if not ohlc_invalid.any() else "warning",
        },
        {"check": "calendar_gaps_over_10_days", "value": len(gap_rows), "status": "review"},
        {"check": "close_interpolation_applied", "value": False, "status": "pass"},
    ]
    return pd.DataFrame(rows), quality


def validate_and_save(data_path: str | Path, output_root: str | Path = ".") -> tuple[pd.DataFrame, dict[str, object]]:
    frame, metadata = load_price_data(data_path)
    table, quality = validate_price_data(frame, metadata)
    root = Path(output_root)
    table.to_csv(root / "reports/tables/data_quality.csv", index=False)
    header = "| " + " | ".join(table.columns) + " |\n| " + " | ".join(["---"] * len(table.columns)) + " |\n"
    body = "\n".join("| " + " | ".join(map(str, row)) + " |" for row in table.itertuples(index=False, name=None))
    (root / "reports/tables/data_quality.md").write_text(header + body + "\n", encoding="utf-8")
    if {"open", "high", "low", "close"}.issubset(frame.columns):
        invalid = (frame["high"] < frame[["open", "close"]].max(axis=1) - 1e-8) | (
            frame["low"] > frame[["open", "close"]].min(axis=1) + 1e-8
        )
        frame.loc[invalid].to_csv(root / "reports/diagnostics/ohlc_violations.csv", index=False)
    gaps = frame["date"].diff().dt.days
    gap_table = frame.loc[gaps > 10, ["date"]].copy()
    gap_table["calendar_gap_days"] = gaps[gaps > 10].astype(int)
    gap_table.to_csv(root / "reports/diagnostics/calendar_gaps.csv", index=False)
    write_json(root / "reports/diagnostics/data_quality.json", quality)
    frame.to_csv(root / "data/processed/vnindex_clean.csv", index=False)
    return frame, quality
