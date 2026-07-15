from pathlib import Path

from vnindex_model.data import canonical_columns, load_price_data, validate_price_data


def test_column_aliases():
    assert canonical_columns(["Trading Date", "Adjusted_Close", "VOLUME"]) == {
        "Trading Date": "date",
        "Adjusted_Close": "close",
        "VOLUME": "volume",
    }


def test_split_thousands_csv(tmp_path: Path):
    path = tmp_path / "VNINDEX.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Volume,,,,,,,\n13/7/2026,1,829.50,1,829.50,1,781.45,1,800.54,728,451,840,\n",
        encoding="utf-8",
    )
    frame, metadata = load_price_data(path)
    assert frame.iloc[0]["close"] == 1800.54
    assert frame.iloc[0]["volume"] == 728451840
    assert metadata["rows_loaded"] == 1


def test_quality_constraints(synthetic_ohlcv):
    table, quality = validate_price_data(
        synthetic_ohlcv,
        {
            "rows_loaded": len(synthetic_ohlcv),
            "duplicate_rows_removed": 0,
            "source_file": "synthetic",
            "start_date": "2020-01-01",
            "end_date": "2022-01-01",
            "sha256": "x",
            "duplicate_date_records": 0,
            "malformed_rows": [],
            "columns": list(synthetic_ohlcv),
        },
    )
    assert quality["ohlc_constraint_violations"] == 0
    assert not table.empty
