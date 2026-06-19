import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.data import DataLoader, FeatureEngineer, StateBuilder
from src.utils.logging import get_logger

logger = get_logger(__name__)


def test_data_pipeline():
    coin_list = ["BTC-USD", "ETH-USD", "SOL-USD"]
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    data_dir = Path("./data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        coin_list=coin_list,
        start_date=start_date,
        end_date=end_date,
        data_dir=str(data_dir),
    )
    df = loader.load_data()
    assert isinstance(df, pd.DataFrame), "load_data must return a DataFrame"
    assert isinstance(df.index, pd.MultiIndex), "DataFrame must have MultiIndex (date, symbol)"
    assert df.index.names == ["date", "symbol"], f"Expected ['date', 'symbol'], got {df.index.names}"
    expected_cols = {"open", "high", "low", "close", "volume"}
    assert expected_cols.issubset(set(df.columns)), f"Missing OHLCV columns. Got {set(df.columns)}"
    assert df.isna().sum().sum() == 0, "DataFrame contains NaN values after _handle_missing"
    assert df.index.get_level_values("date").is_monotonic_increasing, "Dates not sorted"

    dates = df.index.get_level_values("date").unique()
    symbols = df.index.get_level_values("symbol").unique()
    assert len(symbols) == len(coin_list), f"Expected {len(coin_list)} symbols, got {len(symbols)}"
    for date in dates:
        day_symbols = df.xs(date, level="date").index
        assert len(day_symbols) == len(coin_list), (
            f"Date {date} has {len(day_symbols)} symbols, expected {len(coin_list)}"
        )

    print("✓ DataLoader: load_data works correctly")

    fe = FeatureEngineer()
    df_tech, df_mkt = fe.compute_features(df)

    assert isinstance(df_tech, pd.DataFrame), "df_tech must be a DataFrame"
    assert isinstance(df_mkt, pd.DataFrame), "df_mkt must be a DataFrame"
    assert df_tech.index.names == ["date", "symbol"], f"df_tech index: {df_tech.index.names}"
    assert df_mkt.index.name == "date", f"df_mkt index: {df_mkt.index.name}"

    for f in fe.tech_feature_names:
        assert f in df_tech.columns, f"Missing tech feature: {f}"
    for f in fe.market_feature_names:
        assert f in df_mkt.columns, f"Missing market feature: {f}"

    print(f"  Technical features: {fe.tech_feature_names}")
    print(f"  Market features: {fe.market_feature_names}")
    print(f"  df_tech shape: {df_tech.shape}, df_mkt shape: {df_mkt.shape}")
    print("✓ FeatureEngineer: all features computed")

    L = 30
    sb = StateBuilder(window_length=L)
    tech_symbols = sorted(df_tech.index.get_level_values("symbol").unique())
    A = len(tech_symbols)
    F1 = len(sb.tech_feature_names)
    F2 = len(sb.market_feature_names)

    all_dates = df_tech.index.get_level_values("date").unique().sort_values()
    test_date = all_dates[max(L, min(365, len(all_dates) - 2 * L))]
    S_tech, S_mkt = sb.build_state(test_date, df_tech, df_mkt)

    assert isinstance(S_tech, torch.Tensor), "S_tech must be a torch.Tensor"
    assert isinstance(S_mkt, torch.Tensor), "S_mkt must be a torch.Tensor"
    assert S_tech.shape == (L, A, F1), f"S_tech shape: {S_tech.shape}, expected ({L}, {A}, {F1})"
    assert S_mkt.shape == (L, F2), f"S_mkt shape: {S_mkt.shape}, expected ({L}, {F2})"
    assert S_tech.dtype == torch.float32, f"S_tech dtype: {S_tech.dtype}"
    assert S_mkt.dtype == torch.float32, f"S_mkt dtype: {S_mkt.dtype}"

    print(f"  S_tech shape: {S_tech.shape}, S_mkt shape: {S_mkt.shape}")
    print("✓ StateBuilder: tensors have correct shape and dtype")

    for ax in range(F1):
        feat_slice = S_tech[:, :, ax]
        if feat_slice.numel() > 0 and not torch.allclose(feat_slice, torch.zeros_like(feat_slice)):
            mean_val = feat_slice.mean().item()
            std_val = feat_slice.std().item()
            assert abs(mean_val) < 1e-5, (
                f"Tech feature {sb.tech_feature_names[ax]} mean={mean_val:.4f}, expected ~0"
            )
            assert abs(std_val - 1.0) < 0.15, (
                f"Tech feature {sb.tech_feature_names[ax]} std={std_val:.4f}, expected ~1"
            )

    for ax in range(F2):
        feat_slice = S_mkt[:, ax]
        if feat_slice.numel() > 0 and not torch.allclose(feat_slice, torch.zeros_like(feat_slice)):
            mean_val = feat_slice.mean().item()
            std_val = feat_slice.std().item()
            assert abs(mean_val) < 1e-5, (
                f"Market feature {sb.market_feature_names[ax]} mean={mean_val:.4f}, expected ~0"
            )
            assert abs(std_val - 1.0) < 0.15, (
                f"Market feature {sb.market_feature_names[ax]} std={std_val:.4f}, expected ~1"
            )

    print("✓ StateBuilder: z-score normalization produces mean~0, std~1")

    states = sb.build_states(df_tech, df_mkt)
    n_expected = len(df_tech.index.get_level_values("date").unique()) - L + 1
    assert len(states) == n_expected, (
        f"Expected {n_expected} states, got {len(states)}"
    )
    print(f"  Generated {len(states)} rolling states")
    print("✓ StateBuilder: build_states produces correct number of states")

    train, val, test = DataLoader.split_data(df)
    total_dates = len(df.index.get_level_values("date").unique())
    train_dates = len(train.index.get_level_values("date").unique())
    val_dates = len(val.index.get_level_values("date").unique())
    test_dates = len(test.index.get_level_values("date").unique())
    assert train_dates + val_dates + test_dates == total_dates, (
        f"Split date counts don't sum: {train_dates} + {val_dates} + {test_dates} != {total_dates}"
    )
    print(f"  Split: train={train_dates} dates, val={val_dates} dates, test={test_dates} dates")
    print("✓ DataLoader.split_data: chronological split works")

    cache_path = Path("./data/processed/features.parquet")
    df_tech_cached, df_mkt_cached = fe.compute_features(df, cache_path=cache_path)
    assert df_tech_cached.shape == df_tech.shape, "Cached features shape mismatch"
    assert cache_path.exists(), "Cache file not created"
    print("✓ Caching: features saved and loaded correctly")

    print("\n" + "=" * 60)
    print("All data pipeline tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_data_pipeline()
