
# features.py — Feature Engineering Module
# Tạo toàn bộ daily-level features cho bài toán Sales Forecasting.

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42  # re-exported constant

_CACHE: dict[str, pd.DataFrame] = {}

def _load_csv(data_dir: Path, name: str, **kwargs) -> pd.DataFrame:
    key = str(data_dir / name)
    if key not in _CACHE:
        _CACHE[key] = pd.read_csv(data_dir / name, **kwargs)
    return _CACHE[key].copy()


def _clear_cache():
    _CACHE.clear()

# ============================================================================
# 1. HOLIDAY FLAGS — Lễ Việt Nam
# ============================================================================

_TET_FIRST_DAY: dict[int, str] = {
    2012: "2012-01-23", 2013: "2013-02-10", 2014: "2014-01-31",
    2015: "2015-02-19", 2016: "2016-02-08", 2017: "2017-01-28",
    2018: "2018-02-16", 2019: "2019-02-05", 2020: "2020-01-25",
    2021: "2021-02-12", 2022: "2022-02-01", 2023: "2023-01-22",
    2024: "2024-02-10",
}


def _build_holiday_set(years: list[int]) -> set[pd.Timestamp]:
    holidays: set[pd.Timestamp] = set()

    for y in years:
        if y in _TET_FIRST_DAY:
            tet_center = pd.Timestamp(_TET_FIRST_DAY[y])
            for delta in range(-4, 8):      #4 days before to 7 days after Tết
                holidays.add(tet_center + pd.Timedelta(days=delta))

        for mmdd in ["01-01", "04-30", "05-01", "09-02"]:
            holidays.add(pd.Timestamp(f"{y}-{mmdd}"))

    return holidays

def _build_tet_series(dates: pd.Series) -> pd.Series:
    #Return number of days to the *next* Tết for each date (capped at 365).
    tet_ts = sorted(
        pd.Timestamp(v) for v in _TET_FIRST_DAY.values()
        if pd.Timestamp(v) >= dates.min()
    )
    out = np.full(len(dates), 365, dtype=np.float32)
    for i, d in enumerate(dates):
        for t in tet_ts:
            diff = (t - d).days
            if 0 <= diff < 365:
                out[i] = float(diff)
                break
    return pd.Series(out, index=dates.index, name="days_to_tet")

def add_holiday_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:

    d = df.copy()
    years   = sorted(d[date_col].dt.year.unique().tolist())
    hol_set = _build_holiday_set(years)

    d["is_holiday"]      = d[date_col].isin(hol_set).astype(np.int8)
    d["days_to_tet"]     = _build_tet_series(d[date_col]).values
    d["is_pre_tet_rush"] = ((d["days_to_tet"] > 0) & (d["days_to_tet"] <= 21)).astype(np.int8)
    d["is_tet_holiday"]  = ((d["days_to_tet"] <= 0) & (d["days_to_tet"] >= -6)).astype(np.int8)

    pre_set, post_set = set(), set()
    for h in hol_set:
        for delta in range(-3, 0):
            pre_set.add(h + pd.Timedelta(days=delta))
        for delta in range(1, 4):
            post_set.add(h + pd.Timedelta(days=delta))

    d["is_pre_holiday"]  = d[date_col].isin(pre_set).astype(np.int8)
    d["is_post_holiday"] = d[date_col].isin(post_set).astype(np.int8)
    d["is_holiday_halo"] = (
        d["is_holiday"] | d["is_pre_holiday"] | d["is_post_holiday"]
    ).astype(np.int8)

    return d


# ============================================================================
# 2. CALENDAR FEATURES
# ============================================================================

def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:

    d = df.copy()
    dt = d[date_col]

    # Time decomposition cơ bản
    d["day_of_week"]   = dt.dt.dayofweek.astype(np.int8)   # 0=Mon … 6=Sun
    d["day_of_month"]  = dt.dt.day.astype(np.int8)
    d["day_of_year"]   = dt.dt.dayofyear.astype(np.int16)
    d["week_of_year"]  = dt.dt.isocalendar().week.astype(np.int8)
    d["month"]         = dt.dt.month.astype(np.int8)
    d["quarter"]       = dt.dt.quarter.astype(np.int8)
    d["year"]          = dt.dt.year.astype(np.int16)

    # Binary flags
    d["is_weekend"]     = (dt.dt.dayofweek >= 5).astype(np.int8)
    d["is_month_start"] = dt.dt.is_month_start.astype(np.int8)
    d["is_month_end"]   = dt.dt.is_month_end.astype(np.int8)
    d["is_quarter_end"] = dt.dt.is_quarter_end.astype(np.int8)
    d["is_payday_window"] = ((dt.dt.day >= 25) | (dt.dt.day <= 5)).astype(np.int8)
    d["is_double_date"] = (dt.dt.month == dt.dt.day).astype(np.int8)

    # Fourier encoding chu kỳ năm (period=365.25)
    doy = d["day_of_year"].astype(float)
    for k in [1, 2, 3]:
        d[f"sin_year_{k}"] = np.sin(2 * np.pi * k * doy / 365.25).astype(np.float32)
        d[f"cos_year_{k}"] = np.cos(2 * np.pi * k * doy / 365.25).astype(np.float32)

    # Fourier encoding chu kỳ tuần (period=7)
    dow = d["day_of_week"].astype(float)
    for k in [1, 2]:
        d[f"sin_week_{k}"] = np.sin(2 * np.pi * k * dow / 7).astype(np.float32)
        d[f"cos_week_{k}"] = np.cos(2 * np.pi * k * dow / 7).astype(np.float32)

    # Trend proxy
    origin = pd.Timestamp("2012-07-04")
    d["trend_days"] = (dt - origin).dt.days.astype(np.int32)

    d["trend_days_sq"] = (d["trend_days"] ** 2).astype(np.float64)

    return d


# ============================================================================
# 3. LAG & ROLLING FEATURES
# ============================================================================

def add_lag_rolling_features(
    df: pd.DataFrame,
    target_cols: list[str] = ("Revenue", "COGS"),
    lags: list[int]        = (1, 7, 14, 28, 30, 91),
    roll_windows: list[int] = (7, 14, 30, 91),
) -> pd.DataFrame:

    d = df.sort_values("Date").copy()

    for col in target_cols:
        if col not in d.columns:
            continue

        series = d[col]
        log_s = np.log1p(series)

        # Lag features
        for lag in lags:
            d[f"{col}_lag{lag}"] = series.shift(lag).astype(np.float32)

        # Cùng ngày năm ngoái (mạnh nhất cho e-commerce thời trang)
        d[f"{col}_lag365"] = series.shift(365).astype(np.float32)

        for lag in [1, 7, 28, 91, 365]:
            d[f"log_{col}_lag{lag}"] = log_s.shift(lag).astype(np.float32)

        d[f"{col}_yoy_log_diff"] = (
                log_s.shift(1) - log_s.shift(366)
        ).astype(np.float32)

        # Shift 1 trước khi rolling
        s_shifted = series.shift(1)
        log_shifted = log_s.shift(1)

        # Rolling mean & std
        for w in roll_windows:
            d[f"{col}_roll_mean_{w}"]     = s_shifted.rolling(w, min_periods=1).mean().astype(np.float32)
            d[f"{col}_roll_std_{w}"]      = s_shifted.rolling(w, min_periods=1).std().fillna(0).astype(np.float32)
            d[f"log_{col}_roll_mean_{w}"] = log_shifted.rolling(w, min_periods=1).mean().astype(np.float32)

        # EWM (trọng số gần hơn nặng hơn)
        d[f"{col}_ewm7"]  = s_shifted.ewm(span=7,  adjust=False).mean().astype(np.float32)
        d[f"{col}_ewm30"] = s_shifted.ewm(span=30, adjust=False).mean().astype(np.float32)
        d[f"log_{col}_ewm30"] = log_shifted.ewm(span=30, adjust=False).mean().astype(np.float32)

        # YoY ratio: momentum dài hạn
        prev_year = series.shift(365)
        d[f"{col}_yoy_ratio"] = (
            (s_shifted / (prev_year.shift(1) + 1e-9)).astype(np.float32)
        )

    # Cross-feature: tỉ lệ COGS/Revenue (margin proxy)
    if "Revenue" in target_cols and "COGS" in target_cols:
        rev_s1  = d["Revenue"].shift(1)
        cogs_s1 = d["COGS"].shift(1)
        d["margin_ratio_lag1"] = (
            (cogs_s1 / (rev_s1 + 1e-9)).astype(np.float32)
        )

    return d


# ============================================================================
# 4. AUXILIARY TABLE FEATURES
# ============================================================================

def add_web_traffic_features(
    df: pd.DataFrame,
    data_dir: Path,
    is_test: bool = False,
) -> pd.DataFrame:

    wt = _load_csv(data_dir, "web_traffic.csv", parse_dates=["date"])
    wt = wt.rename(columns={"date": "Date"})

    # Tổng hợp tất cả traffic source thành 1 dòng/ngày
    agg = (
        wt.groupby("Date")
        .agg(
            wt_sessions         = ("sessions",                "sum"),
            wt_unique_visitors  = ("unique_visitors",         "sum"),
            wt_page_views       = ("page_views",              "sum"),
            wt_bounce_rate      = ("bounce_rate",             "mean"),
            wt_session_duration = ("avg_session_duration_sec","mean"),
        )
        .sort_index()
        .reset_index()
    )

    # Pivot per-source session count (organic_search, direct, paid_search, …)
    pivot = (
        wt.pivot_table(
            index="Date",
            columns="traffic_source",
            values="sessions",
            aggfunc="sum",
        )
        .reset_index()
    )
    pivot.columns = ["Date"] + [
        f"wt_src_{c.replace(' ', '_')}" for c in pivot.columns[1:]
    ]

    traffic = agg.merge(pivot, on="Date", how="outer").sort_values("Date").reset_index(drop=True)

    # Đồng nhất train/test feature space
    feat_cols_wt = [c for c in traffic.columns if c != "Date"]

    for col in feat_cols_wt:
        # Ghi đè bằng lag-1 value, giữ nguyên tên cột
        traffic[col] = traffic[col].shift(1).astype(np.float32)

    # Thêm rolling-7 với suffix _roll7 (cả train và test đều có)
    for col in feat_cols_wt:
        traffic[f"{col}_roll7"] = (
            traffic[col].rolling(7, min_periods=1).mean().astype(np.float32)
        )

    return df.merge(traffic, on="Date", how="left")


def add_review_features(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    rv = _load_csv(data_dir, "reviews.csv", parse_dates=["review_date"])
    rv = rv.rename(columns={"review_date": "Date"})

    daily = (
        rv.groupby("Date")
        .agg(
            rev_avg_rating=("rating", "mean"),
            rev_count=("rating", "count"),
            rev_low_count=("rating", lambda x: (x <= 2).sum()),
        )
        .reset_index()
        .sort_values("Date")
    )

    # Derived: fraction of negative reviews per day
    daily["rev_low_rate"] = (
            daily["rev_low_count"] / daily["rev_count"].clip(lower=1)
    ).astype(np.float32)

    # Lag-1 to avoid leakage
    for col in ["rev_avg_rating", "rev_count", "rev_low_rate"]:
        daily[f"{col}_lag1"] = daily[col].shift(1).astype(np.float32)

    # Rolling 7-day stats (already shifted by 1 via lag1 columns)
    daily["rev_avg_rating_7d"] = daily["rev_avg_rating_lag1"].rolling(7, min_periods=1).mean().astype(np.float32)
    daily["rev_count_roll7"] = daily["rev_count_lag1"].rolling(7, min_periods=1).sum().astype(np.float32)
    daily["rev_low_rating_rate7d"] = daily["rev_low_rate_lag1"].rolling(7, min_periods=1).mean().astype(np.float32)

    # Drop intermediate raw columns; keep only lagged / rolled
    keep = ["Date", "rev_avg_rating_lag1", "rev_count_lag1",
            "rev_avg_rating_7d", "rev_count_roll7", "rev_low_rating_rate7d"]
    daily = daily[keep]

    # Merge; fill days with no reviews using global mean rating (neutral signal)
    merged = df.merge(daily, on="Date", how="left")
    global_mean_rating = rv["rating"].mean()
    merged["rev_avg_rating_lag1"] = merged["rev_avg_rating_lag1"].fillna(global_mean_rating).astype(np.float32)
    merged["rev_avg_rating_7d"] = merged["rev_avg_rating_7d"].fillna(global_mean_rating).astype(np.float32)
    merged["rev_count_lag1"] = merged["rev_count_lag1"].fillna(0).astype(np.float32)
    merged["rev_count_roll7"] = merged["rev_count_roll7"].fillna(0).astype(np.float32)
    merged["rev_low_rating_rate7d"] = merged["rev_low_rating_rate7d"].fillna(0).astype(np.float32)

    return merged


def add_shipment_features(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    sh = _load_csv(data_dir, "shipments.csv", parse_dates=["ship_date", "delivery_date"])

    sh["delivery_days"] = (
            sh["delivery_date"] - sh["ship_date"]
    ).dt.days.clip(lower=0)

    daily = (
        sh.groupby("ship_date")
        .agg(
            ship_avg_delivery_days = ("delivery_days", "mean"),
            ship_std_delivery_days = ("delivery_days", "std"),
            ship_avg_fee           = ("shipping_fee",  "mean"),
            ship_count             = ("order_id",      "count"),
        )
        .reset_index()
        .rename(columns={"ship_date": "Date"})
        .sort_values("Date")
    )

    daily["ship_std_delivery_days"] = daily["ship_std_delivery_days"].fillna(0)

    # Lag-1 raw values
    for col in ["ship_avg_delivery_days", "ship_std_delivery_days",
                "ship_avg_fee", "ship_count"]:
        daily[f"{col}_lag1"] = daily[col].shift(1).astype(np.float32)

    # Rolling 7d on the lag-1 series
    daily["ship_avg_delivery_days_7d"] = (
        daily["ship_avg_delivery_days_lag1"].rolling(7, min_periods=1).mean().astype(np.float32)
    )
    daily["ship_delivery_std_7d"] = (
        daily["ship_std_delivery_days_lag1"].rolling(7, min_periods=1).mean().astype(np.float32)
    )

    keep = ["Date",
            "ship_avg_delivery_days_lag1", "ship_avg_delivery_days_7d",
            "ship_delivery_std_7d", "ship_avg_fee_lag1", "ship_count_lag1"]
    daily = daily[keep]

    merged = df.merge(daily, on="Date", how="left")

    # Fill NaN: use forward-fill for delivery time (stable), 0 for counts/fees
    for col in ["ship_avg_delivery_days_lag1", "ship_avg_delivery_days_7d",
                "ship_delivery_std_7d"]:
        merged[col] = merged[col].ffill().fillna(3.0).astype(np.float32)
    for col in ["ship_avg_fee_lag1", "ship_count_lag1"]:
        merged[col] = merged[col].fillna(0).astype(np.float32)

    return merged

def add_return_features(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    rt = _load_csv(data_dir, "returns.csv", parse_dates=["return_date"])

    daily = (
        rt.groupby("return_date")
        .agg(
            ret_count=("return_id", "count"),
            ret_refund=("refund_amount", "sum"),
            ret_qty=("return_quantity", "sum"),
        )
        .reset_index()
        .rename(columns={"return_date": "Date"})
        .sort_values("Date")
    )

    # Lag-1
    for col in ["ret_count", "ret_refund", "ret_qty"]:
        daily[f"{col}_lag1"] = daily[col].shift(1).astype(np.float32)

    # Rolling 30d on the lag-1 series
    daily["ret_count_roll30"] = daily["ret_count_lag1"].rolling(30, min_periods=1).sum().astype(np.float32)
    daily["ret_refund_roll30"] = daily["ret_refund_lag1"].rolling(30, min_periods=1).sum().astype(np.float32)

    # Rolling 7d for quicker reaction
    daily["ret_count_roll7"] = daily["ret_count_lag1"].rolling(7, min_periods=1).sum().astype(np.float32)

    keep = ["Date", "ret_count_lag1", "ret_refund_lag1",
            "ret_count_roll7", "ret_count_roll30", "ret_refund_roll30"]
    daily = daily[keep]

    merged = df.merge(daily, on="Date", how="left")
    for col in keep[1:]:
        merged[col] = merged[col].fillna(0).astype(np.float32)

    return merged

def add_geography_features(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    rv = _load_csv(data_dir, "reviews.csv", parse_dates=["review_date"])
    cust = _load_csv(data_dir, "customers.csv")
    geo = _load_csv(data_dir, "geography.csv")

    # reviews → customer zip (M:1, safe)
    rv_cust = rv[["customer_id", "review_date"]].merge(
        cust[["customer_id", "zip"]],
        on="customer_id",
        how="left",
    )

    # customer zip → region (M:1, safe)
    rv_cust = rv_cust.merge(
        geo[["zip", "region"]].drop_duplicates("zip"),
        on="zip",
        how="left",
    )

    rv_cust["region"] = rv_cust["region"].fillna("Unknown")

    # Daily regional review counts (pivot)
    daily_region = (
        rv_cust.groupby(["review_date", "region"])
        .size()
        .reset_index(name="region_count")
    )

    # Total daily reviews
    daily_total = (
        daily_region.groupby("review_date")["region_count"]
        .sum()
        .reset_index(name="total_count")
    )

    # Join to get share per region per day
    daily_region = daily_region.merge(daily_total, on="review_date", how="left")
    daily_region["share"] = (
            daily_region["region_count"] / daily_region["total_count"].clip(lower=1)
    )

    # Aggregate HHI per day: sum(share²) — ranges 0→1 (1 = monopoly)
    hhi = (
        daily_region.groupby("review_date")
        .apply(lambda g: (g["share"] ** 2).sum())
        .reset_index(name="geo_hhi")
    )

    # Top-region share per day
    top_share = (
        daily_region.groupby("review_date")["share"]
        .max()
        .reset_index(name="geo_top_region_share")
    )

    # Number of active regions per day
    n_regions = (
        daily_region.groupby("review_date")["region"]
        .nunique()
        .reset_index(name="geo_n_active_regions")
    )

    # Top-tier city share (HCMC / Hanoi dominant flag)
    top_cities = {"Ho Chi Minh City", "Hanoi", "HCM", "Hà Nội", "TP.HCM"}
    rv_cust2 = rv_cust.copy()
    rv_cust2["is_top_city"] = rv_cust2["region"].isin(top_cities).astype(int)
    top_city_share = (
        rv_cust2.groupby("review_date")
        .agg(top_city_count=("is_top_city", "sum"),
             total=("is_top_city", "count"))
        .reset_index()
    )
    top_city_share["geo_top_city_share"] = (
            top_city_share["top_city_count"] / top_city_share["total"].clip(lower=1)
    )
    top_city_share = top_city_share[["review_date", "geo_top_city_share"]]

    # Combine all geo signals into one daily frame
    geo_daily = (
        hhi
        .merge(top_share, on="review_date", how="outer")
        .merge(n_regions, on="review_date", how="outer")
        .merge(top_city_share, on="review_date", how="outer")
        .rename(columns={"review_date": "Date"})
        .sort_values("Date")
    )

    # Flag: top-tier city dominates (>60%)
    geo_daily["geo_is_top_city_dominant"] = (
            geo_daily["geo_top_city_share"] > 0.60
    ).astype(np.int8)

    # Lag-1 all geo signals
    for col in ["geo_hhi", "geo_top_region_share",
                "geo_n_active_regions", "geo_top_city_share",
                "geo_is_top_city_dominant"]:
        geo_daily[f"{col}_lag1"] = geo_daily[col].shift(1).astype(np.float32)

    # Rolling 7d on lagged values
    geo_daily["geo_hhi_7d"] = (
        geo_daily["geo_hhi_lag1"].rolling(7, min_periods=1).mean().astype(np.float32)
    )

    keep_cols = ["Date"] + [c for c in geo_daily.columns
                            if c.endswith("_lag1") or c == "geo_hhi_7d"]
    geo_daily = geo_daily[keep_cols]

    merged = df.merge(geo_daily, on="Date", how="left")

    # Fill NaN: HHI → 0.5 (moderate concentration), shares → 0, count → 0
    fill_map = {
        "geo_hhi_lag1": 0.5,
        "geo_hhi_7d": 0.5,
        "geo_top_region_share_lag1": 0.5,
        "geo_n_active_regions_lag1": 0.0,
        "geo_top_city_share_lag1": 0.5,
        "geo_is_top_city_dominant_lag1": 0.0,
    }
    for col, fill_val in fill_map.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(fill_val).astype(np.float32)

    return merged

def add_promotion_features(
    df: pd.DataFrame,
    data_dir: Path,
) -> pd.DataFrame:

    promos = _load_csv(data_dir, "promotions.csv", parse_dates=["start_date", "end_date"],)

    # Tính effective_rate
    promos = promos.copy()
    pct_mask = promos["promo_type"] == "percentage"

    promos["effective_rate"] = np.where(
        pct_mask,
        promos["discount_value"] / 100.0,
        promos["discount_value"] / promos["min_order_value"].clip(1),
    ).astype(np.float32)

    records = []
    for dt in df["Date"]:
        active = promos[
            (promos["start_date"] <= dt) & (promos["end_date"] >= dt)
        ]
        records.append({
            "Date":                   dt,
            "promo_n_active":         len(active),
            "promo_max_eff_rate": float(np.max(active["effective_rate"])) if len(active) else 0.0,
            "promo_sum_eff_rate": float(active["effective_rate"].sum()) if len(active) else 0.0,
            "promo_any_stackable":    int(active["stackable_flag"].any()) if len(active) else 0,
            "promo_min_order_val":    float(np.min(active["min_order_value"])) if len(active) else 0.0,
            "promo_channel_count":    int(active["promo_channel"].nunique()) if len(active) else 0,
            "promo_has_percentage":   int(
                (active["promo_type"] == "percentage").any()
            ) if len(active) else 0,
            "promo_has_fixed": int(
                (active["promo_type"] == "fixed").any()
            ) if len(active) else 0
        })

    promo_df = pd.DataFrame(records)
    return df.merge(promo_df, on="Date", how="left")


def add_inventory_features_safe(
    df: pd.DataFrame,
    data_dir: Path,
) -> pd.DataFrame:

    inv = _load_csv(data_dir, "inventory.csv", parse_dates=["snapshot_date"])

    # Aggregate tất cả products thành 1 dòng/tháng
    monthly = (
        inv.groupby("snapshot_date")
        .agg(
            inv_stock_on_hand   = ("stock_on_hand",     "sum"),
            inv_units_received  = ("units_received",    "sum"),
            inv_units_sold      = ("units_sold",        "sum"),
            inv_fill_rate       = ("fill_rate",         "mean"),
            inv_days_of_supply  = ("days_of_supply",    "mean"),
            inv_sell_through    = ("sell_through_rate", "mean"),
            inv_n_stockout      = ("stockout_flag",     "sum"),
            inv_n_overstock     = ("overstock_flag",    "sum"),
            inv_n_reorder       = ("reorder_flag",      "sum"),
            inv_stockout_days   = ("stockout_days",     "sum"),
        )
        .reset_index()
        .rename(columns={"snapshot_date": "_snap"})
        .sort_values("_snap")
    )

    monthly["join_date"] = monthly["_snap"] + pd.DateOffset(months=1)
    monthly["_year"] = monthly["join_date"].dt.year
    monthly["_month"] = monthly["join_date"].dt.month

    feat_cols = [c for c in monthly.columns
                 if c.startswith("inv_")]

    temp = df.copy()
    temp["_year"] = temp["Date"].dt.year
    temp["_month"] = temp["Date"].dt.month

    merged = temp.merge(
        monthly[["_year", "_month"] + feat_cols],
        left_on=["_year", "_month"],
        right_on=["_year", "_month"],
        how="left",
    ).drop(columns=["_year", "_month"])

    return merged


# ============================================================================
# 5. MASTER BUILDER — gọi 1 hàm duy nhất từ pipeline
# ============================================================================

def build_features(
    df: pd.DataFrame,
    data_dir: Path,
    is_test: bool = False,
    target_cols: list[str] = ("Revenue", "COGS"),
) -> pd.DataFrame:

    print(f"  [features] Bắt đầu build {'TEST' if is_test else 'TRAIN'} features …")

    d = df.sort_values("Date").copy()

    # Calendar
    d = add_calendar_features(d, date_col="Date")

    # Holiday flags
    d = add_holiday_features(d, date_col="Date")

    # Bước 3: Lag/rolling (chỉ khi có cột target)
    existing_targets = [c for c in target_cols if c in d.columns]
    if existing_targets:
        d = add_lag_rolling_features(d, target_cols=existing_targets)
    else:
        print("  [features] WARN: Không có target columns → skip lag/rolling.")

    # Web traffic
    d = add_web_traffic_features(d, data_dir=data_dir, is_test=is_test)

    # Promotions
    d = add_promotion_features(d, data_dir=data_dir)

    # Inventory
    d = add_inventory_features_safe(d, data_dir=data_dir)

    # Review
    d = add_review_features(d, data_dir=data_dir)

    # Shipment
    d = add_shipment_features(d, data_dir=data_dir)

    # Returns
    d = add_return_features(d, data_dir=data_dir)

    # Geography
    d = add_geography_features(d, data_dir=data_dir)

    print(f"  [features] Hoàn thành: {d.shape[0]} rows × {d.shape[1]} cols")
    return d


def get_feature_columns(df: pd.DataFrame, target_cols: tuple = ("Revenue", "COGS")) -> list[str]:
    """
    Trả về danh sách feature columns (loại trừ target, Date, và helper cols).
    """
    exclude = set(target_cols) | {"Date"}
    return [
        c for c in df.columns
        if c not in exclude and df[c].dtype in (np.float32, np.float64, np.int8, np.int16, np.int32, np.int64)
    ]