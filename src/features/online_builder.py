

import pandas as pd
import numpy as np

RAW_REQUIRED = [
    "invoice_id", "client_id", "client_name",
    "invoice_date", "payment_due_date",
    "invoice_amount", "material_name", "payment_method"
]
RAW_OPTIONAL = [
    "payment_actual_date"  # optional; if present, we can compute target for evaluation
]

def _parse_dates(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def _safe_category(df, col, default="Unknown"):
    if col not in df.columns:
        df[col] = default
    df[col] = df[col].fillna(default).astype(str)
    return df

def _compute_targets_if_available(df):
    # optional target: only if payment_actual_date exists
    if "payment_actual_date" in df.columns:
        df["delay_days"] = (df["payment_actual_date"] - df["payment_due_date"]).dt.days
        df["delayed_flag"] = (df["delay_days"] > 0).astype(int)
        df["target"] = df["delayed_flag"]
    return df

def _historical_features_no_leak(df):
    """
    Build history using only PRIOR invoices of the same client.
    Assumes df is sorted by client_id, invoice_date ascending.
    """
    g = df.groupby("client_id", group_keys=False)

    # simple expanding stats on invoice_amount (shift to use only prior info)
    df["client_prev_txn_count"] = g.cumcount()
    df["client_prev_total_value"] = g["invoice_amount"].shift().groupby(df["client_id"]).cumsum().fillna(0.0)

    # if we have targets (from actual dates), use their history safely
    if "delayed_flag" in df.columns:
        df["client_prev_delay_rate"] = (
            g["delayed_flag"].apply(lambda s: s.shift().expanding().mean())
        )
        df["client_prev_avg_delay"] = (
            g["delay_days"].apply(lambda s: s.shift().expanding().mean())
        )
    else:
        df["client_prev_delay_rate"] = 0.0
        df["client_prev_avg_delay"] = 0.0

    # client-relative amount stats (z-score)
    df["client_amount_mean"] = g["invoice_amount"].apply(lambda s: s.shift().expanding().mean())
    df["client_amount_std"]  = g["invoice_amount"].apply(lambda s: s.shift().expanding().std())
    df["client_amount_z"]    = (df["invoice_amount"] - df["client_amount_mean"]) / (df["client_amount_std"] + 1e-6)

    # 90D exposure rolling (prior only)
    def _rolling_90(group):
        # set time index
        g2 = group.set_index("invoice_date").sort_index()
        # prior values only
        prior = g2["invoice_amount"].shift()
        roll = prior.rolling("90D").sum()
        out = roll.reindex(group["invoice_date"].values)
        return out.values

    df = df.sort_values(["client_id", "invoice_date"])
    df["client_rolling_90_value"] = g.apply(_rolling_90).reset_index(level=0, drop=True)
    df["client_rolling_90_value"] = df["client_rolling_90_value"].fillna(0.0)

    # fill history NaNs with 0 after computing
    for c in ["client_prev_delay_rate","client_prev_avg_delay","client_amount_mean","client_amount_std"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    return df

def _calendar_features(df):
    df["term_days"] = (df["payment_due_date"] - df["invoice_date"]).dt.days
    df["month"]   = df["invoice_date"].dt.month
    df["quarter"] = df["invoice_date"].dt.quarter
    df["year"]    = df["invoice_date"].dt.year

    # flags
    df["is_month_end"]   = df["invoice_date"].dt.is_month_end.astype(int)
    # exact quarter-end: compare to period end
    qend = df["invoice_date"].dt.to_period("Q").dt.end_time.dt.date
    df["is_quarter_end"] = (df["invoice_date"].dt.date == qend).astype(int)

    month_end = df["invoice_date"].dt.to_period("M").dt.end_time
    df["days_to_month_end"] = (month_end - df["invoice_date"]).dt.days

    df["due_dow"] = df["payment_due_date"].dt.weekday
    df["inv_dow"] = df["invoice_date"].dt.weekday

    # log amount
    df["log_amount"] = np.log1p(df["invoice_amount"].clip(lower=0))

    return df

def _one_hot(df):
    # normalize categories first
    df = _safe_category(df, "payment_method", "Unknown")
    df = _safe_category(df, "material_name", "Unknown")
    # one-hot with drop_first=True (to match training)
    df = pd.get_dummies(df, columns=["payment_method","material_name"], drop_first=True)
    return df

def align_to_feature_cols(df_feat, feature_cols):
    """
    Ensure the output has exactly the model's feature columns (order & presence),
    adding any missing columns as 0 and dropping extras.
    """
    for c in feature_cols:
        if c not in df_feat.columns:
            df_feat[c] = 0.0
    extra = [c for c in df_feat.columns if c not in feature_cols]
    if extra:
        df_feat = df_feat.drop(columns=extra)
    df_feat = df_feat[feature_cols]
    # hygiene
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df_feat

def build_features_from_raw(df_raw: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Public entry: takes raw invoices and returns a DataFrame with EXACT feature_cols order.
    Raw schema required: see RAW_REQUIRED.
    """
    # basic checks
    missing = [c for c in RAW_REQUIRED if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    # dates + types
    df = df_raw.copy()
    df = _parse_dates(df, ["invoice_date","payment_due_date","payment_actual_date"])
    for c in ["invoice_amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # sanitize categories
    df = _safe_category(df, "payment_method", "Unknown")
    df = _safe_category(df, "material_name", "Unknown")

    # compute optional targets if we have actual dates (useful for eval only)
    df = _compute_targets_if_available(df)

    # keep strictly increasing time per client
    df = df.sort_values(["client_id","invoice_date"]).reset_index(drop=True)

    # calendar + invoice features
    df = _calendar_features(df)

    # historical (prior only)
    df = _historical_features_no_leak(df)

    # one-hots
    df = _one_hot(df)

    # final alignment
    feat = align_to_feature_cols(df, feature_cols)
    # keep metadata (for display/export convenience)
    meta_cols = [c for c in ["invoice_id","client_id","client_name","invoice_date","invoice_amount","material_name","payment_method","target"] if c in df.columns]
    feat = pd.concat([df[meta_cols].reset_index(drop=True), feat.reset_index(drop=True)], axis=1)

    return feat
