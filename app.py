import io
import json  # kept in case you still want to parse some JSON metadata later
from typing import List, Iterable
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from urllib.parse import urlsplit

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobProperties, ContainerClient  # ← ContainerClient in use

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="IoT Telemetry Dashboard", page_icon="🌿", layout="wide")
st.title("🌿 IoT Telemetry Dashboard")

# -----------------------------
# Read config from secrets
# -----------------------------
ACCOUNT_NAME = st.secrets.get("ACCOUNT_NAME", "")
CONTAINER_NAME = st.secrets.get("CONTAINER_NAME", "")
BLOB_PREFIX = st.secrets.get("BLOB_PREFIX", "")  # optional

# SAS secrets (optional)
ACCOUNT_SAS_URL = st.secrets.get("ACCOUNT_SAS_URL", "").strip()     # e.g. https://<acct>.blob.core.windows.net/?sv=...
CONTAINER_SAS_URL = st.secrets.get("CONTAINER_SAS_URL", "").strip() # e.g. https://<acct>.blob.core.windows.net/<container>?<sas>

# Basic validation for local fallback
if not CONTAINER_SAS_URL and not ACCOUNT_SAS_URL:
    if not ACCOUNT_NAME or not CONTAINER_NAME:
        st.error("Please set ACCOUNT_NAME and CONTAINER_NAME in .streamlit/secrets.toml")
        st.stop()

ACCOUNT_URL = f"https://{ACCOUNT_NAME}.blob.core.windows.net"

# Dynamic caption based on auth mode
if CONTAINER_SAS_URL or ACCOUNT_SAS_URL:
    st.caption("Source: Azure Blob → **SAS (read-only)** → Parquet files")
else:
    st.caption("Source: Azure Blob → **Microsoft Entra ID (DefaultAzureCredential)** → Parquet files")

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Data Source")
    st.write(f"**Account:** `{ACCOUNT_NAME or 'via SAS'}`")
    st.write(f"**Container:** `{CONTAINER_NAME or 'via SAS'}`")
    blob_prefix = st.text_input(
        "Blob prefix (optional)",
        value=BLOB_PREFIX,
        help="Parent folder (e.g., 'telemetry/'). Scan is recursive, so daily subfolders are fine."
    )
    max_files = st.slider("Max files to load (safety)", 1, 5000, 1000, help="Stops after this many Parquet files.")
    st.divider()

    st.header("Visualization")
    lookback_hours = st.slider("Lookback (hours)", 1, 24*14, 4)
    show_roll = st.checkbox("Show 3-interval rolling avg", value=True)
    show_anomalies = st.checkbox("Flag anomalies (robust z-score |z|>3)", value=False)

    st.divider()
    st.header("Targets")
    water_temp_target = st.number_input(
        "Water Temp Target (°C)",
        min_value=0.0, max_value=50.0, value=18.0, step=0.1,
        help="KPI delta will show Latest − Target and color red when above target."
    )

# -----------------------------
# Azure auth & clients
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_container_client() -> ContainerClient:
    """
    Three modes:
      1) CONTAINER_SAS_URL present  -> use it directly (read-only)
      2) ACCOUNT_SAS_URL present    -> scope it to container (read-only)
      3) DefaultAzureCredential     -> local dev / managed identity
    """
    # 1) Container SAS URL
    if CONTAINER_SAS_URL:
        return ContainerClient.from_container_url(CONTAINER_SAS_URL)

    # 2) Account SAS URL
    if ACCOUNT_SAS_URL:
        if not CONTAINER_NAME:
            st.error("CONTAINER_NAME must be set when using ACCOUNT_SAS_URL.")
            st.stop()

        parts = urlsplit(ACCOUNT_SAS_URL)
        if not parts.scheme or not parts.netloc or not parts.query:
            st.error("ACCOUNT_SAS_URL looks invalid. Use the full 'Blob service SAS URL' from the portal (includes ?sv=...).")
            st.stop()

        base = f"{parts.scheme}://{parts.netloc}"
        container_url = f"{base}/{CONTAINER_NAME}?{parts.query}"
        return ContainerClient.from_container_url(container_url)

    # 3) Entra/RBAC
    cred = DefaultAzureCredential(exclude_shared_token_cache_credential=False)
    bsc = BlobServiceClient(account_url=f"https://{ACCOUNT_NAME}.blob.core.windows.net", credential=cred)
    return bsc.get_container_client(CONTAINER_NAME)

container_client = get_container_client()

# -----------------------------
# Helpers for Parquet loading
# -----------------------------
def _ensure_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        st.error("The Parquet reader requires `pyarrow`. Add `pyarrow` to your requirements.")
        return False

def _read_parquet_to_df(raw_bytes: bytes, columns: List[str] | None = None) -> pd.DataFrame:
    buf = io.BytesIO(raw_bytes)
    try:
        return pd.read_parquet(buf, engine="pyarrow", columns=columns)
    except Exception as e:
        # Soft fallback to default engine if available
        try:
            return pd.read_parquet(buf, columns=columns)
        except Exception as e2:
            raise RuntimeError(f"Failed to read parquet (pyarrow & default): {e} | {e2}") from e2

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    expected = ["deviceId", "Timestamp", "AirTempC", "HumidityPct", "WaterTempC", "Lux", "PPD_K"]
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    # Normalize null-like strings globally to NaN
    NULL_LIKE = ["null", "None", "none", "NULL", "N/A", "NA", ""]
    df = df.replace(NULL_LIKE, np.nan)

    # Timestamps
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    # Numeric coercion
    numeric_cols = ["AirTempC", "HumidityPct", "WaterTempC", "Lux", "PPD_K"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    # deviceId fallback
    if "deviceId" not in df.columns or df["deviceId"].isna().all():
        df["deviceId"] = "unknown"

    return df[["deviceId", "Timestamp", "AirTempC", "HumidityPct", "WaterTempC", "Lux", "PPD_K"]]

def _robust_zscore(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return pd.Series(index=series.index, dtype=float)
    median = s.median()
    mad = (s - median).abs().median()
    if mad == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    return 0.6745 * (series - median) / mad

def _last_n_hours(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    if df.empty:
        return df
    end = df["Timestamp"].max()
    return df[(df["Timestamp"] >= end - pd.Timedelta(hours=hours)) & (df["Timestamp"] <= end)].copy()

def _resample_15m(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Safe resample: aggregate numeric columns only, keep NaNs for gaps."""
    if df.empty:
        return df

    frames = []
    for dev, g in df.groupby("deviceId"):
        s = g.set_index("Timestamp").sort_index()
        numeric_s = s[cols].apply(pd.to_numeric, errors="coerce")
        resampled_numeric = numeric_s.resample("15min").mean(numeric_only=True)
        resampled_numeric = resampled_numeric.interpolate(limit=2, limit_direction="both")
        resampled_numeric["deviceId"] = dev
        frames.append(resampled_numeric.reset_index())

    return pd.concat(frames, ignore_index=True)

# -----------------------------
# List & load Parquet blobs
# -----------------------------
@st.cache_data(ttl=300, show_spinner=True)
def list_parquet_blob_names(prefix: str, limit: int, lookback_hours: int) -> List[str]:
    """
    List .parquet blobs under prefix (recursive). Prefer blobs with last_modified
    within the lookback window (+2h buffer). If none found, fallback to newest N.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours + 2)
    recent: List[str] = []
    pager: Iterable[BlobProperties] = container_client.list_blobs(name_starts_with=prefix or None)

    for bp in pager:
        name = bp.name.lower()
        if not name.endswith(".parquet"):
            continue
        if bp.last_modified and bp.last_modified >= cutoff:
            recent.append(bp.name)
            if len(recent) >= limit:
                break

    if recent:
        return recent

    # Fallback: pick newest N parquet blobs
    all_parquet = []
    pager2: Iterable[BlobProperties] = container_client.list_blobs(name_starts_with=prefix or None)
    for bp in pager2:
        n = bp.name.lower()
        if n.endswith(".parquet"):
            all_parquet.append((bp.name, bp.last_modified or datetime(1970, 1, 1, tzinfo=timezone.utc)))
    all_parquet.sort(key=lambda t: t[1], reverse=True)
    return [n for n, _ in all_parquet[:limit]]

@st.cache_data(ttl=300, show_spinner=True)
def load_and_merge_parquet(prefix: str, limit: int, lookback_hours: int) -> pd.DataFrame:
    if not _ensure_pyarrow():
        return pd.DataFrame()

    names = list_parquet_blob_names(prefix, limit, lookback_hours)
    if not names:
        return pd.DataFrame()

    required_columns = ["deviceId", "Timestamp", "AirTempC", "HumidityPct", "WaterTempC", "Lux", "PPD_K"]

    dfs: List[pd.DataFrame] = []
    for n in names:
        try:
            raw = container_client.get_blob_client(n).download_blob().readall()
            dfi = _read_parquet_to_df(raw, columns=required_columns)
            if not dfi.empty:
                dfs.append(dfi)
        except Exception as e:
            st.info(f"Skipped blob due to error: {n} ({e})")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = _clean_df(df)
    return df

# -----------------------------
# Load & prepare data
# -----------------------------
with st.spinner("Loading Parquet blobs from Azure and merging..."):
    df = load_and_merge_parquet(blob_prefix, max_files, lookback_hours)

if df.empty:
    st.warning("No data found. Check container, prefix filter, file freshness, or permissions.")
    st.stop()

# Windowing & cadence
df_window = _last_n_hours(df, lookback_hours)
value_cols = ["AirTempC", "HumidityPct", "WaterTempC", "Lux", "PPD_K"]
df_15 = _resample_15m(df_window, value_cols)

# Device selection
devices = sorted(df_15["deviceId"].dropna().unique().tolist())
selected_devices = st.multiselect("Select device(s)", devices, default=devices)
df_15 = df_15[df_15["deviceId"].isin(selected_devices)]

# Rolling avg
if show_roll:
    for c in value_cols:
        df_15[f"{c}_roll"] = df_15.groupby("deviceId")[c].transform(lambda s: s.rolling(3, min_periods=1).mean())

# Anomalies
if show_anomalies:
    for c in value_cols:
