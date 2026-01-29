import io
import json
from typing import List, Iterable
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from urllib.parse import urlsplit

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobProperties, ContainerClient  # ← added ContainerClient

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

# NEW: SAS secrets (use either one)
ACCOUNT_SAS_URL = st.secrets.get("ACCOUNT_SAS_URL", "").strip()     # e.g. https://<acct>.blob.core.windows.net/?sv=...&ss=b&srt=co&sp=rl&se=...
CONTAINER_SAS_URL = st.secrets.get("CONTAINER_SAS_URL", "").strip() # e.g. https://<acct>.blob.core.windows.net/<container>?<sas>

# Basic validation for local fallback
if not CONTAINER_SAS_URL and not ACCOUNT_SAS_URL:
    if not ACCOUNT_NAME or not CONTAINER_NAME:
        st.error("Please set ACCOUNT_NAME and CONTAINER_NAME in .streamlit/secrets.toml")
        st.stop()

ACCOUNT_URL = f"https://{ACCOUNT_NAME}.blob.core.windows.net"

# Dynamic caption based on auth mode
if CONTAINER_SAS_URL or ACCOUNT_SAS_URL:
    st.caption("Source: Azure Blob → **SAS (read-only)**")
else:
    st.caption("Source: Azure Blob → **Microsoft Entra ID (DefaultAzureCredential)**")

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Data Source")
    st.write(f"**Account:** `{ACCOUNT_NAME or 'via SAS'}`")
    st.write(f"**Container:** `{CONTAINER_NAME or 'via SAS'}`")
    blob_prefix = st.text_input("Blob prefix (optional)", value=BLOB_PREFIX, help="Leave blank to load all blobs in the container.")
    max_files = st.slider("Max files to load (safety)", 1, 5000, 1000, help="Stop after this many blobs to prevent huge loads.")
    st.divider()

    st.header("Visualization")
    lookback_hours = st.slider("Lookback (hours)", 1, 24*14, 48)
    show_roll = st.checkbox("Show 3-interval rolling avg", value=True)
    show_anomalies = st.checkbox("Flag anomalies (robust z-score |z|>3)", value=False)

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
    # 1) Container SAS URL (already scoped to container, e.g. https://acct.blob.core.windows.net/<container>?<sas>)
    if CONTAINER_SAS_URL:
        return ContainerClient.from_container_url(CONTAINER_SAS_URL)

    # 2) Account SAS URL (e.g. https://acct.blob.core.windows.net/?sv=...&ss=b&srt=co&sp=rl...)
    if ACCOUNT_SAS_URL:
        if not CONTAINER_NAME:
            st.error("CONTAINER_NAME must be set when using ACCOUNT_SAS_URL.")
            st.stop()

        parts = urlsplit(ACCOUNT_SAS_URL)
        if not parts.scheme or not parts.netloc or not parts.query:
            st.error("ACCOUNT_SAS_URL looks invalid. Use the full 'Blob service SAS URL' from the portal (includes ?sv=...).")
            st.stop()

        # Build https://<account>.blob.core.windows.net/<container>?<sas-query>
        base = f"{parts.scheme}://{parts.netloc}"
        container_url = f"{base}/{CONTAINER_NAME}?{parts.query}"

        # Optional: show a safe hint to confirm (no token leakage)
        # st.caption(f"Auth: SAS (account) → {base}/{CONTAINER_NAME}?<redacted>")

        return ContainerClient.from_container_url(container_url)

    # 3) Fallback to Entra/RBAC (local dev or Azure with Managed Identity)
    cred = DefaultAzureCredential(exclude_shared_token_cache_credential=False)
    bsc = BlobServiceClient(account_url=f"https://{ACCOUNT_NAME}.blob.core.windows.net", credential=cred)
    return bsc.get_container_client(CONTAINER_NAME)


container_client = get_container_client()

# -----------------------------
# Helpers for loading & parsing
# -----------------------------
def _parse_to_df(raw_bytes: bytes) -> pd.DataFrame:
    """
    Accepts:
      - NDJSON (newline-delimited JSON)
      - JSON array of objects
      - Single JSON object
    """
    # Try NDJSON (common for logs/telemetry)
    try:
        df = pd.read_json(io.BytesIO(raw_bytes), lines=True)
        if not df.empty:
            return df
    except Exception:
        pass

    # Fallback: array or single object
    txt = raw_bytes.decode("utf-8")
    obj = json.loads(txt)
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    return pd.DataFrame([obj])

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

    # Numeric coercion (keeps true NaNs for proper math)
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

        # Coerce defensively to numeric (handles any residual text)
        numeric_s = s[cols].apply(pd.to_numeric, errors="coerce")

        # Resample numeric-only means
        resampled_numeric = numeric_s.resample("15min").mean(numeric_only=True)

        # Interpolate short gaps; keep longer gaps as NaN
        resampled_numeric = resampled_numeric.interpolate(limit=2, limit_direction="both")

        # Restore deviceId
        resampled_numeric["deviceId"] = dev
        resampled_numeric = resampled_numeric.reset_index()

        frames.append(resampled_numeric)

    return pd.concat(frames, ignore_index=True)

@st.cache_data(ttl=300, show_spinner=True)
def list_blob_names(prefix: str, limit: int) -> list[str]:
    names: list[str] = []
    pager: Iterable[BlobProperties] = container_client.list_blobs(name_starts_with=prefix or None)
    for i, bp in enumerate(pager, start=1):
        names.append(bp.name)
        if i >= limit:
            break
    return names

@st.cache_data(ttl=300, show_spinner=True)
def load_and_merge_all(prefix: str, limit: int) -> pd.DataFrame:
    names = list_blob_names(prefix, limit)
    if not names:
        return pd.DataFrame()

    dfs = []
    for n in names:
        try:
            raw = container_client.get_blob_client(n).download_blob().readall()
            dfi = _parse_to_df(raw)
            if not dfi.empty:
                dfs.append(dfi)
        except Exception as e:
            # Keep going but surface a message in the UI
            st.info(f"Skipped blob due to error: {n} ({e})")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = _clean_df(df)
    return df

# -----------------------------
# Load & prepare data
# -----------------------------
with st.spinner("Loading blobs from Azure and merging..."):
    df = load_and_merge_all(blob_prefix, max_files)

if df.empty:
    st.warning("No data found. Check container, prefix filter, or permissions.")
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
        z = df_15.groupby("deviceId")[c].transform(_robust_zscore)
        df_15[f"{c}_anomaly"] = z.abs() > 3

# -----------------------------
# KPIs
# -----------------------------
kpi_cols = st.columns(5)
pretty = {
    "WaterTempC": "Water Temp (°C)",
    "AirTempC": "Air Temp (°C)",
    "HumidityPct": "Humidity (%)",
    "Lux": "Light (Lux)",
    "PPD_K": "PPD (K)"
}

def _kpi_metric(col, label, dfk):
    if dfk.empty:
        col.metric(pretty[label], "–", "–"); return
    bydev = dfk.sort_values("Timestamp").groupby("deviceId")[label]
    latest = bydev.last().mean()
    earliest = bydev.first().mean()
    delta = (latest - earliest) if pd.notna(latest) and pd.notna(earliest) else None
    col.metric(pretty[label], f"{latest:.2f}" if pd.notna(latest) else "–", f"{delta:+.2f}" if delta is not None else "–")

for i, key in enumerate(pretty.keys()):
    _kpi_metric(kpi_cols[i], key, df_15)

latest_ts = df_15["Timestamp"].max()
if pd.notna(latest_ts):
    st.caption(f"Latest timestamp in view: **{latest_ts.strftime('%Y-%m-%d %H:%M UTC')}**")

# -----------------------------
# Charts
# -----------------------------
def plot_timeseries(df_plot: pd.DataFrame, y: str, title: str):
    if df_plot.empty:
        st.info(f"No data to plot for {title}.")
        return

    # Build long-form data: Actual vs Rolling
    parts = []

    # Actual
    actual = df_plot[["Timestamp", "deviceId", y]].rename(columns={y: "value"})
    actual["series"] = "Actual"
    parts.append(actual)

    # Rolling (only if exists & checkbox is on)
    roll_col = f"{y}_roll"
    if show_roll and roll_col in df_plot.columns:
        rolling = df_plot[["Timestamp", "deviceId", roll_col]].rename(columns={roll_col: "value"})
        rolling["series"] = "Rolling"
        parts.append(rolling)

    df_long = pd.concat(parts, ignore_index=True).dropna(subset=["value"])

    # Explicit colors: Actual=blue, Rolling=red
    color_map = {
        "Actual": "#1f77b4",   # Plotly default blue
        "Rolling": "#d62728",  # Plotly default red
    }

    fig = px.line(
        df_long,
        x="Timestamp",
        y="value",
        color="series",          # color by series type (Actual vs Rolling)
        line_group="deviceId",   # group lines per device so pairs don't mix
        hover_data=["deviceId"],
        title=title,
        template="plotly_white",
        labels={"value": title, "deviceId": "Device", "series": "Series"},
        color_discrete_map=color_map,
    )

    # Style lines: Actual thicker; Rolling dashed
    fig.update_traces(
        selector=lambda tr: tr.name == "Actual",
        line=dict(width=3)
    )
    fig.update_traces(
        selector=lambda tr: tr.name == "Rolling",
        line=dict(dash="dash", width=2)
    )

    fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Optional anomaly message
    anom_col = f"{y}_anomaly"
    if show_anomalies and anom_col in df_plot.columns and df_plot[anom_col].any():
        st.warning(f"Anomalies flagged for **{title}**: {int(df_plot[anom_col].sum())} points (|robust z| > 3).")

c1, c2 = st.columns(2)
with c1:
    plot_timeseries(df_15, "WaterTempC", "Water Temperature (°C)")
    plot_timeseries(df_15, "HumidityPct", "Humidity (%)")
    plot_timeseries(df_15, "PPD_K", "Predicted Percentage Dissatisfied (K)")
with c2:
    plot_timeseries(df_15, "AirTempC", "Air Temperature (°C)")
    plot_timeseries(df_15, "Lux", "Illuminance (Lux)")

# Raw table
with st.expander("Raw data (current window)"):
    st.dataframe(df_15.sort_values(["deviceId", "Timestamp"]), use_container_width=True, hide_index=True)

st.success("Dashboard ready. Adjust the sidebar to change window, devices, anomalies, etc.")