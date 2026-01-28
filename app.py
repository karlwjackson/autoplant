import io
import json
from typing import List, Iterable
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobProperties, ContainerClient  # 🔁 added ContainerClient

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="IoT Telemetry Dashboard", page_icon="🌿", layout="wide")
st.title("🌿 IoT Telemetry Dashboard")

# -----------------------------
# Read config from secrets
# -----------------------------
ACCOUNT_NAME   = st.secrets.get("ACCOUNT_NAME", "")
CONTAINER_NAME = st.secrets.get("CONTAINER_NAME", "")
BLOB_PREFIX    = st.secrets.get("BLOB_PREFIX", "")  # optional

# 🔁 New: SAS secrets (either ACCOUNT_SAS_URL or CONTAINER_SAS_URL)
ACCOUNT_SAS_URL   = st.secrets.get("ACCOUNT_SAS_URL", "").strip()
CONTAINER_SAS_URL = st.secrets.get("CONTAINER_SAS_URL", "").strip()

# Basic validation for local fallback
if not CONTAINER_SAS_URL and not ACCOUNT_SAS_URL:
    if not ACCOUNT_NAME or not CONTAINER_NAME:
        st.error("Please set ACCOUNT_NAME and CONTAINER_NAME in Streamlit secrets.")
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
    # Show what we know; in SAS mode, we still display container name from secrets
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
    Client factory with three paths:
      1) CONTAINER_SAS_URL (preferred if provided): read-only SAS scoped to container
      2) ACCOUNT_SAS_URL: read-only SAS at account level, scoped to container in code
      3) DefaultAzureCredential: local dev / managed identity (RBAC)
    """
    # 1) Container SAS URL (exact container path)
    if CONTAINER_SAS_URL:
        return ContainerClient.from_container_url(CONTAINER_SAS_URL)

    # 2) Account SAS URL (Blob-only; Container+Object; Read+List)
    if ACCOUNT_SAS_URL:
        if not CONTAINER_NAME:
            st.error("CONTAINER_NAME must be set when using ACCOUNT_SAS_URL.")
            st.stop()
        container_url = f"{ACCOUNT_SAS_URL.rstrip('/')}/{CONTAINER_NAME}"
        return ContainerClient.from_container_url(container_url)

    # 3) Fallback to Entra/RBAC (local dev or Azure-hosted with Managed Identity)
    cred = DefaultAzureCredential(exclude_shared_token_cache_credential=False)
    bsc = BlobServiceClient(account_url=ACCOUNT_URL, credential=cred)
    return bsc.get_container_client(CONTAINER_NAME)

container_client = get_container_client()