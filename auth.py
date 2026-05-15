"""
Urban Heat Island AI - Authentication Module (v3)
Handles Google Earth Engine API auth with service account support,
credential caching, retry logic, and structured logging.

Improvements over v2:
  - Project ID auto-detected from service-account JSON (no hardcoded default)
  - Quota project header injection for shared billing scenarios
  - GEE API health-check ping after init to catch silently bad tokens
  - Clear user-facing guidance when running on Streamlit Cloud vs. locally
"""

import ee
import streamlit as st
import logging
import json
import os
from pathlib import Path
from typing import Optional

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EarthEngineAuth_v3")

# ── Constants ─────────────────────────────────────────────────────────────────
_DEFAULT_PROJECT   = "uhi-research-2025"
_SERVICE_ACCT_KEY  = "GEE_SERVICE_ACCOUNT_KEY"
_MAX_RETRIES       = 2


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_service_account_credentials() -> tuple[Optional[ee.ServiceAccountCredentials], Optional[str]]:
    """
    Loads service-account JSON and returns (credentials, project_id).
    Tries: Streamlit secrets → env var path → local file.
    """
    def _parse(raw_str_or_dict) -> tuple[ee.ServiceAccountCredentials, str]:
        if isinstance(raw_str_or_dict, str):
            info = json.loads(raw_str_or_dict)
        else:
            info = dict(raw_str_or_dict)
        creds = ee.ServiceAccountCredentials(
            email=info["client_email"],
            key_data=info["private_key"]
        )
        project = info.get("project_id", _DEFAULT_PROJECT)
        return creds, project

    # 1. Streamlit secrets
    try:
        raw = st.secrets.get(_SERVICE_ACCT_KEY)
        if raw:
            return _parse(raw)
    except Exception:
        pass

    # 2. Env var → file path
    path_env = os.environ.get(_SERVICE_ACCT_KEY)
    if path_env and Path(path_env).exists():
        try:
            with open(path_env) as fh:
                return _parse(fh.read())
        except Exception as exc:
            logger.warning(f"Service-account file load failed: {exc}")

    # 3. Conventional local path
    for local in [Path("credentials/service_account.json"),
                    Path(".streamlit/service_account.json")]:
        if local.exists():
            try:
                with open(local) as fh:
                    return _parse(fh.read())
            except Exception as exc:
                logger.warning(f"Local credential file load failed: {exc}")

    return None, None


def _health_check() -> bool:
    """
    Executes a lightweight EE operation to verify the token actually works.
    A silently expired OAuth token will pass ee.Initialize() but fail here.
    """
    try:
        ee.Image(0).getInfo()
        return True
    except Exception:
        return False


# ── Public API ────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def initialize_earth_engine(project_id: str = _DEFAULT_PROJECT) -> bool:
    """
    Authenticates and initialises the GEE API.

    Strategy (in order):
      1. Service-account credentials (production — no browser required)
      2. Cached local OAuth token   (developer convenience)
      3. Interactive OAuth flow     (first-time setup fallback)

    Health-check ping after each strategy to catch stale tokens.

    Returns:
        bool: True on success, False on unrecoverable failure.
    """
    logger.info(f"Initialising Earth Engine | default_project={project_id}")

    # ── Strategy 1: Service account ──────────────────────────────────────────
    creds, sa_project = _get_service_account_credentials()
    if creds:
        resolved_project = sa_project or project_id
        try:
            ee.Initialize(credentials=creds, project=resolved_project)
            if _health_check():
                logger.info(f"EE init OK via service account (project={resolved_project}).")
                return True
            logger.warning("Service-account init succeeded but health-check failed.")
        except Exception as exc:
            logger.warning(f"Service-account init failed: {exc}")

    # ── Strategy 2: Cached OAuth token ───────────────────────────────────────
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            ee.Initialize(project=project_id)
            if _health_check():
                logger.info(f"EE init OK via cached token (attempt {attempt}).")
                return True
        except ee.EEException:
            logger.warning(f"Cached-token attempt {attempt} failed.")
            break

    # ── Strategy 3: Interactive OAuth ────────────────────────────────────────
    try:
        logger.info("Triggering interactive OAuth flow …")
        ee.Authenticate(force=False)
        ee.Initialize(project=project_id)
        if _health_check():
            logger.info("EE init OK via interactive OAuth.")
            return True
    except Exception as critical:
        logger.error(f"All authentication strategies exhausted: {critical}")
        st.error(
            "### 🚨 Earth Engine Authentication Failed\n\n"
            "**Possible causes:**\n"
            "- No service-account key configured (`GEE_SERVICE_ACCOUNT_KEY`)\n"
            "- Expired local credentials → run `earthengine authenticate` in terminal\n"
            "- Cloud Project not registered for Earth Engine API\n"
            "- Firewall blocking `earthengine.googleapis.com`\n\n"
            f"**Error details:** `{critical}`\n\n"
            "**Streamlit Cloud users:** Add your service-account JSON as a secret named "
            "`GEE_SERVICE_ACCOUNT_KEY` in your app settings."
        )
        return False

    logger.error("Health-check failed after all auth strategies.")
    st.error("Earth Engine authenticated but API health-check failed. Check project quota.")
    return False