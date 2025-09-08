# src/app/aksum_risk_app.py
# Aksum.co.in — Payment Risk Scoring (Internal)
# Polished two-phase UX, lazy model load, spinners, and pro layout.
import sys
import os

# Get the project root directory (2 levels up from current file)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Now this import should work
from src.features.online_builder import build_features_from_raw
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
from io import StringIO
from pathlib import Path
from time import sleep







# ------------------- Page Setup & Theme -------------------
st.set_page_config(
    page_title="Aksum.co.in • Payment Risk Scoring (Internal)",
    page_icon="assets/favicon.png" if Path("assets/favicon.png").exists() else None,
    layout="wide"
)

# ------------------- Small CSS polish -------------------
st.markdown("""
<style>
/* subtle card look */
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
hr {border: 0; border-top: 1px solid #e9ecef;}
.ak-card {background: #fff; border: 1px solid #eef2f6; border-radius: 12px; padding: 16px;}
.ak-muted {color:#6c757d;}
.ak-chip {display:inline-block; padding:4px 10px; border-radius:999px; background:#f1f3f5; color:#495057; font-size:12px;}
thead tr th {font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ------------------- Brand Header -------------------
def brand_header():
    left, mid, right = st.columns([1, 4, 1], vertical_alignment="center")
    with left:
        logo_path = Path(r"C:\Resume Projects\credit-risk\assets\aksum_logo.png")
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)
        else:
            st.markdown("### **Aksum.co.in**")
    with mid:
        st.markdown(
            """
            <div>
              <h2 style="margin:0 0 6px 0;">Payment Default / Credit Risk — Internal Scoring</h2>
              <div class="ak-muted">Aksum.co.in • Confidential • For Internal Use Only</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with right:
        st.markdown("<div style='text-align:right' class='ak-muted'>v1.0 • ML Scoring</div>", unsafe_allow_html=True)

brand_header()
st.markdown("<hr>", unsafe_allow_html=True)

# ------------------- Paths (use relative; avoid hard-coded drive paths) -------------------
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / r"C:\Resume Projects\credit-risk\models\rf_model_v1.pkl"
META_PATH  = MODELS_DIR / r"C:\Resume Projects\credit-risk\models\rf_metadata_v1.json"

# ------------------- Helpers -------------------
def build_template_csv(headers):
    """Return a CSV (bytes) with just headers, so users know the exact format."""
    headers = list(dict.fromkeys(headers))  # dedupe while preserving order
    buf = StringIO()
    pd.DataFrame(columns=headers).to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

@st.cache_resource(show_spinner=False)
def load_meta_only():
    """Load metadata quickly (no heavy model yet)."""
    if not META_PATH.exists():
        raise FileNotFoundError("Metadata not found at models/rf_metadata_v1.json")
    with open(META_PATH, "r") as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def load_model_safely():
    """Lazy-load the model only when needed. mmap_mode speeds large arrays."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found at models/rf_model_v1.pkl")
    return joblib.load(MODEL_PATH, mmap_mode="r")

def to_rag(p, thr):
    if p >= max(thr, 0.70): return "RED"
    if p >= max(thr*0.75, 0.40): return "AMBER"
    return "GREEN"

# ------------------- Phase 1: Instructions + Template (fast: meta only) -------------------
try:
    meta = load_meta_only()
    feature_cols = list(meta.get("feature_cols", []))
    default_thr  = float(meta.get("threshold", 0.50))
    with st.container():
        st.success(f"Metadata loaded • Expected features: {len(feature_cols)}")
except Exception as e:
    st.error(f"Could not load metadata. Ensure models/rf_metadata_v1.json exists.\n\n{e}")
    st.stop()

st.subheader("Step 1 — Upload Instructions & Template")
with st.expander("📋 What file do I upload? (click to expand)", expanded=True):
    st.markdown(
        "Please upload a **processed features CSV** prepared in the same format used for training (`features_v2`)."
    )
    st.markdown("Required model features:")
    st.dataframe(pd.DataFrame({"required_feature_columns": feature_cols}), use_container_width=True, height=300)
    st.caption(
        "Optional helpful fields (if present, they will be displayed in results): "
        "`invoice_id`, `client_id`, `client_name`, `invoice_date`, `invoice_amount`, `material_name`, `payment_method`, `target`."
    )

    tmpl_headers = feature_cols + [
        "invoice_id","client_id","client_name","invoice_date",
        "invoice_amount","material_name","payment_method","target"
    ]
    st.download_button(
        "⬇️ Download CSV Template (Headers Only)",
        data=build_template_csv(tmpl_headers),
        file_name="aksum_risk_template.csv",
        mime="text/csv",
        help="Gives you the exact header structure the app expects."
    )

st.markdown("---")

# ------------------- Phase 1.5: Upload Gate -------------------
st.subheader("Step 2 — Upload File")
uploaded_file = st.file_uploader(
    "Upload features CSV (use the template or your exported features_v2 file)",
    type=["csv"],
    accept_multiple_files=False
)

if uploaded_file is None:
    # Beautiful idle card
    st.markdown(
        """
        <div class="ak-card">
          <div><span class="ak-chip">Status</span> Waiting for file upload…</div>
          <div class="ak-muted" style="margin-top:6px;">Tip: Use the template above or a slice of `features_v2.csv`.</div>
        </div>
        """, unsafe_allow_html=True
    )
    st.stop()

# Read CSV safely
try:
    df_raw = pd.read_csv(uploaded_file, low_memory=False)
except Exception as e:
    st.error(f"Could not read the CSV file.\n\nError: {e}")
    st.stop()

st.toast("Upload successful ✅", icon="✅")

# Schema check before any heavy work
missing = [c for c in feature_cols if c not in df_raw.columns]
if missing:
    st.error(f"Missing expected feature columns ({len(missing)}). First few: {missing[:12]}{' ...' if len(missing)>12 else ''}")
    st.stop()

# ------------------- Phase 2: Controls & Lazy Model Load -------------------
st.subheader("Step 3 — Scoring Controls")
col_thr, col_tip = st.columns([2, 3])
with col_thr:
    thr = st.slider("Risk Threshold (move right = stricter, fewer flags)", 0.0, 1.0, value=default_thr, step=0.01)
with col_tip:
    st.markdown(
        """
        **Policy guide:**  
        • Lower threshold → higher recall (catch more risky invoices), but lower precision (more false alarms).  
        • Higher threshold → fewer flags (more precise), but risk of missing some delayed payments.
        """
    )

# Load the model *now* (so the UI appeared instantly earlier)
with st.spinner("Loading model…"):
    try:
        model = load_model_safely()
        sleep(0.2)  # tiny pause so spinner is visible
    except Exception as e:
        st.error(f"Model load failed. Ensure models/rf_model_v1.pkl exists.\n\n{e}")
        st.stop()

# Prepare features & score
X = df_raw[feature_cols].copy()
X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

with st.spinner("Scoring records…"):
    try:
        probs = model.predict_proba(X)[:, 1]
        pred  = (probs >= thr).astype(int)
    except Exception as e:
        st.error(f"Model could not score. Check columns & dtypes.\n\n{e}")
        st.stop()

df_out = df_raw.copy()
df_out["risk_score"] = probs.round(4)
df_out["risk_flag"]  = pred
df_out["risk_band"]  = [to_rag(p, thr) for p in probs]

# Lightweight reasons from RF importances
importances = getattr(model, "feature_importances_", None)
if importances is not None and len(importances) == len(feature_cols):
    imp = pd.Series(importances, index=feature_cols)
    imp = (imp / (imp.sum() + 1e-12)).sort_values(ascending=False)
    def top_reasons(row, k=3):
        vals = row[feature_cols]
        score = (vals.abs() * imp).sort_values(ascending=False)
        return ", ".join(score.index[:k])
    with st.spinner("Generating reason codes…"):
        df_out["top_reasons"] = df_out.apply(top_reasons, axis=1)
else:
    df_out["top_reasons"] = "—"

# ------------------- Results -------------------
st.subheader("Step 4 — Results")

# KPI cards
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Invoices", len(df_out))
k2.metric("Flagged Risky", int(df_out["risk_flag"].sum()))
k3.metric("Avg Risk Score", f"{df_out['risk_score'].mean():.2f}")
if "target" in df_out.columns:
    from sklearn.metrics import precision_score, recall_score
    try:
        precision = precision_score(df_out["target"], df_out["risk_flag"])
        recall    = recall_score(df_out["target"], df_out["risk_flag"])
        k4.metric("Precision / Recall", f"{precision:.2f} / {recall:.2f}")
    except Exception:
        k4.metric("Precision / Recall", "—")
else:
    k4.metric("Precision / Recall", "—")

tab_table, tab_client, tab_charts, tab_export, tab_raw = st.tabs(
    ["📄 Scored Table", "🏢 Client Summary", "📈 Charts", "⬇️ Export", "🛠️ Raw → Features → Score"]
)

with tab_table:
    st.markdown("**Top 100 scored records**")
    cols_show = [c for c in ["invoice_id","client_id","client_name","invoice_date"] if c in df_out.columns]
    cols_show += ["risk_score","risk_flag","risk_band","top_reasons"]
    st.dataframe(df_out[cols_show].head(100), use_container_width=True)

with tab_client:
    st.markdown("**Risk by Client**")
    for col in ["client_id","client_name"]:
        if col not in df_out.columns:
            df_out[col] = ""
    client_view = df_out.groupby(["client_id","client_name"]).agg(
        invoices=("risk_flag","count"),
        risky=("risk_flag","sum"),
        avg_score=("risk_score","mean"),
        exposure=("invoice_amount","sum") if "invoice_amount" in df_out.columns else ("risk_flag","count")
    ).reset_index()
    client_view["risky_pct"] = (client_view["risky"] / client_view["invoices"] * 100).round(1)
    client_view = client_view.sort_values(["risky_pct","avg_score","invoices"], ascending=[False, False, False])
    st.dataframe(client_view.head(50), use_container_width=True)
    st.caption("Tip: prioritize clients with high risky% and exposure to tighten terms or request LC.")

with tab_charts:
    st.markdown("**Risk Distribution**")
    colA, colB = st.columns(2)
    with colA:
        st.bar_chart(df_out["risk_band"].value_counts())
    with colB:
        if "material_name" in df_out.columns:
            mat = df_out.groupby("material_name")["risk_flag"].mean().sort_values(ascending=False) * 100
            st.bar_chart(mat)
        else:
            st.write("_No material_name column found._")

    st.markdown("**Risk vs Amount (binned)**")
    if "invoice_amount" in df_out.columns:
        # build deciles
        df_out["amount_bin"] = pd.qcut(df_out["invoice_amount"], q=10, duplicates="drop")

        # compute risk % by bin
        amt = (
            df_out.groupby("amount_bin")["risk_flag"]
            .mean()
            .mul(100)
            .reset_index()
            .rename(columns={"risk_flag": "risk_pct"})
        )

        # sort bins by numeric left edge (so they appear in ascending order)
        amt["bin_left"] = amt["amount_bin"].apply(lambda x: x.left)
        amt = amt.sort_values("bin_left").drop(columns="bin_left")

        # convert Interval -> string labels for plotting
        amt["amount_bin"] = amt["amount_bin"].astype(str)

        # simplest & robust: bar chart (categorical x works well)
        st.bar_chart(amt.set_index("amount_bin")["risk_pct"])
    else:
        st.write("_No invoice_amount column found._")

with tab_export:
    st.markdown("**Download Scored File**")
    st.download_button(
        "⬇️ Download CSV",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="aksum_scored_invoices.csv",
        mime="text/csv"
    )
    st.caption("Reason codes are heuristic for fast context. For audit-grade explanations, we can plug in SHAP next phase.")
with tab_raw:
    st.markdown("### Raw Invoices → Feature Build → Score")
    st.caption("Upload **raw invoice exports** (not the processed features). The app will compute features to match the model schema and then score.")

    # 1) Raw schema help + template
    with st.expander("📋 Raw file format (click to expand)", expanded=False):
        st.write("**Required columns:**")
        st.code(", ".join([
            "invoice_id", "client_id", "client_name",
            "invoice_date", "payment_due_date",
            "invoice_amount", "material_name", "payment_method"
        ]), language="text")
        st.write("**Optional columns:** `payment_actual_date` (used only to compute target for evaluations)")

        # downloadable raw template
        raw_headers = [
            "invoice_id","client_id","client_name",
            "invoice_date","payment_due_date","invoice_amount",
            "material_name","payment_method","payment_actual_date"
        ]
        from io import StringIO
        def _raw_template():
            buf = StringIO()
            pd.DataFrame(columns=raw_headers).to_csv(buf, index=False)
            return buf.getvalue().encode("utf-8")

        st.download_button(
            "⬇️ Download RAW CSV Template",
            data=_raw_template(),
            file_name="aksum_raw_template.csv",
            mime="text/csv"
        )

    # 2) Raw uploader
    raw_file = st.file_uploader("Upload RAW invoices CSV", type=["csv"], key="raw_uploader")

    if raw_file is None:
        st.info("Upload a raw CSV to continue.")
    else:
        try:
            raw_df = pd.read_csv(raw_file, low_memory=False)
        except Exception as e:
            st.error(f"Could not read raw CSV: {e}")
            st.stop()

        st.write("**Raw preview (first 5 rows):**")
        st.dataframe(raw_df.head(), use_container_width=True)

        # 3) Build features with the utility (matches model feature_cols)
        with st.spinner("Building features (no leakage)…"):
            try:
                built = build_features_from_raw(raw_df, feature_cols)
            except Exception as e:
                st.error(f"Feature build failed: {e}")
                st.stop()

        st.success("Features built successfully ✅")
        st.write("**Features preview (first 5 rows):**")
        st.dataframe(built.head(), use_container_width=True)

        # 4) Split meta + X for scoring
        meta_cols = [c for c in ["invoice_id","client_id","client_name","invoice_date","invoice_amount","material_name","payment_method","target"] if c in built.columns]
        X_built = built.drop(columns=meta_cols, errors="ignore")

        # 5) Score with SAME model & threshold slider above
        with st.spinner("Scoring…"):
            try:
                probs_raw = model.predict_proba(X_built)[:, 1]
                pred_raw  = (probs_raw >= thr).astype(int)
            except Exception as e:
                st.error(f"Scoring failed: {e}")
                st.stop()

        out_raw = built[meta_cols].copy()
        out_raw["risk_score"] = probs_raw.round(4)
        out_raw["risk_flag"]  = pred_raw
        out_raw["risk_band"]  = [to_rag(p, thr) for p in probs_raw]

        # lightweight reasons (optional): we can reuse the importances heuristic if desired
        importances = getattr(model, "feature_importances_", None)
        if importances is not None and len(importances) == len(X_built.columns):
            imp_raw = pd.Series(importances, index=X_built.columns)
            imp_raw = (imp_raw / (imp_raw.sum() + 1e-12)).sort_values(ascending=False)
            def _reasons(row, k=3):
                vals = row[X_built.columns]
                score = (vals.abs() * imp_raw).sort_values(ascending=False)
                return ", ".join(score.index[:k])
            out_full = pd.concat([built.reset_index(drop=True), out_raw[["risk_score","risk_flag","risk_band"]]], axis=1)
            out_full["top_reasons"] = out_full.apply(_reasons, axis=1)
            scored_view = out_full
        else:
            scored_view = pd.concat([built.reset_index(drop=True), out_raw[["risk_score","risk_flag","risk_band"]]], axis=1)
            scored_view["top_reasons"] = "—"

        # 6) Show & download
        st.markdown("**Scored (first 100):**")
        cols_disp = [c for c in ["invoice_id","client_id","client_name","invoice_date","invoice_amount","material_name","payment_method","risk_score","risk_flag","risk_band","top_reasons"] if c in scored_view.columns]
        st.dataframe(scored_view[cols_disp].head(100), use_container_width=True)

        st.download_button(
            "⬇️ Download Scored RAW → Features CSV",
            data=scored_view.to_csv(index=False).encode("utf-8"),
            file_name="aksum_scored_from_raw.csv",
            mime="text/csv"
        )

# Footer watermark
st.markdown(
    "<div style='text-align:center;color:#adb5bd;margin-top:24px;'>© Aksum.co.in — Internal Use Only</div>",
    unsafe_allow_html=True
)
