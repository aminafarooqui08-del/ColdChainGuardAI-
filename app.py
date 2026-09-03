# ============================================================
# ❄️ COLDCHAINGUARD AI
# Predictive Cold-Chain Intelligence Platform
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# MODEL + DATA FILE PATHS
# ============================================================
# These paths are relative to this app.py file, so the files only
# need to be kept in the same ColdChainGuard folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "coldchainguard_random_forest.pkl")
CSV_FILE = os.path.join(BASE_DIR, "coldchainguard_dashboard_data.csv")


# ============================================================
# LOAD MODEL + DATA
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None, f"Model file not found: {MODEL_FILE}"
    try:
        return joblib.load(MODEL_FILE), None
    except Exception as e:
        return None, f"Could not load model: {e}"


@st.cache_data
def load_live_model_data():
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame()
    try:
        data = pd.read_csv(CSV_FILE)
        data.columns = data.columns.astype(str).str.strip()
        return data
    except Exception:
        return pd.DataFrame()


model, model_error = load_model()
live_model_df = load_live_model_data()


# ============================================================
# MODEL HELPERS — FROM app.py LOGIC
# ============================================================

def get_expected_features(model):
    if model is None:
        return []

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    for step in getattr(model, "named_steps", {}).values():
        if hasattr(step, "feature_names_in_"):
            return list(step.feature_names_in_)

        transformers = getattr(step, "transformers_", None)
        if transformers is not None:
            cols = []
            for _, transformer, selected in transformers:
                if selected == "drop" or isinstance(selected, str):
                    continue
                try:
                    cols.extend(list(selected))
                except TypeError:
                    pass
            if cols:
                return list(dict.fromkeys(cols))

    return [
        "Current_Temperature_C",
        "Humidity_Pct",
        "Door_Open_Count",
        "Shock_Events",
        "Battery_Level_Pct",
        "Distance_KM",
        "Duration_Hours",
        "Product_Criticality",
        "Estimated_Value_INR",
        "Chain_Handoff_Count",
        "Destination_Type",
        "Packaging_Condition",
    ]


def get_categorical_info(model):
    categorical = {}
    if model is None:
        return categorical

    candidates = list(getattr(model, "named_steps", {}).values()) + [model]

    for obj in candidates:
        transformers = getattr(obj, "transformers_", None)
        if transformers is None:
            continue

        for _, transformer, columns in transformers:
            if transformer == "drop" or isinstance(columns, str):
                continue

            encoder = transformer
            if hasattr(transformer, "named_steps"):
                for step in transformer.named_steps.values():
                    if hasattr(step, "categories_"):
                        encoder = step
                        break

            if hasattr(encoder, "categories_"):
                try:
                    for column, categories in zip(columns, encoder.categories_):
                        categorical[column] = [str(x) for x in categories.tolist()]
                except Exception:
                    pass

    return categorical


def infer_categories(column):
    model_categories = get_categorical_info(model)

    if column in model_categories and model_categories[column]:
        return model_categories[column]

    if column in live_model_df.columns:
        values = [str(x) for x in live_model_df[column].dropna().unique().tolist()]
        values = list(dict.fromkeys(values))
        if 0 < len(values) <= 50:
            return values

    defaults = {
        "Product_Criticality": ["Low", "Medium", "High", "Critical"],
        "Destination_Type": ["Urban", "Rural", "Remote"],
        "Packaging_Condition": ["Good", "Fair", "Poor"],
        "Weather_Condition": ["Normal", "Hot", "Rainy", "Cold"],
        "Transport_Mode": ["Road", "Air", "Rail"],
    }
    return defaults.get(column, [])


def default_numeric(column):
    defaults = {
        "Current_Temperature_C": 4.0,
        "Humidity_Pct": 60.0,
        "Door_Open_Count": 2,
        "Shock_Events": 0,
        "Battery_Level_Pct": 90.0,
        "Distance_KM": 300.0,
        "Duration_Hours": 8.0,
        "Estimated_Value_INR": 50000.0,
        "Chain_Handoff_Count": 2,
    }

    if column in defaults:
        return defaults[column]

    if column in live_model_df.columns:
        values = pd.to_numeric(live_model_df[column], errors="coerce").dropna()
        if len(values):
            return float(values.median())

    return 0.0


def risk_label_from_prediction(prediction, confidence):
    text = str(prediction).lower()

    risk_words = ["risk", "critical", "danger", "unsafe", "high", "failure", "alert", "bad"]
    safe_words = ["safe", "normal", "low", "no risk", "healthy"]

    if any(word in text for word in safe_words):
        return "SAFE", "status-normal"

    if any(word in text for word in risk_words):
        return "HIGH RISK", "status-critical"

    # Standard binary classification used by this project:
    # 0 = SAFE, 1 = HIGH RISK.
    if isinstance(prediction, (int, float, np.integer, np.floating)):
        if float(prediction) >= 1:
            return "HIGH RISK", "status-critical"
        return "SAFE", "status-normal"

    if confidence >= 80:
        return "HIGH CONFIDENCE", "status-warning"

    return "MODEL RESULT", "status-warning"

def is_shipment_feature(column):
    """Return True for raw/one-hot shipment ID model features."""
    name = str(column).strip().lower().replace("_", " ")
    return name.startswith("shipment id") or name.startswith("shipmentid")


def shipment_feature_matches(column, shipment_id):
    """Check whether a one-hot shipment feature belongs to a shipment."""
    feature = str(column).strip().lower()
    sid = str(shipment_id).strip().lower()
    compact_feature = "".join(ch for ch in feature if ch.isalnum())
    compact_sid = "".join(ch for ch in sid if ch.isalnum())
    return compact_sid in compact_feature


def get_ui_features(model):
    """Model features shown to the user, excluding shipment-ID dummy features."""
    expected = get_expected_features(model)
    return [x for x in expected if not is_shipment_feature(x)]


def value_from_row(row, column, fallback=None):
    """Safely get a shipment value from the selected CSV row."""
    if column in row.index:
        value = row[column]
        if pd.notna(value):
            return value
    return fallback


def build_live_input(selected_row, selected_shipment):
    """
    Build ONE model row from ONE selected shipment.

    Shipment ID is selected once at the top of the UI. It is never shown
    as dozens of + / - number inputs. All other model conditions remain
    editable and start with the selected shipment's real CSV values.
    """
    expected = get_expected_features(model)

    if not expected:
        st.error("❌ Unable to determine model features.")
        return pd.DataFrame()

    preferred_order = [
        "Current_Temperature_C",
        "Humidity_Pct",
        "Door_Open_Count",
        "Shock_Events",
        "Battery_Level_Pct",
        "Distance_KM",
        "Duration_Hours",
        "Product_Criticality",
        "Estimated_Value_INR",
        "Chain_Handoff_Count",
        "Destination_Type",
        "Packaging_Condition",
    ]

    # Show only editable conditions from the start of the model feature list
    # through the Data Logger fields. Everything after Data Logger (destination,
    # date, storage facility, route metadata, etc.) stays hidden from the UI.
    non_shipment_expected = [
        x for x in expected
        if not is_shipment_feature(x)
        and str(x).strip().lower().replace("_", " ") != "shipment id"
    ]

    logger_positions = [
        i for i, x in enumerate(non_shipment_expected)
        if "data logger" in str(x).lower()
    ]

    if logger_positions:
        cutoff = max(logger_positions)
        ui_expected = non_shipment_expected[:cutoff + 1]
    else:
        # Fallback if the trained model uses different Data Logger names.
        ui_expected = [x for x in preferred_order if x in expected]

    # Keep the familiar order for the main physical conditions, then append
    # Data Logger fields in the model's original order.
    ordered = [x for x in preferred_order if x in ui_expected]
    ordered += [x for x in ui_expected if x not in ordered]

    values = {}

    st.subheader("📡 Current Shipment Conditions")
    st.caption(
        "Values are loaded from the selected shipment. You can change any "
        "condition before running the AI prediction."
    )

    left, right = st.columns(2)

    for index, column in enumerate(ordered):
        label = column.replace("_", " ")
        categories = infer_categories(column)
        raw_value = value_from_row(
            selected_row,
            column,
            default_numeric(column) if not categories else None
        )

        # --------------------------------------------------------
        # CATEGORICAL CONDITIONS
        # --------------------------------------------------------
        if categories:
            categories = [str(x) for x in categories]
            current = str(raw_value) if raw_value is not None else categories[0]

            if current not in categories:
                categories = [current] + categories

            with (left if index % 2 == 0 else right):
                values[column] = st.selectbox(
                    label,
                    categories,
                    index=categories.index(current),
                    key=f"live_{selected_shipment}_{column}"
                )
            continue

        # --------------------------------------------------------
        # NUMERIC CONDITIONS
        # --------------------------------------------------------
        value = raw_value
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = float(default_numeric(column))

        integer_like = column in {
            "Door_Open_Count",
            "Shock_Events",
            "Chain_Handoff_Count"
        }

        if column in live_model_df.columns:
            try:
                integer_like = integer_like or pd.api.types.is_integer_dtype(
                    live_model_df[column]
                )
            except Exception:
                pass

        with (left if index % 2 == 0 else right):
            if integer_like:
                values[column] = st.number_input(
                    label,
                    min_value=0,
                    value=max(0, int(round(value))),
                    step=1,
                    key=f"live_{selected_shipment}_{column}"
                )

            elif "Temperature" in column:
                values[column] = st.number_input(
                    "🌡️ " + label + " (°C)",
                    min_value=-50.0,
                    max_value=100.0,
                    value=float(np.clip(value, -50, 100)),
                    step=0.1,
                    key=f"live_{selected_shipment}_{column}"
                )

            elif "Humidity" in column or "Battery" in column:
                values[column] = st.number_input(
                    "💧 " + label + " (%)" if "Humidity" in column
                    else "🔋 " + label + " (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(np.clip(value, 0, 100)),
                    step=1.0,
                    key=f"live_{selected_shipment}_{column}"
                )

            elif "Distance" in column:
                values[column] = st.number_input(
                    "📍 " + label + " (KM)",
                    min_value=0.0,
                    max_value=100000.0,
                    value=max(0.0, value),
                    step=10.0,
                    key=f"live_{selected_shipment}_{column}"
                )

            elif "Duration" in column:
                values[column] = st.number_input(
                    "⏱️ " + label + " (Hours)",
                    min_value=0.0,
                    max_value=10000.0,
                    value=max(0.0, value),
                    step=0.5,
                    key=f"live_{selected_shipment}_{column}"
                )

            elif "Value" in column or "INR" in column:
                values[column] = st.number_input(
                    "💰 " + label + " (₹)",
                    min_value=0.0,
                    max_value=1000000000.0,
                    value=max(0.0, value),
                    step=1000.0,
                    key=f"live_{selected_shipment}_{column}"
                )

            else:
                values[column] = st.number_input(
                    label,
                    value=float(value),
                    step=1.0,
                    key=f"live_{selected_shipment}_{column}"
                )

    # ------------------------------------------------------------
    # BUILD EXACT MODEL INPUT
    # ------------------------------------------------------------
    model_values = {}

    for column in expected:
        # Raw Shipment_ID feature
        if str(column).strip().lower().replace("_", " ") == "shipment id":
            model_values[column] = selected_shipment

        # One-hot shipment feature, e.g. "Shipment ID CC100003"
        elif is_shipment_feature(column):
            model_values[column] = int(
                shipment_feature_matches(column, selected_shipment)
            )

        # Editable condition
        elif column in values:
            model_values[column] = values[column]

        # Extra model feature not displayed in our preferred list
        else:
            raw = value_from_row(selected_row, column, None)
            if raw is not None:
                model_values[column] = raw
            else:
                categories = infer_categories(column)
                model_values[column] = (
                    categories[0] if categories else default_numeric(column)
                )

    return pd.DataFrame([model_values], columns=expected)


def predict_live(live_data):
    """Run the trained model and calculate confidence vs actual risk probability."""
    prediction = model.predict(live_data)[0]

    confidence = 0.0
    risk_probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(live_data)[0]
        confidence = float(np.max(probabilities) * 100)

        classes = getattr(model, "classes_", None)

        if classes is not None:
            classes_text = [str(c).strip().lower() for c in classes]

            # First look for a class explicitly named as risk.
            risk_idx = next(
                (i for i, name in enumerate(classes_text)
                 if any(word in name for word in [
                     "risk", "critical", "danger", "unsafe",
                     "high", "alert", "bad"
                 ])),
                None
            )

            # For the common binary 0/1 classifier, 1 = risk.
            if risk_idx is None and len(classes) == 2:
                try:
                    risk_idx = list(classes).index(1)
                except ValueError:
                    try:
                        risk_idx = list(classes).index("1")
                    except ValueError:
                        risk_idx = 1

            if risk_idx is not None and risk_idx < len(probabilities):
                risk_probability = float(probabilities[risk_idx] * 100)

    # If the model does not expose probabilities, infer a simple binary
    # risk signal from the prediction itself rather than calling confidence risk.
    if risk_probability is None:
        prediction_text = str(prediction).strip().lower()
        risk_probability = (
            100.0
            if prediction_text in [
                "1", "risk", "risky", "critical", "danger",
                "unsafe", "high", "alert", "bad"
            ]
            else 0.0
        )

    label, css_class = risk_label_from_prediction(prediction, confidence)

    return prediction, confidence, risk_probability, label, css_class



# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ColdChainGuard AI",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# USER / ADMIN ACCESS CONTROL
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_role" not in st.session_state:
    st.session_state.user_role = None

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "user_email" not in st.session_state:
    st.session_state.user_email = ""


# ============================================================
# LOGIN SCREEN
# ============================================================

if not st.session_state.logged_in:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:35px 10px 20px 10px;
        ">
            <div style="
                font-size:52px;
                font-weight:900;
            ">
                ❄️ ColdChainGuard AI
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            text-align:center;
            color:#9db4c4;
            font-size:17px;
            margin-top:3px;
        ">
            Predictive Cold-Chain Intelligence Platform
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    left, center, right = st.columns([1, 2, 1])

    with center:

        st.subheader("🔐 Secure Access")

        role = st.radio(
            "Select your access type",
            [
                "👤 User",
                "🛡️ Administrator"
            ],
            horizontal=True
        )

        name = st.text_input(
            "👤 Full Name",
            placeholder="Enter your name"
        )

        email = st.text_input(
            "📧 Email Address",
            placeholder="Enter your email"
        )

        password = st.text_input(
            "🔑 Password",
            type="password",
            placeholder="Enter password"
        )

        st.write("")

        login = st.button(
            "🚀 ENTER COLDCHAINGUARD",
            use_container_width=True
        )

        if login:

            if not name.strip():
                st.error("Please enter your name.")

            elif not email.strip():
                st.error("Please enter your email.")

            elif "@" not in email:
                st.error("Please enter a valid email address.")

            elif not password:
                st.error("Please enter your password.")

            else:

                # ------------------------------------------------
                # ADMIN LOGIN
                # ------------------------------------------------

                if role == "🛡️ Administrator":

                    if (
                        email.lower().strip()
                        == "admin@coldchainguard.com"
                        and password == "admin123"
                    ):

                        st.session_state.logged_in = True
                        st.session_state.user_role = "ADMIN"
                        st.session_state.user_name = name
                        st.session_state.user_email = email

                        st.rerun()

                    else:

                        st.error(
                            "❌ Invalid administrator credentials."
                        )

                        st.info(
                            "Demo Admin Login: "
                            "admin@coldchainguard.com / admin123"
                        )

                # ------------------------------------------------
                # USER LOGIN
                # ------------------------------------------------

                else:

                    st.session_state.logged_in = True
                    st.session_state.user_role = "USER"
                    st.session_state.user_name = name
                    st.session_state.user_email = email

                    st.rerun()

    st.write("")
    st.write("")

    st.markdown(
        """
        <div style="
            text-align:center;
            color:#718899;
            font-size:12px;
        ">
            🔒 Secure Role-Based Access
            &nbsp; • &nbsp;
            AI-Powered Cold-Chain Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# ROLE INFORMATION
# ============================================================

USER_ROLE = st.session_state.user_role == "USER"
ADMIN_ROLE = st.session_state.user_role == "ADMIN"


# ============================================================
# LOGOUT FUNCTION
# ============================================================

def logout():

    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.user_name = ""
    st.session_state.user_email = ""

    st.rerun()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(20,150,220,.12),
                transparent 25%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(0,220,160,.08),
                transparent 25%
            ),
            #06101c;
        color: #eef6fb;
    }

    html, body, [class*="css"] {
        font-family: "Segoe UI", sans-serif;
    }

    h1, h2, h3, h4 {
        color: #f5fbff !important;
    }

    p {
        color: #c8d7e2;
    }

    section[data-testid="stSidebar"] {
        background: #081522 !important;
        border-right: 1px solid rgba(255,255,255,.08);
    }

    section[data-testid="stSidebar"] * {
        color: #dce8ef;
    }

    .stButton > button {
        width: 100%;
        min-height: 44px;
        border-radius: 10px;
        background: linear-gradient(
            135deg,
            #12334c,
            #0d263a
        );
        border: 1px solid rgba(60,190,255,.30);
        color: #f4fbff !important;
        font-weight: 750;
    }

    .stButton > button:hover {
        border-color: #27baff;
    }

    .card {
        background: linear-gradient(
            145deg,
            #102a42,
            #091b2b
        );
        border: 1px solid rgba(255,255,255,.09);
        border-radius: 16px;
        padding: 20px;
    }

    .card-title {
        color: #f3f9fc;
        font-size: 14px;
        font-weight: 800;
    }

    .card-value {
        color: #ffffff;
        font-size: 29px;
        font-weight: 850;
        margin-top: 7px;
    }

    .card-sub {
        color: #91a7b7;
        font-size: 11px;
        margin-top: 5px;
    }

    .reason {
        background: #0d2031;
        border: 1px solid rgba(255,255,255,.07);
        border-radius: 10px;
        padding: 13px;
        margin-bottom: 8px;
        color: #d7e4eb;
    }

    .status-normal {
        background: rgba(0,230,160,.08);
        border: 1px solid rgba(0,230,160,.25);
        color: #39edb3;
        padding: 13px;
        border-radius: 10px;
        font-weight: 800;
    }

    .status-warning {
        background: rgba(255,179,71,.09);
        border: 1px solid rgba(255,179,71,.28);
        color: #ffc36c;
        padding: 13px;
        border-radius: 10px;
        font-weight: 800;
    }

    .status-critical {
        background: rgba(255,72,100,.10);
        border: 1px solid rgba(255,72,100,.30);
        color: #ff7185;
        padding: 13px;
        border-radius: 10px;
        font-weight: 800;
    }

    .big-title {
        font-size: 38px;
        font-weight: 850;
        color: #f5fbff;
    }

    .subtitle {
        color: #91a7b7;
        font-size: 14px;
    }

    .demo-header {
        background: linear-gradient(
            135deg,
            #0f344e,
            #071928
        );
        border: 1px solid rgba(60,190,255,.20);
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 20px;
    }

    .footer {
        text-align: center;
        color: #718899;
        font-size: 11px;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD CSV
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "coldchainguard_dashboard_data.csv"
    )


try:

    df = load_data()

except FileNotFoundError:

    st.error("❌ CSV file not found!")

    st.info(
        "Make sure coldchainguard_dashboard_data.csv "
        "is in the SAME folder as app.py."
    )

    st.stop()

except Exception as e:

    st.error("❌ Error while loading CSV file.")

    st.exception(e)

    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

for col in df.columns:

    converted = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    if converted.notna().sum() >= len(df) * 0.7:

        df[col] = converted


numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()


for col in numeric_cols:

    if df[col].notna().any():

        df[col] = df[col].fillna(
            df[col].median()
        )

    else:

        df[col] = df[col].fillna(0)


# ============================================================
# HELPER - FIND COLUMN
# ============================================================

def find_column(possible_names):

    for name in possible_names:

        if name in df.columns:

            return name

    return None


# ============================================================
# FIND IMPORTANT COLUMNS
# ============================================================

shipment_col = find_column([
    "Shipment_ID",
    "ShipmentID",
    "Shipment"
])

risk_col = find_column([
    "Risk_Score",
    "RiskScore",
    "AI_Risk_Score"
])

risk_probability_col = find_column([
    "Risk_Probability",
    "Risk_Probability_Pct",
    "ML_AT_RISK_Probability"
])

temperature_col = find_column([
    "Current_Temperature_C",
    "Temperature_C",
    "Current_Temperature"
])

deviation_col = find_column([
    "Temperature_Deviation_C",
    "Product_Temperature_Deviation_C"
])

door_col = find_column([
    "Door_Open_Count",
    "Door_Openings"
])

battery_col = find_column([
    "Battery_Level_Pct",
    "Battery"
])

shock_col = find_column([
    "Shock_Events",
    "Shock_Count"
])

value_risk_col = find_column([
    "Value_At_Risk_INR",
    "Value_At_Risk"
])

value_col = find_column([
    "Estimated_Value_INR",
    "Estimated_Value"
])

product_col = find_column([
    "Product_Type",
    "Product",
    "Product_Category"
])

destination_col = find_column([
    "Destination_Name",
    "Destination"
])

transport_col = find_column([
    "Transport_Mode",
    "Transport"
])

route_col = find_column([
    "Route_Risk",
    "Route_Risk_Level",
    "Route Risk",
    "route_risk"
])


# ============================================================
# CREATE SHIPMENT COLUMN IF MISSING
# ============================================================

if shipment_col is None:

    df["Shipment_ID"] = [
        f"SHIP-{i + 1:04d}"
        for i in range(len(df))
    ]

    shipment_col = "Shipment_ID"


# ============================================================
# CREATE RISK SCORE IF MISSING
# ============================================================

if risk_col is None:

    if risk_probability_col is not None:

        probability = pd.to_numeric(
            df[risk_probability_col],
            errors="coerce"
        ).fillna(0)

        # Handle both 0-1 and 0-100 probability formats
        if probability.max() <= 1:

            df["AI_Risk_Score"] = (
                probability.clip(0, 1) * 100
            )

        else:

            df["AI_Risk_Score"] = (
                probability.clip(0, 100)
            )

    elif deviation_col is not None:

        deviation = abs(
            pd.to_numeric(
                df[deviation_col],
                errors="coerce"
            ).fillna(0)
        )

        df["AI_Risk_Score"] = (
            deviation * 20
        ).clip(0, 100)

    else:

        df["AI_Risk_Score"] = 20

    risk_col = "AI_Risk_Score"


df[risk_col] = pd.to_numeric(
    df[risk_col],
    errors="coerce"
).fillna(0).clip(0, 100)


# ============================================================
# DASHBOARD CALCULATED VALUES
# ============================================================

if value_risk_col is not None:

    value_at_risk = pd.to_numeric(
        df[value_risk_col],
        errors="coerce"
    ).fillna(0).sum()

elif value_col is not None:

    value_at_risk = pd.to_numeric(
        df[value_col],
        errors="coerce"
    ).fillna(0).sum()

else:

    value_at_risk = 0


total_shipments = len(df)


high_risk_shipments = int(
    (df[risk_col] >= 60).sum()
)


average_risk = (
    df[risk_col].mean()
    if len(df) > 0
    else 0
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def risk_label(score):

    if score >= 80:
        return "CRITICAL"

    if score >= 60:
        return "HIGH"

    if score >= 35:
        return "MEDIUM"

    return "LOW"


def risk_class(score):

    if score >= 80:
        return "status-critical"

    if score >= 60:
        return "status-warning"

    return "status-normal"


def time_to_critical(score):

    return max(
        5,
        int(60 - score * 0.45)
    )


def get_value(row, column, default=0):

    if column is None:
        return default

    if column not in row.index:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    return value


def format_money(value):

    try:

        return f"₹{float(value):,.0f}"

    except (ValueError, TypeError):

        return "₹0"


# ============================================================
# AI EXPLANATION
# ============================================================

def explain_shipment(row):

    reasons = []

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    if deviation_col is not None:

        try:

            deviation = abs(
                float(
                    get_value(
                        row,
                        deviation_col,
                        0
                    )
                )
            )

        except (ValueError, TypeError):

            deviation = 0

        if deviation >= 2:

            reasons.append(
                (
                    "🌡️",
                    "Severe temperature deviation",
                    40
                )
            )

        elif deviation > 0:

            reasons.append(
                (
                    "🌡️",
                    "Temperature deviation detected",
                    25
                )
            )

    # --------------------------------------------------------
    # DOORS
    # --------------------------------------------------------

    if door_col is not None:

        try:

            doors = float(
                get_value(
                    row,
                    door_col,
                    0
                )
            )

        except (ValueError, TypeError):

            doors = 0

        if doors >= 6:

            reasons.append(
                (
                    "🚪",
                    "Frequent door openings",
                    25
                )
            )

        elif doors >= 3:

            reasons.append(
                (
                    "🚪",
                    "Elevated door activity",
                    15
                )
            )

    # --------------------------------------------------------
    # BATTERY
    # --------------------------------------------------------

    if battery_col is not None:

        try:

            battery = float(
                get_value(
                    row,
                    battery_col,
                    100
                )
            )

        except (ValueError, TypeError):

            battery = 100

        if battery < 30:

            reasons.append(
                (
                    "🔋",
                    "Critically low sensor battery",
                    20
                )
            )

        elif battery < 50:

            reasons.append(
                (
                    "🔋",
                    "Sensor battery degradation",
                    10
                )
            )

    # --------------------------------------------------------
    # SHOCK
    # --------------------------------------------------------

    if shock_col is not None:

        try:

            shocks = float(
                get_value(
                    row,
                    shock_col,
                    0
                )
            )

        except (ValueError, TypeError):

            shocks = 0

        if shocks >= 4:

            reasons.append(
                (
                    "📦",
                    "Multiple shock events",
                    20
                )
            )

        elif shocks >= 2:

            reasons.append(
                (
                    "📦",
                    "Handling shocks detected",
                    10
                )
            )

    # --------------------------------------------------------
    # ROUTE
    # --------------------------------------------------------

    if route_col is not None:

        route = str(
            get_value(
                row,
                route_col,
                ""
            )
        ).lower()

        if "high" in route:

            reasons.append(
                (
                    "🛣️",
                    "High route risk",
                    20
                )
            )

        elif "medium" in route:

            reasons.append(
                (
                    "🛣️",
                    "Medium route risk",
                    10
                )
            )

    if not reasons:

        reasons.append(
            (
                "✅",
                "No major abnormal factor detected",
                0
            )
        )

    return sorted(
        reasons,
        key=lambda x: x[2],
        reverse=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:27px;
            font-weight:850;
            color:#f4fbff;
        ">
            ❄️ ColdChainGuard
        </div>

        <div style="
            color:#94aabb;
            font-size:12px;
            margin-bottom:20px;
        ">
            Predictive Cold-Chain Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            color:#7f9aac;
            font-size:10px;
            font-weight:850;
            letter-spacing:1px;
            margin-bottom:8px;
        ">
            CONTROL CENTER
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # LOGGED-IN USER
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div style="
            background:#102a42;
            border:1px solid rgba(255,255,255,.08);
            border-radius:12px;
            padding:12px;
            margin-bottom:15px;
        ">

        <div style="
            font-size:15px;
            font-weight:800;
            color:#ffffff;
        ">
        👤 {st.session_state.user_name}
        </div>

        <div style="
            font-size:11px;
            color:#91a7b7;
            margin-top:4px;
        ">
        {st.session_state.user_email}
        </div>

        <div style="
            font-size:11px;
            color:#39edb3;
            margin-top:6px;
            font-weight:700;
        ">
        ● {st.session_state.user_role} ACCESS
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # USER MENU
    # ========================================================

    if USER_ROLE:

        st.markdown(
            """
            <div style="
                color:#7f9aac;
                font-size:10px;
                font-weight:850;
                letter-spacing:1px;
                margin-bottom:8px;
            ">
                USER CENTER
            </div>
            """,
            unsafe_allow_html=True
        )

        page = st.radio(
            "User Navigation",
            [
                "📡 Live AI Prediction",
                "📦 Shipment Intelligence",
                "🚨 Smart Alerts",
                "🧪 What-If Simulator",
                "🤖 ColdChain Copilot",
                "📍 Live Shipment Map",
                "🚀 About ColdChainGuard"
            ],
            label_visibility="collapsed"
        )

    # ========================================================
    # ADMIN MENU
    # ========================================================

    else:

        st.markdown(
            """
            <div style="
                color:#7f9aac;
                font-size:10px;
                font-weight:850;
                letter-spacing:1px;
                margin-bottom:8px;
            ">
                ADMIN CONTROL CENTER
            </div>
            """,
            unsafe_allow_html=True
        )

        page = st.radio(
            "Admin Navigation",
            [
                "Command Center",
                "📡 Live AI Prediction",
                "🚨 Smart Alerts",
                "📦 Shipment Intelligence",
                "🧪 What-If Simulator",
                "🚑 Rescue Center",
                "🤖 ColdChain Copilot",
                "📍 Live Shipment Map",
                "🚀 About ColdChainGuard"
            ],
            label_visibility="collapsed"
        )

    st.divider()

    # ========================================================
    # SYSTEM STATUS
    # ========================================================

    st.success("● AI SYSTEM ONLINE")

    st.caption("Prediction Engine: Active")
    st.caption("Anomaly Engine: Active")
    st.caption("Rescue Optimizer: Ready")
    st.caption("Explainability: Active")

    st.divider()

    # ========================================================
    # LOGOUT
    # ========================================================

    if st.button(
        "🚪 Logout",
        use_container_width=True
    ):

        logout()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
    ">

    <div>

    <div class="big-title">
    ❄️ ColdChainGuard
    </div>

    <div class="subtitle">
    Predict → Explain → Decide → Rescue
    </div>

    </div>

    <div class="status-normal"
    style="
        padding:8px 14px;
        font-size:11px;
    ">

    ● LIVE AI SYSTEM

    </div>

    </div>
    """,
    unsafe_allow_html=True
)

st.write("")


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "Command Center":

    st.header("📡 Command Center")

    total = len(df)

    critical = int(
        (df[risk_col] >= 80).sum()
    )

    high = int(
        (
            (df[risk_col] >= 60)
            &
            (df[risk_col] < 80)
        ).sum()
    )

    medium = int(
        (
            (df[risk_col] >= 35)
            &
            (df[risk_col] < 60)
        ).sum()
    )

    safe = total - critical - high - medium

    average_risk = (
        df[risk_col].mean()
        if len(df) > 0
        else 0
    )

    # ========================================================
    # KPI METRICS
    # ========================================================

    c1, c2, c3, c4, c5 = st.columns(5)

    metrics = [
        ("📦 Total Shipments", total),
        ("🚨 High Risk Shipments", high + critical),
        ("💰 Value at Risk", format_money(value_at_risk)),
        ("📊 Avg Risk Score", f"{average_risk:.1f}%"),
        ("🛣️ Route Risk", "Available" if route_col else "N/A")
    ]

    for col, (title, value) in zip(
        [c1, c2, c3, c4, c5],
        metrics
    ):

        with col:

            st.metric(
                title,
                value
            )

    st.write("")

    # ========================================================
    # RISK DISTRIBUTION GRAPHS
    # ========================================================

    left, right = st.columns(2)

    # --------------------------------------------------------
    # AI RISK DISTRIBUTION
    # --------------------------------------------------------

    with left:

        if "Priority_Level" in df.columns:

            risk_data = (
                df["Priority_Level"]
                .astype(str)
                .value_counts()
                .reset_index()
            )

            risk_data.columns = [
                "Risk Level",
                "Shipments"
            ]

            fig = px.bar(
                risk_data,
                x="Risk Level",
                y="Shipments",
                title="AI Risk Distribution",
                text="Shipments"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            risk_data = pd.DataFrame({
                "Risk Level": [
                    "LOW",
                    "MEDIUM",
                    "HIGH",
                    "CRITICAL"
                ],
                "Shipments": [
                    safe,
                    medium,
                    high,
                    critical
                ]
            })

            fig = px.bar(
                risk_data,
                x="Risk Level",
                y="Shipments",
                title="AI Risk Distribution",
                text="Shipments"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # --------------------------------------------------------
    # ROUTE RISK DISTRIBUTION
    # --------------------------------------------------------

    with right:

        if route_col is not None:

            route_data = (
                df[route_col]
                .astype(str)
                .value_counts()
                .reset_index()
            )

            route_data.columns = [
                "Route Risk",
                "Shipments"
            ]

            fig = px.pie(
                route_data,
                names="Route Risk",
                values="Shipments",
                title="Route Risk Distribution",
                hole=0.35
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "⚠️ Route_Risk column not found."
            )

    # ========================================================
    # HIGHEST RISK SHIPMENTS
    # ========================================================

    st.subheader("🚨 Highest Risk Shipments")

    # Use ML probability if available
    # otherwise use existing risk column

    if "ML_AT_RISK_Probability" in df.columns:

        sort_col = "ML_AT_RISK_Probability"

    elif "Risk_Probability" in df.columns:

        sort_col = "Risk_Probability"

    elif risk_col is not None:

        sort_col = risk_col

    else:

        st.warning(
            "⚠️ No risk column found in dashboard dataset."
        )

        sort_col = None

    if sort_col is not None:

        display_cols = [
            "Shipment_ID",
            "Product_Category",
            "Destination",
            "Destination_Name",
            "Transport_Mode",
            "Risk_Probability",
            "Risk_Score",
            "ML_Prediction",
            "ML_AT_RISK_Probability",
            "ML_Risk_Level",
            "Value_At_Risk_INR"
        ]

        display_cols = [
            col
            for col in display_cols
            if col in df.columns
        ]

        highest_risk = (
            df.sort_values(
                sort_col,
                ascending=False
            )[display_cols]
            .head(15)
        )

        st.dataframe(
            highest_risk,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # VALUE AT RISK
    # ========================================================

    st.markdown(
        f"""
        <div class="card">

        <div class="card-title">
        💰 TOTAL VALUE AT RISK
        </div>

        <div class="card-value">
        {format_money(value_at_risk)}
        </div>

        <div class="card-sub">
        Estimated financial exposure across monitored shipments
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )
elif page == "📡 Live AI Prediction":

    st.header("📡 Live AI Shipment Prediction")
    st.caption("SELECT ONE SHIPMENT → EDIT LIVE CONDITIONS → AI PREDICTION")

    if model is None:
        st.error("❌ Trained model could not be loaded.")
        st.code(model_error or f"Expected model file: {MODEL_FILE}")

    elif shipment_col is None or df.empty:
        st.error("❌ Shipment data is not available.")

    else:
        st.markdown(
            '<div class="status-normal">'
            '● REAL MODEL PREDICTION ENGINE ONLINE'
            '</div>',
            unsafe_allow_html=True
        )
        st.write("")

        # --------------------------------------------------------
        # ONE SHIPMENT SELECTOR ONLY
        # --------------------------------------------------------
        st.subheader("📦 Select One Shipment")
        shipment_options = (
            df[shipment_col]
            .astype(str)
            .str.strip()
            .drop_duplicates()
            .tolist()
        )

        selected_shipment = st.selectbox(
            "Shipment ID",
            shipment_options,
            key="live_selected_shipment"
        )

        selected_rows = df[
            df[shipment_col].astype(str).str.strip() == selected_shipment
        ]

        if selected_rows.empty:
            st.error(f"❌ Shipment {selected_shipment} was not found.")
        else:
            selected_row = selected_rows.iloc[0]

            st.success(
                f"✓ Shipment {selected_shipment} selected. "
                "Its current values have been loaded below."
            )

            # ----------------------------------------------------
            # ALL CONDITIONS ARE EDITABLE
            # ----------------------------------------------------
            live_data = build_live_input(
                selected_row,
                selected_shipment
            )

            st.write("")

            with st.expander(
                "🧾 Show exact data sent to the AI model",
                expanded=False
            ):
                st.dataframe(
                    live_data,
                    use_container_width=True,
                    hide_index=True
                )

            # ----------------------------------------------------
            # RUN PREDICTION
            # ----------------------------------------------------
            if st.button(
                "🔮 RUN AI PREDICTION FOR THIS SHIPMENT",
                use_container_width=True,
                key="run_live_prediction"
            ):
                try:
                    with st.spinner(
                        "🤖 AI model is analysing the selected shipment..."
                    ):
                        prediction, confidence, risk_probability, label, css_class = predict_live(
                            live_data
                        )

                    temp = None
                    if "Current_Temperature_C" in live_data.columns:
                        temp = float(
                            live_data.iloc[0]["Current_Temperature_C"]
                        )

                    st.markdown(
                        f'<div class="{css_class}" style="font-size:18px;">'
                        f'● {label} &nbsp;&nbsp;&nbsp; '
                        f'AI Prediction: {prediction}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric(
                        "🤖 Model Output",
                        str(prediction)
                    )

                    c2.metric(
                        "🧠 AI Confidence",
                        f"{confidence:.1f}%"
                    )

                    c3.metric(
                        "🚨 Risk Signal",
                        f"{risk_probability:.1f}%"
                        if risk_probability is not None
                        else "N/A"
                    )

                    c4.metric(
                        "🌡️ Live Temperature",
                        f"{temp:.1f} °C"
                        if temp is not None
                        else "N/A"
                    )

                    # ------------------------------------------------
                    # AI REASONING
                    # ------------------------------------------------
                    st.subheader("🧠 AI Reasoning")
                    reasons = []

                    if temp is not None:
                        if temp > 8:
                            reasons.append(
                                "🌡️ Temperature is above 8°C, which indicates "
                                "possible cold-chain exposure."
                            )
                        else:
                            reasons.append(
                                f"🌡️ Temperature is {temp:.1f}°C and is within "
                                "the monitored cold-chain threshold."
                            )

                    if "Battery_Level_Pct" in live_data.columns:
                        battery = float(live_data.iloc[0]["Battery_Level_Pct"])
                        if battery < 30:
                            reasons.append(
                                f"🔋 Sensor battery is low at {battery:.0f}% and "
                                "may affect continuous monitoring."
                            )
                        else:
                            reasons.append(
                                f"🔋 Sensor battery is healthy at {battery:.0f}%."
                            )

                    if "Door_Open_Count" in live_data.columns:
                        doors = float(live_data.iloc[0]["Door_Open_Count"])
                        if doors >= 6:
                            reasons.append(
                                f"🚪 Door activity is high ({doors:.0f} openings) "
                                "and may increase temperature exposure."
                            )
                        else:
                            reasons.append(
                                f"🚪 Door activity is currently {doors:.0f} openings."
                            )

                    if "Shock_Events" in live_data.columns:
                        shocks = float(live_data.iloc[0]["Shock_Events"])
                        if shocks >= 4:
                            reasons.append(
                                f"📦 {shocks:.0f} shock events indicate possible "
                                "handling damage."
                            )
                        else:
                            reasons.append(
                                f"📦 Shock activity is low at {shocks:.0f} events."
                            )

                    # Explicitly distinguish model confidence from risk.
                    reasons.append(
                        f"🤖 The trained model classified Shipment {selected_shipment} "
                        f"as {label} (output: {prediction})."
                    )
                    reasons.append(
                        f"🧠 The model is {confidence:.1f}% confident in its predicted class."
                    )
                    reasons.append(
                        f"🚨 Estimated probability of the risk class is "
                        f"{risk_probability:.1f}%."
                    )

                    for reason in reasons:
                        st.markdown(
                            f'<div class="reason">{reason}</div>',
                            unsafe_allow_html=True
                        )

                    # ------------------------------------------------
                    # RECOMMENDED ACTION
                    # ------------------------------------------------
                    st.subheader("🚑 Recommended Action")

                    if css_class == "status-critical":
                        st.warning(
                            "🚨 High-risk conditions detected. Inspect "
                            "refrigeration, verify the shipment, check "
                            "power and battery status, and move the "
                            "shipment to controlled conditions if required."
                        )
                    else:
                        st.success(
                            "✅ Continue live monitoring. The trained model "
                            "did not flag the current input as an immediate "
                            "high-risk incident."
                        )

                except Exception as e:
                    st.error("❌ Prediction failed.")
                    st.code(str(e))
                    st.info(
                        "The saved .pkl model may expect a different "
                        "preprocessing pipeline. Make sure the same "
                        "preprocessing used during training is included "
                        "with the model."
                    )

# ============================================================
# SMART ALERTS
# ============================================================

elif page == "🚨 Smart Alerts":

    st.header(
        "🚨 Smart Alert Center"
    )

    alerts = (
        df.sort_values(
            risk_col,
            ascending=False
        )
        .head(25)
    )

    for _, row in alerts.iterrows():

        risk = float(
            row[risk_col]
        )

        label = risk_label(
            risk
        )

        icons = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢"
        }

        st.markdown(
            f"""
            <div class="{risk_class(risk)}"
            style="margin-bottom:9px;">

            {icons[label]}

            <strong>
            {label}
            </strong>

            &nbsp;&nbsp;

            {get_value(row, shipment_col, "Shipment")}

            &nbsp;&nbsp;

            Risk:
            <strong>
            {risk:.1f}%
            </strong>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# SHIPMENT INTELLIGENCE
# ============================================================

elif page == "📦 Shipment Intelligence":

    st.header(
        "📦 Shipment Intelligence"
    )

    if shipment_col is None:

        st.warning(
            "Shipment column not found."
        )

    elif len(df) == 0:

        st.warning(
            "No shipment data available."
        )

    else:

        selected = st.selectbox(
            "Select Shipment",
            df[
                shipment_col
            ].astype(str).tolist()
        )

        selected_rows = df[
            df[
                shipment_col
            ].astype(str)
            == selected
        ]

        if selected_rows.empty:

            st.warning(
                "Shipment not found."
            )

        else:

            row = selected_rows.iloc[0]

            risk = float(
                row[risk_col]
            )

            try:
                temperature_value = float(
                    get_value(
                        row,
                        temperature_col,
                        0
                    )
                )
            except:
                temperature_value = 0

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "🌡️ Temperature",
                    f"{temperature_value:.1f} °C"
                )

            with c2:

                st.metric(
                    "🧠 AI Risk",
                    f"{risk:.1f}%"
                )

            with c3:

                st.metric(
                    "⏱️ Time to Critical",
                    f"{time_to_critical(risk)} min"
                )

            with c4:

                st.metric(
                    "❤️ Shipment Health",
                    f"{max(1, int(100-risk))}/100"
                )

            st.divider()

            left, right = st.columns(2)

            with left:

                st.subheader(
                    "📦 Shipment Profile"
                )

                st.markdown(
                    f"""
                    <div class="card">

                    Product:
                    <strong>
                    {get_value(row, product_col, "Unknown")}
                    </strong>

                    <br><br>

                    Destination:
                    <strong>
                    {get_value(row, destination_col, "Unknown")}
                    </strong>

                    <br><br>

                    Transport:
                    <strong>
                    {get_value(row, transport_col, "Unknown")}
                    </strong>

                    <br><br>

                    Temperature:
                    <strong>
                    {get_value(row, temperature_col, 0)}
                    °C
                    </strong>

                    <br><br>

                    Battery:
                    <strong>
                    {get_value(row, battery_col, 100)}
                    %
                    </strong>

                    <br><br>

                    Door Openings:
                    <strong>
                    {get_value(row, door_col, 0)}
                    </strong>

                    <br><br>

                    Shock Events:
                    <strong>
                    {get_value(row, shock_col, 0)}
                    </strong>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with right:

                st.subheader(
                    "🧠 AI Explanation"
                )

                for icon, reason, score in explain_shipment(row):

                    st.markdown(
                        f"""
                        <div class="reason">

                        {icon}

                        {reason}

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ============================================================
# WHAT IF SIMULATOR
# ============================================================

elif page == "🧪 What-If Simulator":

    st.header(
        "🧪 What-If Simulator"
    )

    if shipment_col is None:

        st.warning(
            "Shipment column not found."
        )

    elif len(df) == 0:

        st.warning(
            "No shipment data available."
        )

    else:

        selected = st.selectbox(
            "Select Shipment",
            df[
                shipment_col
            ].astype(str).tolist()
        )

        selected_rows = df[
            df[
                shipment_col
            ].astype(str)
            == selected
        ]

        if selected_rows.empty:

            st.warning(
                "Shipment not found."
            )

        else:

            row = selected_rows.iloc[0]

            try:
                original_temp = float(
                    get_value(
                        row,
                        temperature_col,
                        4
                    )
                )
            except:
                original_temp = 4.0

            try:
                original_doors = int(
                    get_value(
                        row,
                        door_col,
                        0
                    )
                )
            except:
                original_doors = 0

            try:
                original_battery = int(
                    get_value(
                        row,
                        battery_col,
                        80
                    )
                )
            except:
                original_battery = 80

            original_temp = float(
                np.clip(
                    original_temp,
                    -10,
                    30
                )
            )

            original_doors = int(
                np.clip(
                    original_doors,
                    0,
                    20
                )
            )

            original_battery = int(
                np.clip(
                    original_battery,
                    10,
                    100
                )
            )

            c1, c2 = st.columns(2)

            with c1:

                future_temp = st.slider(
                    "🌡️ Future Temperature",
                    -10.0,
                    30.0,
                    original_temp
                )

                future_doors = st.slider(
                    "🚪 Door Openings",
                    0,
                    20,
                    original_doors
                )

            with c2:

                future_battery = st.slider(
                    "🔋 Battery %",
                    10,
                    100,
                    original_battery
                )

                route_delay = st.slider(
                    "🛣️ Route Delay",
                    0,
                    180,
                    0
                )

            if st.button(
                "🔮 RUN WHAT-IF ANALYSIS"
            ):

                current_risk = float(
                    row[risk_col]
                )

                simulated = current_risk

                simulated += (
                    future_temp
                    -
                    original_temp
                ) * 7

                simulated += (
                    future_doors
                    -
                    original_doors
                ) * 1.5

                simulated -= (
                    future_battery
                    -
                    original_battery
                ) * .08

                simulated += (
                    route_delay * .08
                )

                simulated = np.clip(
                    simulated,
                    1,
                    99
                )

                st.divider()

                a, b, c = st.columns(3)

                with a:

                    st.metric(
                        "CURRENT RISK",
                        f"{current_risk:.1f}%"
                    )

                with b:

                    st.metric(
                        "SIMULATED RISK",
                        f"{simulated:.1f}%",
                        delta=f"{simulated-current_risk:.1f}%"
                    )

                with c:

                    st.metric(
                        "TIME TO CRITICAL",
                        f"{time_to_critical(simulated)} min"
                    )

                if simulated < current_risk:

                    st.success(
                        "✅ Simulated conditions improve shipment safety."
                    )

                else:

                    st.warning(
                        "⚠️ Simulated conditions increase risk."
                    )


# ============================================================
# RESCUE CENTER
# ============================================================

elif page == "🚑 Rescue Center":

    st.header(
        "🚑 Rescue Optimizer"
    )

    rescue = df.copy()

    rescue["Rescue_Priority"] = (
        rescue[risk_col]
    )

    rescue = rescue.sort_values(
        "Rescue_Priority",
        ascending=False
    )

    rescue["Priority"] = range(
        1,
        len(rescue) + 1
    )

    rescue["Time_to_Critical"] = (
        rescue[risk_col]
        .apply(time_to_critical)
    )

    columns = [

        "Priority",

        shipment_col,

        product_col,

        risk_col,

        "Rescue_Priority",

        "Time_to_Critical",

        value_risk_col

    ]

    columns = [

        c for c in columns

        if c is not None
        and c in rescue.columns

    ]

    st.dataframe(
        rescue[
            columns
        ].head(25),
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "🚑 Rescue priority combines AI risk and urgency."
    )


# ============================================================
# COLDCHAIN COPILOT
# ============================================================

elif page == "🤖 ColdChain Copilot":

    st.header(
        "🤖 ColdChain Copilot"
    )

    st.caption(
        "Dataset-aware AI decision assistant"
    )

    question = st.text_input(
        "Ask ColdChainGuard",
        placeholder="Which shipment needs attention first?"
    )

    if st.button(
        "🤖 ASK COPILOT"
    ):

        q = question.lower().strip()

        if not q:

            st.warning(
                "Please enter a question."
            )

        elif (
            "first" in q
            or "priority" in q
            or "attention" in q
        ):

            row = (
                df.sort_values(
                    risk_col,
                    ascending=False
                )
                .iloc[0]
            )

            st.success(
                f"""
                🚨 HIGHEST PRIORITY

                Shipment:
                {get_value(row, shipment_col, "Unknown")}

                AI Risk:
                {float(row[risk_col]):.1f}%

                Time to critical:
                {time_to_critical(float(row[risk_col]))} minutes.
                """
            )

        elif "risk" in q:

            st.info(
                f"""
                📊 Current average network risk:

                {df[risk_col].mean():.1f}%

                Critical shipments:

                {(df[risk_col] >= 80).sum()}

                High-risk shipments:

                {(
                    (df[risk_col] >= 60)
                    &
                    (df[risk_col] < 80)
                ).sum()}
                """
            )

        elif (
            "critical" in q
            or "danger" in q
        ):

            count = (
                df[risk_col] >= 80
            ).sum()

            st.warning(
                f"🔴 {count} critical shipments detected."
            )

        elif (
            "money" in q
            or "value" in q
            or "loss" in q
        ):

            # FIXED:
            # risk_value was undefined in the original code.
            # Correct variable is value_at_risk.

            st.info(
                f"""
                💰 Current estimated value at risk:

                {format_money(value_at_risk)}
                """
            )

        elif (
            "why" in q
            or "cause" in q
        ):

            row = (
                df.sort_values(
                    risk_col,
                    ascending=False
                )
                .iloc[0]
            )

            st.subheader(
                "🧠 Main risk factors"
            )

            for icon, reason, score in explain_shipment(row):

                st.markdown(
                    f"""
                    <div class="reason">
                    {icon} {reason}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                """
                Try:

                • Which shipment needs attention first?

                • What is the current risk?

                • Which shipments are critical?

                • What is the value at risk?

                • Why is the highest-risk shipment dangerous?
                """
            )
# ============================================================
# ABOUT
# ============================================================

elif page == "🚀 About ColdChainGuard":

    st.header(
        "🚀 About ColdChainGuard"
    )
    st.write(
        """
        ColdChainGuard is a predictive cold-chain intelligence platform that combines
        exploratory data analysis, visualization, risk scoring, AI-based decision support,
        alerts, rescue prioritization, what-if simulation, and shipment intelligence to
        improve the monitoring and management of temperature-sensitive logistics.

        It goes beyond simple monitoring by combining:

        • Predictive risk analysis
        • Explainable AI
        • What-If analysis
        • Rescue prioritization
        • Machine-learning model comparison
        """
    )

    st.divider()

    # ========================================================
    # 🎯 OBJECTIVES
    # ========================================================

    st.subheader(
        "🎯 OBJECTIVES OF COLDCHAIN GUARD"
    )

    st.write(
        """
        • To monitor the temperature of cold-chain products continuously during
          storage and transportation.

        • To detect abnormal temperature and environmental conditions.

        • To predict shipment risk using Machine Learning.

        • To prioritize critical shipments for immediate intervention.

        • To provide explainable AI-based risk insights.

        • To simulate sensor conditions using What-If analysis.

        • To support rescue and intervention decisions.
        """
    )

    st.divider()

    # ========================================================
    # 🚀 AI WORKFLOW
    # ========================================================

    st.subheader(
        "🚀 AI Workflow"
    )

    workflow = [

        "📡 MONITOR",

        "🕵️ DETECT",

        "🔮 PREDICT",

        "🧠 EXPLAIN",

        "🧪 SIMULATE",

        "🎯 DECIDE",

        "🚑 RESCUE",

        "💰 MEASURE"

    ]

    cols = st.columns(4)

    for i, item in enumerate(workflow):

        with cols[i % 4]:

            st.markdown(
                f"""
                <div class="card"
                style="margin-bottom:12px;">

                <div style="
                    font-size:18px;
                    font-weight:800;
                ">
                {item}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    # ========================================================
    # ⭐ WHY THIS PROJECT IS DIFFERENT
    # ========================================================

    st.subheader(
        "⭐ Why This Project Is Different"
    )

    features = [

        (
            "🔮",
            "Predictive AI",
            "Predict risk before the failure becomes critical."
        ),

        (
            "🧠",
            "Explainable AI",
            "Show why a shipment is considered dangerous."
        ),

        (
            "🧪",
            "What-If Simulation",
            "Test future conditions before they happen."
        ),

        (
            "🎯",
            "Intervention AI",
            "Rank possible rescue actions."
        ),

        (
            "🚑",
            "Rescue Optimization",
            "Prioritize the shipment that needs help first."
        )

    ]

    for icon, title, description in features:

        st.markdown(
            f"""
            <div class="card"
            style="margin-bottom:10px;">

            <div style="font-size:24px;">
            {icon}
            </div>

            <div class="card-title">
            {title}
            </div>

            <div style="
                color:#9fb2c0;
                margin-top:6px;
            ">
            {description}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # ========================================================
    # 🛠️ TECHNOLOGIES USED
    # ========================================================

    st.subheader(
        "🛠️ Technologies Used"
    )

    st.write(
        """
        Python • Pandas • NumPy • Scikit-learn

        Streamlit • Plotly • Machine Learning

        Data Visualization • Predictive Analytics
        """
    )

    st.divider()

    # ========================================================
    # 👥 TEAM MEMBERS
    # ========================================================

    st.subheader(
        "👥 Team Members"
    )

    team_members = [

        "👩‍💻 Amina Farooqui",

        "👩‍💻 Hiba Sayed",

        "👩‍💻 Zaveriya Khan"

    ]

    team_cols = st.columns(3)

    for i, member in enumerate(team_members):

        with team_cols[i]:

            st.markdown(
                f"""
                <div class="card"
                style="
                    text-align:center;
                    padding:20px;
                    margin-bottom:15px;
                ">

                <div style="
                    font-size:20px;
                    font-weight:700;
                ">
                {member}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    # ========================================================
    # 🎓 ACADEMIC INFORMATION
    # ========================================================

    st.subheader(
        "🎓 Academic Information"
    )

    st.write(
        """
        **Department:** Computer Engineering

        **Institute:** M.H. Saboo Siddik Polytechnic

        **Academic Year:** 2026–27

        **Project:** ColdChainGuard AI
        """
    )

    st.divider()

    # ========================================================
    # ❄️ FOOTER
    # ========================================================

    st.markdown(
        """
        <div class="footer">

        ❄️ <strong style="color:#bcd3df;">
        ColdChainGuard AI
        </strong>

        <br><br>

        Predictive Cold-Chain Intelligence

        <br><br>

        coldchainguardai@gmail.com

        </div>
        """,
        unsafe_allow_html=True
    )
# ============================================================
# 📍 LIVE SHIPMENT MAP
# ============================================================

elif page == "📍 Live Shipment Map":

    st.markdown("## 📍 Live Shipment Map")

    st.caption(
        "Interactive visualization of shipment locations, "
        "destinations, routes, risk levels and rescue facilities."
    )

    # ========================================================
    # CITY COORDINATES
    # ========================================================

    locations = {

        "Mumbai": (19.0760, 72.8777),

        "Delhi": (28.6139, 77.2090),

        "Bangalore": (12.9716, 77.5946),

        "Hyderabad": (17.3850, 78.4867),

        "Chennai": (13.0827, 80.2707),

        "Kolkata": (22.5726, 88.3639),

        "Pune": (18.5204, 73.8567),

        "Ahmedabad": (23.0225, 72.5714),

        "Jaipur": (26.9124, 75.7873),

        "Lucknow": (26.8467, 80.9462)

    }

    cities = list(locations.keys())

    # ========================================================
    # CREATE MAP DATA
    # ========================================================

    np.random.seed(42)

    map_df = df.copy()

    # ========================================================
    # ORIGIN
    # ========================================================

    if "Origin" not in map_df.columns:

        map_df["Origin"] = np.random.choice(
            cities,
            len(map_df)
        )

    # ========================================================
    # DESTINATION
    # ========================================================

    if "Destination" not in map_df.columns:

        map_df["Destination"] = np.random.choice(
            cities,
            len(map_df)
        )

    # Make sure Origin != Destination

    for index in map_df.index:

        if (
            map_df.loc[index, "Origin"]
            ==
            map_df.loc[index, "Destination"]
        ):

            available = [
                city
                for city in cities
                if city != map_df.loc[index, "Origin"]
            ]

            map_df.loc[index, "Destination"] = np.random.choice(
                available
            )

    # ========================================================
    # RISK LEVEL
    # ========================================================

    if "ML_Risk_Level" in map_df.columns:

        map_df["Risk_Level"] = (
            map_df["ML_Risk_Level"]
            .astype(str)
            .str.upper()
        )

    elif "ML_AT_RISK_Probability" in map_df.columns:

        def calculate_map_risk(probability):

            try:
                probability = float(probability)
            except:
                probability = 0

            if probability >= 80:
                return "CRITICAL"

            elif probability >= 60:
                return "HIGH"

            elif probability >= 35:
                return "MEDIUM"

            else:
                return "LOW"

        map_df["Risk_Level"] = (
            pd.to_numeric(
                map_df["ML_AT_RISK_Probability"],
                errors="coerce"
            )
            .fillna(0)
            .apply(calculate_map_risk)
        )

    elif "Shipment_Status" in map_df.columns:

        map_df["Risk_Level"] = (
            map_df["Shipment_Status"]
            .astype(str)
            .str.upper()
            .map({
                "SAFE": "LOW",
                "AT_RISK": "HIGH"
            })
            .fillna("MEDIUM")
        )

    else:

        map_df["Risk_Level"] = "LOW"

    # ========================================================
    # RISK COLORS
    # ========================================================

    risk_colors = {

        "LOW": "green",

        "MEDIUM": "yellow",

        "HIGH": "orange",

        "CRITICAL": "red"

    }

    # ========================================================
    # CREATE MAP
    # ========================================================

    fig = go.Figure()

    # ========================================================
    # ROUTES
    # ========================================================

    for _, row in map_df.iterrows():

        origin = row["Origin"]

        destination = row["Destination"]

        if origin not in locations:
            continue

        if destination not in locations:
            continue

        origin_lat, origin_lon = locations[origin]

        destination_lat, destination_lon = locations[destination]

        shipment_id = row.get(
            "Shipment_ID",
            "Unknown"
        )

        risk = row["Risk_Level"]

        probability = row.get(
            "ML_AT_RISK_Probability",
            np.nan
        )

        if pd.notna(probability):

            probability_text = (
                f"{float(probability):.1f}%"
            )

        else:

            probability_text = "N/A"

        route_color = risk_colors.get(
            risk,
            "blue"
        )

        fig.add_trace(

            go.Scattergeo(

                lat=[
                    origin_lat,
                    destination_lat
                ],

                lon=[
                    origin_lon,
                    destination_lon
                ],

                mode="lines",

                line=dict(
                    width=2,
                    color=route_color
                ),

                opacity=0.55,

                hoverinfo="text",

                hovertext=(
                    f"<b>🚚 Shipment {shipment_id}</b><br>"
                    f"📍 {origin}<br>"
                    f"🏁 {destination}<br>"
                    f"⚠️ Risk: {risk}<br>"
                    f"📊 Probability: "
                    f"{probability_text}"
                ),

                showlegend=False

            )
        )

    # ========================================================
    # SHIPMENT LOCATIONS
    # ========================================================

    shipment_lat = []

    shipment_lon = []

    shipment_text = []

    shipment_colors = []

    for _, row in map_df.iterrows():

        origin = row["Origin"]

        if origin not in locations:
            continue

        lat, lon = locations[origin]

        shipment_id = row.get(
            "Shipment_ID",
            "Unknown"
        )

        destination = row["Destination"]

        risk = row["Risk_Level"]

        probability = row.get(
            "ML_AT_RISK_Probability",
            np.nan
        )

        if pd.notna(probability):

            probability_text = (
                f"{float(probability):.1f}%"
            )

        else:

            probability_text = "N/A"

        shipment_lat.append(lat)

        shipment_lon.append(lon)

        shipment_colors.append(
            risk_colors.get(
                risk,
                "blue"
            )
        )

        shipment_text.append(

            f"<b>🚚 Shipment {shipment_id}</b><br>"
            f"📍 Origin: {origin}<br>"
            f"🏁 Destination: {destination}<br>"
            f"⚠️ Risk: {risk}<br>"
            f"📊 AT-RISK Probability: "
            f"{probability_text}"

        )

    # ========================================================
    # SHIPMENT MARKERS
    # ========================================================

    if shipment_lat:

        fig.add_trace(

            go.Scattergeo(

                lat=shipment_lat,

                lon=shipment_lon,

                mode="markers",

                marker=dict(

                    size=12,

                    color=shipment_colors,

                    line=dict(
                        width=2,
                        color="white"
                    )

                ),

                text=shipment_text,

                hoverinfo="text",

                name="🚚 Shipments"

            )
        )

    # ========================================================
    # 🏥 RESCUE FACILITIES
    # ========================================================

    rescue_facilities = {

        "Mumbai Cold Storage":
            (19.0760, 72.8777),

        "Delhi Emergency Hub":
            (28.6139, 77.2090),

        "Bangalore Cold Hub":
            (12.9716, 77.5946),

        "Hyderabad Rescue Center":
            (17.3850, 78.4867),

        "Chennai Cold Hub":
            (13.0827, 80.2707),

        "Kolkata Emergency Hub":
            (22.5726, 88.3639)

    }

    fig.add_trace(

        go.Scattergeo(

            lat=[
                value[0]
                for value in rescue_facilities.values()
            ],

            lon=[
                value[1]
                for value in rescue_facilities.values()
            ],

            mode="markers",

            marker=dict(

                size=15,

                symbol="star",

                color="blue",

                line=dict(
                    width=2,
                    color="white"
                )

            ),

            text=[
                f"🏥 {name}"
                for name in rescue_facilities.keys()
            ],

            hoverinfo="text",

            name="🏥 Rescue Facilities"

        )
    )

    # ========================================================
    # INDIA MAP
    # ========================================================

    fig.update_geos(

        scope="asia",

        projection_type="natural earth",

        showland=True,

        landcolor="rgb(235, 235, 235)",

        showocean=True,

        oceancolor="rgb(220, 235, 245)",

        showcountries=True,

        countrycolor="gray",

        center=dict(
            lat=21,
            lon=78
        ),

        lataxis_range=[
            7,
            36
        ],

        lonaxis_range=[
            67,
            98
        ]

    )

    # ========================================================
    # MAP DESIGN
    # ========================================================

    fig.update_layout(

        height=650,

        margin=dict(
            l=0,
            r=0,
            t=40,
            b=0
        ),

        title=dict(

            text="🇮🇳 Cold-Chain Shipment Network",

            x=0.5,

            xanchor="center"

        ),

        legend=dict(

            orientation="h",

            yanchor="bottom",

            y=0.01,

            xanchor="center",

            x=0.5

        )

    )

    # ========================================================
    # DISPLAY MAP
    # ========================================================

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # LEGEND
    # ========================================================

    st.markdown("### 🗺️ Risk Legend")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("🟢 **LOW**")

    with col2:
        st.markdown("🟡 **MEDIUM**")

    with col3:
        st.markdown("🟠 **HIGH**")

    with col4:
        st.markdown("🔴 **CRITICAL**")

    with col5:
        st.markdown("🔵 **RESCUE HUB**")

    # ========================================================
    # 🚚 SELECT SHIPMENT
    # ========================================================

    st.markdown("---")

    st.markdown(
        "### 🚚 Shipment Intelligence"
    )

    if "Shipment_ID" in map_df.columns:

        shipment_list = (
            map_df["Shipment_ID"]
            .astype(str)
            .tolist()
        )

    else:

        shipment_list = [
            str(index)
            for index in map_df.index
        ]

    if shipment_list:

        selected_shipment = st.selectbox(

            "🔎 Click/select a shipment to inspect",

            shipment_list

        )

        # ====================================================
        # SELECTED SHIPMENT
        # ====================================================

        if "Shipment_ID" in map_df.columns:

            selected = map_df[
                map_df["Shipment_ID"]
                .astype(str)
                ==
                selected_shipment
            ]

        else:

            selected = map_df.loc[
                [int(selected_shipment)]
            ]

        if not selected.empty:

            shipment = selected.iloc[0]

            shipment_id = shipment.get(
                "Shipment_ID",
                selected_shipment
            )

            origin = shipment["Origin"]

            destination = shipment["Destination"]

            risk = shipment["Risk_Level"]

            probability = shipment.get(
                "ML_AT_RISK_Probability",
                np.nan
            )

            # ================================================
            # SUMMARY
            # ================================================

            st.markdown(
                "#### 📊 Shipment Overview"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Shipment",
                    str(shipment_id)
                )

            with c2:

                st.metric(
                    "📍 Location",
                    origin
                )

            with c3:

                st.metric(
                    "🏁 Destination",
                    destination
                )

            with c4:

                if pd.notna(probability):

                    value = (
                        f"{float(probability):.1f}%"
                    )

                else:

                    value = "N/A"

                st.metric(
                    "⚠️ Risk Probability",
                    value
                )

            # ================================================
            # RISK MESSAGE
            # ================================================

            if risk == "CRITICAL":

                st.error(
                    "🔴 CRITICAL SHIPMENT — "
                    "Immediate intervention recommended."
                )

            elif risk == "HIGH":

                st.warning(
                    "🟠 HIGH-RISK SHIPMENT — "
                    "Operator attention recommended."
                )

            elif risk == "MEDIUM":

                st.warning(
                    "🟡 MEDIUM-RISK SHIPMENT — "
                    "Continue monitoring."
                )

            else:

                st.success(
                    "🟢 LOW-RISK SHIPMENT — "
                    "Shipment currently appears stable."
                )

            # ================================================
            # FIND NEAREST RESCUE FACILITY
            # ================================================

            origin_lat, origin_lon = locations[origin]

            nearest_facility = None

            nearest_distance = float("inf")

            for facility, coordinates in rescue_facilities.items():

                facility_lat, facility_lon = coordinates

                distance = (

                    (origin_lat - facility_lat) ** 2

                    +

                    (origin_lon - facility_lon) ** 2

                ) ** 0.5

                if distance < nearest_distance:

                    nearest_distance = distance

                    nearest_facility = facility

            st.markdown(
                f"### 🏥 Nearest Rescue Facility: "
                f"**{nearest_facility}**"
            )

            # ================================================
            # DETAILS
            # ================================================

            st.markdown(
                "#### 🔎 Shipment Details"
            )

            detail_columns = [

                "Shipment_ID",

                "Shipment_Status",

                "ML_Prediction",

                "ML_Risk_Level",

                "ML_AT_RISK_Probability",

                "Product_Category",

                "Product_Type",

                "Transport_Mode",

                "Route_Risk",

                "Refrigeration_Status",

                "Current_Temperature_C",

                "Avg_Temperature_C",

                "Max_Temperature_C",

                "Temperature_Deviation_C",

                "Temperature_Excursion_Minutes",

                "Humidity_Pct",

                "Door_Open_Count",

                "Battery_Level_Pct",

                "Shock_Events",

                "Distance_KM",

                "Duration_Hours",

                "Estimated_Value_INR"

            ]

            detail_columns = [

                column

                for column in detail_columns

                if column in shipment.index

            ]

            if detail_columns:

                st.dataframe(

                    shipment[
                        detail_columns
                    ].to_frame("Value"),

                    use_container_width=True

                )

    else:

        st.warning(
            "⚠️ No shipments available for the map."
        )

    # ========================================================
    # 🚨 CRITICAL SHIPMENTS
    # ========================================================

    st.markdown("---")

    st.markdown(
        "## 🚨 Critical Shipments"
    )

    critical = map_df[
        map_df["Risk_Level"] == "CRITICAL"
    ]

    if len(critical) > 0:

        columns = [

            "Shipment_ID",

            "Origin",

            "Destination",

            "Risk_Level",

            "ML_AT_RISK_Probability"

        ]

        columns = [

            column

            for column in columns

            if column in critical.columns

        ]

        critical_display = critical[columns]

        if "ML_AT_RISK_Probability" in critical_display.columns:

            critical_display = critical_display.sort_values(
                "ML_AT_RISK_Probability",
                ascending=False
            )

        st.dataframe(

            critical_display,

            use_container_width=True

        )

    else:

        st.success(
            "✅ No critical shipments currently detected."
        )

    # ========================================================
    # NOTE
    # ========================================================

    st.markdown("---")

    st.info(
        "Prototype note: city-level coordinates are currently "
        "used for visualization. A production version can replace "
        "these with real GPS/IoT coordinates."
    )
