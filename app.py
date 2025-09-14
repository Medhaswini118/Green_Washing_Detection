import streamlit as st
import joblib
import pandas as pd

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Greenwashing Detector 🌱",
    page_icon="🌍",
    layout="centered",
    initial_sidebar_state="collapsed"
)
# ------------------ CUSTOM STYLES ----------------
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(135deg, #d0f0e0 0%, #ffffff 100%);
        }
        .stApp {
            background: linear-gradient(135deg, #d0f0e0 0%, #ffffff 100%);
        }
    </style>
    """,
    unsafe_allow_html=True
)
# ------------------ TEXT COLOR FIX ----------------
st.markdown(
    """
    <style>
        /* All markdown text inside the app */
        .stMarkdown p,
        .stMarkdown h3,
        .stMarkdown h4,
        .stMarkdown div {
            color: #1b5e20 !important;   /* nice dark green */
        }

        /* Captions / small help text */
        .stCaption, .st-emotion-cache-0 {
            color: #444 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------ LOAD MODEL -------------------
model = joblib.load("greenwash_detector_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ------------------ CUSTOM STYLES ----------------
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(135deg, #e8f5e9 0%, #ffffff 100%);
        }
        .big-title {
            font-size: 55px;
            font-weight: 800;
            color: #1b5e20;
            text-align: center;
            padding-top: 10px;
        }
        .subtitle {
            font-size: 22px;
            text-align: center;
            color: #444;
            margin-bottom: 25px;
        }
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
        /* Force Browse button background and text */
        .stFileUploader [data-testid="stFileUploadDropzone"] button {
            background-color: #4caf50 !important;  /* light green background */
            color: #ffffff !important;             /* white text */
            font-weight: 700;
            border-radius: 6px;
            padding: 0.25rem 0.75rem;
        }

        /* Ensure hover keeps text visible */
        .stFileUploader [data-testid="stFileUploadDropzone"] button:hover {
            background-color: #43a047 !important;
            color: #ffffff !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------ HEADER -----------------------
st.markdown(
    """
    <div style="text-align:center;
                font-size:50px;
                font-weight:800;
                color:#1b5e20;
                padding-top:5px;">
        🌱 GREENWASHING DETECTOR
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="text-align:center;
                font-size:20px;
                color:#444;
                margin-bottom:25px;">
        Analyse environmental claims and discover if they are
        <b style="color:green;">Genuine</b> or
        <b style="color:red;">Greenwashing</b>.
    </div>
    """,
    unsafe_allow_html=True
)
# ------------------ SIDEBAR ----------------------
st.sidebar.header("📌 About")
st.sidebar.info(
    "This app uses a Machine Learning model trained on environmental claims to detect potential greenwashing."
    "\n\nCreated with ❤️ using Streamlit & scikit-learn."
)

# ------------------ INPUT AREA -------------------
st.markdown("### ✍️ Enter an Environmental Claim")
user_input = st.text_area("", height=120, placeholder="e.g., Our packaging is 100% eco-friendly and compostable.")

# ------------------ PREDICT ----------------------
if st.button("🔍 Analyse Claim"):
    if user_input.strip():
        X = vectorizer.transform([user_input])
        pred = model.predict(X)[0]

        if pred == "genuine":
            st.success("✅ **Genuine Claim!** 🌍 This statement looks environmentally credible.")
        else:
            st.error("🚩 **Possible Greenwashing!** ❌ This claim may lack real sustainability backing.")
    else:
        st.warning("Please type a claim to analyse.")

st.write("---")

# ------------------ BATCH UPLOAD -----------------
st.markdown("### 📂 Test Multiple Claims (CSV)")
st.caption("Upload a CSV with a column named `Claim` to check many statements at once.")
file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    data = pd.read_csv(file)
    if "Claim" in data.columns:
        X_batch = vectorizer.transform(data["Claim"])
        preds = model.predict(X_batch)
        data["Prediction"] = preds
        st.dataframe(data)
        st.download_button(
            "⬇️ Download Results",
            data.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv"
        )
    else:
        st.error("CSV must contain a column named **Claim**.")

# ------------------ FOOTER -----------------------
st.write("---")
st.caption("🌿 Developed as a project on Environmental Monitoring & Pollution Control • 2025")
