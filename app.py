from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# NYC Room Type Classifier — Streamlit UI
# Put Model_Pipeline.pkl beside this file.
# Dataset is optional for prediction, but powers the Explore tab.
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "Model_Pipeline.pkl"
DATA_PATH = ROOT / "data" / "AB_NYC_2019.csv"

st.set_page_config(
    page_title="NYC StaySense | Room Type AI",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Visual theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #0b1020; color: #edf2ff; }
    [data-testid="stSidebar"] { background: #11182b; border-right: 1px solid #26314b; }
    [data-testid="stHeader"] { background: rgba(11,16,32,0.85); }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -0.035em; }
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 22px; margin-bottom: 1rem;
        background: radial-gradient(circle at 90% 10%, rgba(88, 112, 255, .30), transparent 32%),
                    linear-gradient(120deg, #172544 0%, #182039 55%, #242044 100%);
        border: 1px solid rgba(167, 184, 255, .18);
    }
    .eyebrow { color: #a5b4fc; text-transform: uppercase; font-size: .72rem; font-weight: 800; letter-spacing: .16em; }
    .hero h1 { color: #f8faff; font-size: clamp(2rem, 4vw, 3.15rem); margin: .3rem 0 .5rem 0; }
    .hero p { color: #c4cde4; font-size: 1rem; margin-bottom: .2rem; max-width: 800px; }
    .metric-card {
        background: linear-gradient(145deg, #151e34, #11182a);
        border: 1px solid #2a3653; border-radius: 16px; padding: 1rem 1.1rem;
        min-height: 105px;
    }
    .metric-label { color: #9caac8; font-size: .78rem; font-weight: 600; margin-bottom: .35rem; }
    .metric-value { color: #f5f7ff; font-size: 1.65rem; font-weight: 800; line-height: 1.2; }
    .metric-note { color: #91a1c3; font-size: .75rem; margin-top: .35rem; }
    .result-card {
        border-radius: 20px; padding: 1.25rem 1.35rem;
        background: linear-gradient(135deg, rgba(52,211,153,.14), rgba(99,102,241,.12));
        border: 1px solid rgba(110,231,183,.28);
    }
    .result-label { color: #a7f3d0; font-size: .78rem; text-transform: uppercase; letter-spacing: .13em; font-weight: 800; }
    .result-name { color: #ffffff; font-family: 'Manrope', sans-serif; font-size: 1.8rem; font-weight: 800; margin: .35rem 0; }
    .muted { color: #9caac8; font-size: .88rem; }
    div[data-testid="stForm"] { background: #11182a; padding: 1rem 1.2rem; border: 1px solid #26314b; border-radius: 18px; }
    div[data-testid="stTabs"] button { font-weight: 700; }
    .small-note { color: #8795b4; font-size: .78rem; }
    div.stButton > button, div.stFormSubmitButton > button {
        border-radius: 12px; border: 0; font-weight: 800; min-height: 2.8rem;
        background: linear-gradient(90deg, #6d7cff, #8b5cf6); color: white;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover { color: white; border: 1px solid #a5b4fc; }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOM_COLORS = {
    "Entire home/apt": "#818cf8",
    "Private room": "#34d399",
    "Shared room": "#fbbf24",
}
ROOM_ORDER = ["Entire home/apt", "Private room", "Shared room"]

@st.cache_resource(show_spinner="Loading your trained model…")
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        return None
    data = pd.read_csv(DATA_PATH)
    if "reviews_per_month" in data.columns:
        data["reviews_per_month"] = data["reviews_per_month"].fillna(0)
    return data

model = None
model_error = None
try:
    model = load_model()
except Exception as exc:
    model_error = str(exc)

df = load_data()

with st.sidebar:
    st.markdown("## 🏙️ StaySense")
    st.caption("NYC room-type intelligence")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["✨ Predict room type", "📊 Explore NYC data", "🧠 Model & methodology"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**Project snapshot**")
    st.markdown("`3` room categories")
    st.markdown("`Random Forest` classifier")
    st.markdown("`NYC Airbnb · 2019` dataset")
    st.markdown("")
    st.caption("Built with Streamlit + scikit-learn")
    if model is None:
        st.warning("Model file not detected or could not be loaded.")
        if model_error:
            st.caption("Load error: " + model_error[:220])

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">NYC • STAYS • MACHINE LEARNING</div>
      <h1>Find the room type behind a listing.</h1>
      <p>Explore New York City Airbnb listings and predict whether a listing is an entire home, a private room, or a shared room — using your trained classification pipeline.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if df is not None:
    total_listings = len(df)
    borough_count = df["neighbourhood_group"].nunique()
    neighborhood_count = df["neighbourhood"].nunique()
    median_price = df["price"].median()
else:
    total_listings, borough_count, neighborhood_count, median_price = 48895, 5, 221, 106

m1, m2, m3, m4 = st.columns(4)
for col, label, value, note in [
    (m1, "LISTINGS IN DATA", f"{total_listings:,}", "NYC Airbnb 2019"),
    (m2, "BOROUGHS", f"{borough_count}", "Across New York City"),
    (m3, "NEIGHBORHOODS", f"{neighborhood_count}", "Represented in the dataset"),
    (m4, "MEDIAN NIGHTLY PRICE", f"${median_price:,.0f}", "Dataset median, USD"),
]:
    col.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

if page == "✨ Predict room type":
    st.subheader("✨ Listing profile")
    st.markdown(
        '<p class="muted">Set the location and listing details below. The saved pipeline handles imputation, scaling, encoding, and classification.</p>',
        unsafe_allow_html=True,
    )

    if df is not None:
        boroughs = sorted(df["neighbourhood_group"].dropna().unique().tolist())
    else:
        boroughs = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    with st.form("prediction_form"):
        st.markdown("#### 01 / Location")
        loc1, loc2 = st.columns(2)
        with loc1:
            borough = st.selectbox("Borough", boroughs, index=boroughs.index("Manhattan") if "Manhattan" in boroughs else 0)
        if df is not None:
            borough_df = df[df["neighbourhood_group"] == borough]
            neighborhoods = sorted(borough_df["neighbourhood"].dropna().unique().tolist())
        else:
            neighborhoods = ["Midtown", "Bedford-Stuyvesant", "Astoria"]
        with loc2:
            default_neighborhood = "Midtown" if "Midtown" in neighborhoods else neighborhoods[0]
            neighborhood = st.selectbox("Neighborhood", neighborhoods, index=neighborhoods.index(default_neighborhood))

        if df is not None:
            loc_rows = df[(df["neighbourhood_group"] == borough) & (df["neighbourhood"] == neighborhood)]
            if loc_rows.empty:
                loc_rows = df[df["neighbourhood_group"] == borough]
            default_lat = float(loc_rows["latitude"].median())
            default_lon = float(loc_rows["longitude"].median())
        else:
            default_lat, default_lon = 40.758, -73.985

        st.markdown("#### 02 / Listing details")
        f1, f2, f3 = st.columns(3)
        with f1:
            price = st.number_input("Price per night (USD)", min_value=0, max_value=799, value=150, step=10, help="Your training notebook capped price at its 99th percentile (~$799).")
            minimum_nights = st.number_input("Minimum nights", min_value=1, max_value=45, value=2, step=1, help="Your training notebook capped minimum nights at its 99th percentile (45).")
        with f2:
            number_of_reviews = st.number_input("Number of reviews", min_value=0, max_value=10000, value=10, step=1)
            reviews_per_month = st.number_input("Reviews per month", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
        with f3:
            calculated_host_listings_count = st.number_input("Host's listing count", min_value=1, max_value=1000, value=1, step=1)
            availability_365 = st.slider("Availability (days/year)", min_value=0, max_value=365, value=180)

        st.markdown("#### 03 / Geographic coordinates")
        g1, g2 = st.columns(2)
        with g1:
            latitude = st.number_input("Latitude", min_value=40.4, max_value=41.0, value=default_lat, format="%.5f", help="Defaults to the median coordinates for the selected neighborhood.")
        with g2:
            longitude = st.number_input("Longitude", min_value=-74.3, max_value=-73.6, value=default_lon, format="%.5f", help="Defaults to the median coordinates for the selected neighborhood.")

        submitted = st.form_submit_button("✦ Predict room type", use_container_width=True)

    if submitted:
        if model is None:
            st.error("I couldn't load Model_Pipeline.pkl. Put it in the same folder as app.py and install the requirements listed in requirements.txt.")
        else:
            input_df = pd.DataFrame([{
                "neighbourhood_group": borough,
                "neighbourhood": neighborhood,
                "latitude": latitude,
                "longitude": longitude,
                "price": price,
                "minimum_nights": minimum_nights,
                "number_of_reviews": number_of_reviews,
                "reviews_per_month": reviews_per_month,
                "calculated_host_listings_count": calculated_host_listings_count,
                "availability_365": availability_365,
            }])
            try:
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None
                st.markdown("")
                st.markdown(
                    f'<div class="result-card"><div class="result-label">MODEL PREDICTION</div>'
                    f'<div class="result-name">{prediction}</div>'
                    f'<div class="muted">For a listing in {neighborhood}, {borough} · ${price:,.0f}/night · {minimum_nights} minimum night(s)</div></div>',
                    unsafe_allow_html=True,
                )
                if probabilities is not None:
                    prob_df = pd.DataFrame({"Room type": model.classes_, "Probability": probabilities})
                    prob_df["Probability (%)"] = prob_df["Probability"] * 100
                    prob_df = prob_df.sort_values("Probability", ascending=True)
                    left, right = st.columns([1.3, 1])
                    with left:
                        fig = px.bar(
                            prob_df, x="Probability (%)", y="Room type", orientation="h",
                            color="Room type", color_discrete_map=ROOM_COLORS,
                            text=prob_df["Probability (%)"].map(lambda x: f"{x:.1f}%"),
                            title="Prediction confidence by class",
                        )
                        fig.update_layout(
                            template="plotly_dark", height=330, showlegend=False,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=8, r=15, t=55, b=10), xaxis_range=[0, 105],
                            xaxis_title="Model probability", yaxis_title="",
                        )
                        fig.update_traces(textposition="outside", cliponaxis=False)
                        st.plotly_chart(fig, use_container_width=True)
                    with right:
                        st.markdown("#### How to read this")
                        top_prob = float(max(probabilities)) * 100
                        st.metric("Highest class probability", f"{top_prob:.1f}%")
                        st.progress(min(max(top_prob / 100, 0.0), 1.0))
                        st.caption("These are model-estimated class probabilities, not a guarantee. Real listings may differ.")
                        st.markdown("**Submitted profile**")
                        st.write(f"📍 {borough} · {neighborhood}")
                        st.write(f"💵 ${price:,.0f} per night")
                        st.write(f"🗓️ {minimum_nights} minimum night(s)")
                st.session_state["last_prediction"] = {
                    "prediction": str(prediction),
                    "borough": borough,
                    "neighborhood": neighborhood,
                    "price": int(price),
                }
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.info("Check that this model file matches the scikit-learn version in requirements.txt.")
    else:
        st.info("Fill in the listing profile and select **Predict room type** to run your trained model.")

elif page == "📊 Explore NYC data":
    st.subheader("📊 Explore the dataset")
    if df is None:
        st.warning("Dataset file not found. Place AB_NYC_2019.csv inside the data/ folder to enable these charts.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            counts = df["room_type"].value_counts().reindex(ROOM_ORDER).fillna(0).reset_index()
            counts.columns = ["Room type", "Listings"]
            fig = px.bar(counts, x="Room type", y="Listings", color="Room type",
                         color_discrete_map=ROOM_COLORS, text="Listings",
                         title="Room-type distribution")
            fig.update_layout(template="plotly_dark", showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=5,r=5,t=55,b=5))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            price_df = df[df["price"].between(0, df["price"].quantile(.99))]
            fig = px.box(price_df, x="room_type", y="price", color="room_type",
                         color_discrete_map=ROOM_COLORS, points=False,
                         category_orders={"room_type": ROOM_ORDER},
                         title="Nightly price by room type (99th percentile view)")
            fig.update_layout(template="plotly_dark", showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=5,r=5,t=55,b=5),
                              xaxis_title="", yaxis_title="Price (USD)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Neighborhood snapshot")
        borough_filter = st.selectbox("Filter by borough", ["All boroughs"] + sorted(df["neighbourhood_group"].dropna().unique().tolist()))
        explore_df = df if borough_filter == "All boroughs" else df[df["neighbourhood_group"] == borough_filter]
        n1, n2, n3 = st.columns(3)
        n1.metric("Listings in selection", f"{len(explore_df):,}")
        n2.metric("Median nightly price", f"${explore_df['price'].median():,.0f}")
        n3.metric("Neighborhoods", f"{explore_df['neighbourhood'].nunique():,}")

        borough_mix = (explore_df.groupby(["neighbourhood_group", "room_type"]).size()
                       .reset_index(name="Listings"))
        fig = px.bar(borough_mix, x="neighbourhood_group", y="Listings", color="room_type",
                     barmode="stack", color_discrete_map=ROOM_COLORS,
                     category_orders={"room_type": ROOM_ORDER}, title="Room types across boroughs")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=5,r=5,t=55,b=5),
                          xaxis_title="", yaxis_title="Listings", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Preview the raw dataset"):
            display_cols = ["neighbourhood_group", "neighbourhood", "room_type", "price",
                            "minimum_nights", "number_of_reviews", "availability_365"]
            st.dataframe(explore_df[display_cols].head(100), use_container_width=True, hide_index=True)
            st.download_button("Download filtered preview as CSV",
                               data=explore_df[display_cols].head(1000).to_csv(index=False).encode("utf-8"),
                               file_name="nyc_airbnb_preview.csv", mime="text/csv")

elif page == "🧠 Model & methodology":
    st.subheader("🧠 About this classifier")
    st.markdown(
        "This app uses the fitted `Model_Pipeline.pkl` from your notebook. It predicts the room type from location and listing-level attributes."
    )
    a, b, c = st.columns(3)
    a.metric("Test accuracy", "85.66%", "From your notebook output")
    b.metric("Test macro F1", "74.06%", "From your notebook output")
    c.metric("Best CV macro F1", "73.32%", "3-fold CV during tuning")
    st.caption("These are the scores recorded in the uploaded notebook; they are not recalculated by the app.")
    st.markdown("#### Features used by the saved pipeline")
    feature_df = pd.DataFrame({
        "Feature": [
            "neighbourhood_group", "neighbourhood", "latitude", "longitude", "price",
            "minimum_nights", "number_of_reviews", "reviews_per_month",
            "calculated_host_listings_count", "availability_365"
        ],
        "Type": ["Categorical", "Categorical", "Numeric", "Numeric", "Numeric",
                 "Numeric", "Numeric", "Numeric", "Numeric", "Numeric"],
        "Purpose": [
            "NYC borough", "Neighborhood name", "Location", "Location", "Nightly price",
            "Minimum stay length", "Total review count", "Average reviews per month",
            "Host's listing count", "Days available in a year"
        ]
    })
    st.dataframe(feature_df, use_container_width=True, hide_index=True)
    st.markdown("#### Pipeline steps")
    st.markdown(
        """
        1. **Numeric features:** median imputation followed by standard scaling.
        2. **Categorical features:** most-frequent imputation followed by one-hot encoding.
        3. **Classifier:** tuned Random Forest with class weighting.
        4. **Output:** one of three room types, with per-class probabilities when supported.
        """
    )
    st.warning("Model predictions reflect patterns in the 2019 dataset. They should be treated as a learning/demo tool, not as a guarantee about an individual listing.")

st.markdown("---")
st.markdown(
    '<div class="small-note">STAYSENSE · NYC AIRBNB ROOM-TYPE CLASSIFICATION · For educational and portfolio use.</div>',
    unsafe_allow_html=True,
)
