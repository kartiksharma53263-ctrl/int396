"""Streamlit dashboard for road-segment congestion pattern discovery."""

import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mpl_cache"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "outputs" / "figures"
RAW_DATA = ROOT / "data" / "raw" / "Traffic.csv"
SCALER_PATH = ROOT / "data" / "processed" / "scaler.joblib"
MODEL_PATH = ROOT / "data" / "processed" / "kmeans_model.joblib"

CLUSTERS = {
    0: {
        "persona": "Midday Commercial & Freight Flow",
        "description": "Steady midday traffic with a substantial commercial and freight presence.",
    },
    1: {
        "persona": "Peak Commuter Rush Hour",
        "description": "High-volume commuter traffic, especially personal cars and motorcycles.",
    },
    2: {
        "persona": "Nighttime / Early Dawn Low Flow",
        "description": "Low overall volume with a comparatively high share of overnight freight.",
    },
    3: {
        "persona": "Weekend Leisure & Off-Peak Travel",
        "description": "Weekend-focused personal travel with relatively little commercial transit.",
    },
}

SITUATION_PROBS = {
    0: {"Heavy": 0.0, "High": 47.4, "Normal": 48.9, "Low": 3.6},
    1: {"Heavy": 79.9, "High": 19.8, "Normal": 0.3, "Low": 0.0},
    2: {"Heavy": 0.0, "High": 0.0, "Normal": 1.7, "Low": 98.4},
    3: {"Heavy": 0.0, "High": 16.1, "Normal": 55.6, "Low": 28.3},
}

FIGURE_TITLES = {
    "01_diurnal_hourly_pattern.png": "Mean vehicle composition by hour",
    "03_elbow_curve.png": "K-Means elbow curve",
    "04_silhouette_and_db_comparison.png": "Cluster quality metrics",
    "05_pca_2d_clusters.png": "PCA cluster projection",
    "06_pca_loadings.png": "PCA feature loadings",
    "08_standardized_centroids_heatmap.png": "Standardized cluster centroids",
    "09_traffic_situation_by_cluster.png": "Traffic situation by cluster",
    "10_stability_ari_comparison.png": "Cluster stability",
}


@st.cache_resource
def load_model_resources():
    """Load the model artifacts once per Streamlit process."""
    if not SCALER_PATH.is_file() or not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "The trained model files are missing. Include "
            "data/processed/scaler.joblib and data/processed/kmeans_model.joblib."
        )
    return joblib.load(SCALER_PATH), joblib.load(MODEL_PATH)


@st.cache_data
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def show_figure(filename: str, caption: str = "") -> None:
    path = FIGURES / filename
    if path.is_file():
        st.image(str(path), caption=caption or FIGURE_TITLES.get(filename, ""))
    else:
        st.info(f"Figure not available: {filename}")


def make_prediction(hour: float, day: str, cars: int, bikes: int, buses: int, trucks: int):
    raw_total = cars + bikes + buses + trucks
    safe_total = max(raw_total, 1)
    features = pd.DataFrame(
        [[
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            int(day in ("Saturday", "Sunday")),
            min(raw_total, 220),
            cars / safe_total,
            bikes / safe_total,
            buses / safe_total,
            trucks / safe_total,
        ]],
        columns=[
            "sin_hour",
            "cos_hour",
            "is_weekend",
            "total_volume_capped",
            "car_ratio",
            "bike_ratio",
            "bus_ratio",
            "truck_ratio",
        ],
    )
    scaler, model = load_model_resources()
    scaled = scaler.transform(features)
    cluster = int(model.predict(scaled)[0])
    distance = float(np.linalg.norm(scaled[0] - model.cluster_centers_[cluster]))
    return cluster, distance, raw_total


def dashboard_page() -> None:
    st.title("Road Segment Congestion Pattern Discovery")
    st.caption("Discovering operational traffic regimes from 15-minute vehicle telemetry")

    sample_count = 2976
    if RAW_DATA.is_file():
        sample_count = len(read_csv(str(RAW_DATA)))

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Observations", f"{sample_count:,}")
    kpi2.metric("Selected clusters (k)", "4")
    kpi3.metric("Seed stability (ARI)", "0.9979")
    kpi4.metric("Unit of analysis", "Time-slot")

    st.subheader("Discovered traffic operating regimes")
    profiles_path = ROOT / "outputs" / "cluster_profiles_mean.csv"
    if profiles_path.is_file():
        profiles = read_csv(str(profiles_path))
        cols = st.columns(2)
        for index, row in profiles.iterrows():
            cluster = int(row["cluster"])
            with cols[index % 2]:
                st.markdown(f"**Cluster {cluster} — {CLUSTERS[cluster]['persona']}**")
                st.write(
                    f"{row['percentage']:.1f}% of observations · "
                    f"{row['total_volume']:.1f} vehicles per interval on average"
                )
                st.caption(CLUSTERS[cluster]["description"])
    else:
        st.warning("Cluster profile data is not available.")

    left, right = st.columns(2)
    with left:
        show_figure("01_diurnal_hourly_pattern.png")
    with right:
        show_figure("05_pca_2d_clusters.png")


def classifier_page() -> None:
    st.title("Live Traffic Pattern Classifier")
    st.caption("Simulate a 15-minute traffic interval and classify its operating regime.")

    with st.form("classifier"):
        hour = st.slider("Hour of day", 0.0, 23.75, 8.5, 0.25)
        day = st.selectbox(
            "Day of the week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        )
        columns = st.columns(4)
        cars = columns[0].number_input("Cars", min_value=0, max_value=250, value=85)
        bikes = columns[1].number_input("Bikes", min_value=0, max_value=100, value=25)
        buses = columns[2].number_input("Buses", min_value=0, max_value=50, value=12)
        trucks = columns[3].number_input("Trucks", min_value=0, max_value=60, value=5)
        submitted = st.form_submit_button("Classify traffic pattern", type="primary")

    if not submitted:
        return
    try:
        cluster, distance, total = make_prediction(
            hour, day, int(cars), int(bikes), int(buses), int(trucks)
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(f"Could not classify this scenario: {error}")
        return

    metadata = CLUSTERS.get(
        cluster,
        {"persona": f"Cluster {cluster}", "description": "General traffic pattern."},
    )
    st.subheader(f"Cluster {cluster}: {metadata['persona']}")
    st.write(metadata["description"])
    result1, result2, result3 = st.columns(3)
    result1.metric("Total volume", f"{total} vehicles")
    result2.metric("Truck share", f"{trucks / max(total, 1) * 100:.1f}%")
    result3.metric("Distance to centroid", f"{distance:.3f}")
    st.caption("Congestion likelihoods are empirical historical proportions for this cluster.")
    probabilities = SITUATION_PROBS.get(cluster, {})
    st.bar_chart(pd.Series(probabilities, name="Historical share (%)"))


def diagnostics_page() -> None:
    st.title("Model Diagnostics")
    metrics_path = ROOT / "outputs" / "kmeans_metrics_grid.csv"
    if metrics_path.is_file():
        metrics = read_csv(str(metrics_path))
        st.subheader("K-Means evaluation grid")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
    else:
        st.warning("Evaluation metrics are not available.")
    show_figure("03_elbow_curve.png")
    show_figure("04_silhouette_and_db_comparison.png")


def profiles_page() -> None:
    st.title("Cluster Profiles & Stability")
    profiles_path = ROOT / "outputs" / "cluster_profiles_mean.csv"
    stability_path = ROOT / "outputs" / "stability_analysis.csv"
    if profiles_path.is_file():
        st.subheader("Mean physical-unit profiles")
        st.dataframe(read_csv(str(profiles_path)), use_container_width=True, hide_index=True)
    else:
        st.warning("Cluster profile data is not available.")

    left, right = st.columns(2)
    with left:
        show_figure("08_standardized_centroids_heatmap.png")
    with right:
        show_figure("09_traffic_situation_by_cluster.png")

    if stability_path.is_file():
        st.subheader("Adjusted Rand Index (ARI) stability")
        st.dataframe(read_csv(str(stability_path)), use_container_width=True, hide_index=True)
    show_figure("10_stability_ari_comparison.png")


def dataset_page() -> None:
    st.title("Dataset")
    if RAW_DATA.is_file():
        dataset = read_csv(str(RAW_DATA))
        st.write(f"Dataset contains **{len(dataset):,} rows** and **{len(dataset.columns)} columns**.")
        st.dataframe(dataset.head(50), use_container_width=True, hide_index=True)
    else:
        st.warning("The bundled traffic dataset is not available.")

    uploaded_file = st.file_uploader("Preview another CSV", type="csv")
    if uploaded_file is not None:
        try:
            uploaded = pd.read_csv(uploaded_file)
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            st.error(f"Could not read the uploaded CSV: {error}")
        else:
            st.write(f"Uploaded file contains **{len(uploaded):,} rows**.")
            st.dataframe(uploaded.head(50), use_container_width=True, hide_index=True)
            st.caption("Uploads are preview-only and are not saved or used to retrain the deployed model.")


def main() -> None:
    st.set_page_config(
        page_title="Traffic Pattern Discovery",
        page_icon="🚦",
        layout="wide",
    )
    page = st.sidebar.radio(
        "Navigate",
        ["Dashboard", "Classifier", "Diagnostics", "Profiles & Stability", "Dataset"],
    )
    pages = {
        "Dashboard": dashboard_page,
        "Classifier": classifier_page,
        "Diagnostics": diagnostics_page,
        "Profiles & Stability": profiles_page,
        "Dataset": dataset_page,
    }
    pages[page]()


if __name__ == "__main__":
    main()
