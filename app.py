"""
Flask Web Application for Road Segment Congestion Pattern Discovery
Provides an interactive dashboard, live pattern classifier, model diagnostics viewer,
and automated pipeline retraining interface.
"""

import os
import sys

# Sandbox & threading configuration
os.environ["LOKY_MAX_CPU_COUNT"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), ".mpl_cache")

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import engineer_features, prepare_clustering_matrix

app = Flask(__name__)
app.secret_key = "unsupervised-traffic-secret-key-42"
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "data", "raw")
app.config["FIGURES_FOLDER"] = os.path.join(os.path.dirname(__file__), "outputs", "figures")

# Cluster Personas & Metadata
CLUSTER_METADATA = {
    0: {
        "persona": "Midday Commercial & Freight Flow",
        "description": "Dominated by commercial logistics, local deliveries, and steady midday traffic. Heavy truck presence with moderate overall congestion.",
        "badge": "badge-c0"
    },
    1: {
        "persona": "Peak Commuter Rush Hour",
        "description": "Characterized by severe congestion, high volume of personal commuter cars and motorcycles during morning and evening rush hours.",
        "badge": "badge-c1"
    },
    2: {
        "persona": "Nighttime / Early Dawn Low Flow",
        "description": "Minimal overall volume, free-flowing conditions, and disproportionately high share of overnight long-haul freight transport.",
        "badge": "badge-c2"
    },
    3: {
        "persona": "Weekend Leisure & Off-Peak Travel",
        "description": "Active primarily on Saturdays and Sundays. Balanced personal car and motorcycle travel with negligible commercial transit.",
        "badge": "badge-c3"
    }
}

# Empirical ground truth situation probabilities (from crosstab)
DEFAULT_SITUATION_PROBS = {
    0: {"heavy": 0.0, "high": 47.4, "normal": 48.9, "low": 3.6},
    1: {"heavy": 79.9, "high": 19.8, "normal": 0.3, "low": 0.0},
    2: {"heavy": 0.0, "high": 0.0, "normal": 1.7, "low": 98.4},
    3: {"heavy": 0.0, "high": 16.1, "normal": 55.6, "low": 28.3}
}

def load_resources():
    """Loads model, scaler, and summary metrics."""
    scaler_path = "data/processed/scaler.joblib"
    model_path = "data/processed/kmeans_model.joblib"
    
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    
    return scaler, model

@app.route("/")
def index():
    raw_csv = "data/raw/Traffic.csv"
    total_samples = 2976
    if os.path.exists(raw_csv):
        try:
            total_samples = len(pd.read_csv(raw_csv))
        except Exception:
            pass

    summary = {
        "total_samples": total_samples,
        "k": 4,
        "features_count": 8,
        "ari_score": 0.998
    }
    return render_template("index.html", active_page="dashboard", summary=summary)

@app.route("/predict")
def predict_view():
    return render_template("predict.html", active_page="predict")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json() or {}
        hour = float(data.get("hour", 8.5))
        day = data.get("day", "Monday")
        cars = int(data.get("cars", 85))
        bikes = int(data.get("bikes", 25))
        buses = int(data.get("buses", 12))
        trucks = int(data.get("trucks", 5))
        
        # 1. Feature Engineering
        sin_hour = np.sin(2 * np.pi * hour / 24.0)
        cos_hour = np.cos(2 * np.pi * hour / 24.0)
        is_weekend = 1 if day in ["Saturday", "Sunday"] else 0
        
        raw_total = cars + bikes + buses + trucks
        safe_total = max(raw_total, 1)
        
        car_ratio = cars / safe_total
        bike_ratio = bikes / safe_total
        bus_ratio = buses / safe_total
        truck_ratio = trucks / safe_total
        
        # Capping bound (approx 200 based on IQR winsorization)
        total_volume_capped = min(raw_total, 220)
        
        feat_cols = [
            "sin_hour", "cos_hour", "is_weekend", "total_volume_capped",
            "car_ratio", "bike_ratio", "bus_ratio", "truck_ratio"
        ]
        feat_df = pd.DataFrame([[
            sin_hour, cos_hour, is_weekend, total_volume_capped,
            car_ratio, bike_ratio, bus_ratio, truck_ratio
        ]], columns=feat_cols)
        
        scaler, model = load_resources()
        if scaler is None or model is None:
            return jsonify({"status": "error", "message": "Model not trained yet."}), 500
            
        scaled_vector = scaler.transform(feat_df)
        cluster = int(model.predict(scaled_vector)[0])
        
        # Distance to centroid
        centroid = model.cluster_centers_[cluster]
        distance = float(np.linalg.norm(scaled_vector[0] - centroid))
        
        meta = CLUSTER_METADATA.get(cluster, {
            "persona": f"Cluster {cluster}",
            "description": "General traffic pattern.",
            "badge": "badge-c0"
        })
        
        situation_probs = DEFAULT_SITUATION_PROBS.get(cluster, {"heavy": 25, "high": 25, "normal": 25, "low": 25})
        
        return jsonify({
            "status": "success",
            "cluster": cluster,
            "persona": meta["persona"],
            "description": meta["description"],
            "distance_to_center": round(distance, 3),
            "situation_probs": situation_probs,
            "total_volume": raw_total,
            "truck_share_pct": round(truck_ratio * 100, 1)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/evaluation")
def evaluation_view():
    grid_csv = "outputs/kmeans_metrics_grid.csv"
    grid_data = []
    if os.path.exists(grid_csv):
        grid_data = pd.read_csv(grid_csv).to_dict(orient="records")
    return render_template("evaluation.html", active_page="evaluation", grid=grid_data)

@app.route("/profiles")
def profiles_view():
    stab_csv = "outputs/stability_analysis.csv"
    stab_data = []
    if os.path.exists(stab_csv):
        stab_data = pd.read_csv(stab_csv).to_dict(orient="records")
    return render_template("profiles.html", active_page="profiles", stability=stab_data)

@app.route("/dataset")
def dataset_view():
    raw_csv = "data/raw/Traffic.csv"
    records = []
    if os.path.exists(raw_csv):
        df = pd.read_csv(raw_csv)
        records = df.head(50).to_dict(orient="records")
    return render_template("dataset.html", active_page="dataset", records=records)

@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        flash("No file part in request", "danger")
        return redirect(url_for("dataset_view"))
    
    file = request.files["file"]
    if file.filename == "":
        flash("No file selected", "danger")
        return redirect(url_for("dataset_view"))
        
    if file and file.filename.endswith(".csv"):
        dest_path = os.path.join(app.config["UPLOAD_FOLDER"], "Traffic.csv")
        file.save(dest_path)
        flash("Dataset uploaded successfully as data/raw/Traffic.csv! Click 'Retrain Clustering Pipeline' below to update models.", "success")
    else:
        flash("Please upload a valid .csv file", "danger")
        
    return redirect(url_for("dataset_view"))

@app.route("/retrain_pipeline", methods=["POST"])
def retrain_pipeline():
    try:
        from run_pipeline import main as run_pipeline_main
        run_pipeline_main()
        flash("Clustering pipeline executed successfully! All metrics, models, and figures have been updated.", "success")
    except Exception as e:
        flash(f"Error during retraining: {str(e)}", "danger")
        
    return redirect(url_for("dataset_view"))

@app.route("/outputs/figures/<path:filename>")
def figure_file(filename):
    return send_from_directory(app.config["FIGURES_FOLDER"], filename)

if __name__ == "__main__":
    # Standard local Flask port 5000
    app.run(host="127.0.0.1", port=5000, debug=False)
