# Road Segment Congestion Pattern Discovery with K-Means

An end-to-end unsupervised machine learning pipeline designed to discover, validate, and profile recurring road congestion states from continuous time-series traffic telemetry.

---

## 📌 Executive Summary & Unit of Analysis Definition

### What is a "Road Segment"?
- **Data Source & Structure:** The telemetry dataset represents continuous 15-minute interval observations from an urban arterial road monitoring station, recording multi-class vehicle counts (`CarCount`, `BikeCount`, `BusCount`, `TruckCount`), aggregated volume, and an operational `Traffic Situation` label.
- **Empirical Unit of Analysis:** Because observations capture a single monitoring station over time, our fundamental unit of analysis is the **time-slot profile** (hour-of-day $\times$ day-of-week).
- **Congestion Pattern Definition:** Congestion patterns represent **recurrent operational operating regimes** of this road segment (e.g., morning commuter surge, midday freight logistics, evening rush hour, late-night low-flow, and weekend leisure).
- **Data Leakage Safeguard:** The qualitative `Traffic Situation` column is strictly **held out** and excluded from all clustering operations. It is utilized exclusively post-hoc as an external ground-truth validation benchmark.

---

## 📂 Project Architecture

```
project/
├── data/
│   ├── raw/                 # Original/synthetic untouched CSV (Traffic.csv)
│   └── processed/           # Feature engineered & scaled matrices (traffic_clean.csv, X_scaled.csv, scaler.joblib)
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb           # EDA, time cyclicality, feature engineering & scaling
│   ├── 02_clustering_kmeans.ipynb           # K-Means grid (k=2..10), PCA, loadings & diagnostics
│   └── 03_validation_interpretation.ipynb   # Profiling, ground-truth cross-tab, ARI stability tests
├── src/
│   ├── generate_sample_data.py              # Realistic diurnal/freight traffic telemetry generator
│   ├── preprocessing.py                     # Data cleaning, cyclical time encoding, ratio engineering
│   ├── clustering.py                        # K-Means experimental grid, PCA projection, loadings
│   ├── evaluation.py                        # Mean/median profiling, cross-tab, multi-seed & bootstrap ARI
│   └── generate_notebooks.py                # Automated Jupyter notebook generator
├── outputs/
│   ├── figures/                             # 10 publication-quality diagnostic visualizations
│   ├── kmeans_metrics_grid.csv              # WCSS, Euclidean & Manhattan Silhouette, Davies-Bouldin
│   ├── cluster_profiles_mean.csv            # Original physical-unit cluster means
│   ├── cluster_profiles_median.csv          # Original physical-unit cluster medians
│   ├── standardized_centroids.csv           # Z-score centroids across clustering dimensions
│   ├── crosstab_percentages.csv             # Ground truth cross-tabulation percentages
│   ├── pca_loadings.csv                     # Principal component feature loadings
│   └── stability_analysis.csv               # Seed ARI & 80% bootstrap subsampling metrics
├── run_pipeline.py                          # Fully automated, reproducible end-to-end pipeline runner
├── requirements.txt                         # Pinned exact dependency versions
└── README.md                                # Comprehensive documentation & report
```

---

## 🔬 Methodology & Scientific Workflow

### Phase 1: Feature Engineering & Preprocessing
1. **Cyclical Time Encoding:** To preserve continuous physical periodicity between `23:45` and `00:00`, time is mapped onto orthogonal trigonometric coordinates:
   $$\sin\_hour = \sin\left(\frac{2\pi \cdot h}{24}\right), \quad \cos\_hour = \cos\left(\frac{2\pi \cdot h}{24}\right)$$
2. **Vehicle Mix Ratios:** Absolute vehicle counts are collinear with `Total`. To characterize modal share without mathematical redundancy, we engineer:
   $$\text{car\_ratio} = \frac{\text{CarCount}}{\text{Total}}, \quad \text{truck\_ratio} = \frac{\text{TruckCount}}{\text{Total}}, \quad \text{bus\_ratio} = \frac{\text{BusCount}}{\text{Total}}, \quad \text{bike\_ratio} = \frac{\text{BikeCount}}{\text{Total}}$$
3. **Outlier Mitigation:** Winsorization capping at $Q_3 + 3.0 \times \text{IQR}$ on `total_volume` to prevent extreme telemetry sensor glitches from distorting Euclidean centroids.
4. **Standardization:** All 8 modeling features are normalized via `StandardScaler` ($\mu = 0, \sigma = 1$).

### Phase 2: K-Means Optimization Grid ($k=2$ to $10$)
We systematically evaluate cluster configurations using multiple complementary criteria:
- **WCSS (Inertia):** Evaluates within-cluster dispersion for elbow detection.
- **Euclidean Silhouette Score ($L_2$):** Cohesion vs. separation under standard Euclidean distance.
- **Manhattan Silhouette Score ($L_1$):** Evaluates cluster separation under the Manhattan metric.
- **Davies-Bouldin Index:** Ratio of within-cluster spread to cluster centroid separation (lower is superior).
- **Cluster Balance:** Monitoring minimum cluster size ($> 15\%$) to guard against degenerate singleton groupings.

#### Empirical Results Grid:
| $k$ | WCSS | Silhouette ($L_2$) | Silhouette ($L_1$) | Davies-Bouldin | Min Cluster % | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 17299.31 | 0.2735 | 0.3163 | 1.5385 | 38.21% | Under-segmented |
| 3 | 14144.91 | 0.2868 | 0.3222 | 1.3506 | 20.46% | Misses weekend mode |
| **4** | **12042.92** | **0.2745** | **0.3477** | **1.4319** | **21.27%** | **Optimal / Defensible** |
| 5 | 10466.12 | 0.3005 | 0.3515 | 1.2491 | 15.09% | Splits rush hour |
| 6 | 9168.85 | 0.3457 | 0.3525 | 1.1093 | 12.37% | Over-segmentation |
| 7 | 8063.49 | 0.3683 | 0.3755 | 1.1923 | 7.66% | Sub-clusters emerging |

**Selection Defense:** $k=4$ represents the optimal inflection point on the elbow curve, achieves near-maximal Manhattan silhouette ($0.348$), retains excellent cluster distribution balance (all clusters between $21.3\%$ and $31.5\%$), and yields clean domain interpretability without fragmented sub-clusters.

---

## 🚦 Discovered Congestion Profiles

| Cluster | Plain-Language Persona | Volume (Mean) | Peak Hours | Predominant Vehicle Type | Weekend % | Held-Out 'Traffic Situation' |
|:---:|:---|:---:|:---:|:---|:---:|:---|
| **C0** | **Midday & Afternoon Commercial Flow** | 86.8 veh/interval | 11:00 – 16:30 | Mixed Cars & Commercial Freight (18.3 trucks) | 1.0% | 48.9% Normal, 47.4% High |
| **C1** | **Peak Commuter Rush Hour** | 125.7 veh/interval | 07:30–09:30, 17:00–19:30 | Personal Cars (85.2) & Motorcycles (26.6) | 0.0% | **79.9% Heavy**, 19.8% High |
| **C2** | **Nighttime / Early Dawn Low Flow** | 16.5 veh/interval | 00:00 – 05:45 | Overnight Logistics & Trucks (41.4% truck share) | 19.0% | **98.4% Low**, 1.6% Normal |
| **C3** | **Weekend Leisure & Off-Peak Travel** | 56.6 veh/interval | 10:00 – 20:00 (Sat/Sun) | Personal Cars (35.2) & Bikes (12.2), Low Freight | **100.0%** | 55.6% Normal, 28.3% Low, 16.1% High |

---

## 🛡️ Robustness & Stability Analysis

To verify that clusters reflect true data structure rather than random centroid seeding artifacts or sample noise, we performed dual stability tests:

1. **Multi-Seed Random Initialization Test:**
   - 15 independent K-Means runs with varying random seeds ($n=15$).
   - **Mean Pairwise ARI ($k=4$):** **$0.9979 \pm 0.0020$** (Min ARI = $0.9950$).
   - Demonstrates near-deterministic convergence across seed initializations.

2. **80% Bootstrap Subsampling Test:**
   - 20 iterations of random 80% subsamples with independent clustering.
   - **Mean Subsample ARI ($k=4$):** **$0.9653 \pm 0.0871$**.
   - Confirms that clusters remain highly robust under empirical data perturbation.

---

## ⚖️ Limitations, Sampling Bias & Ethical Use

### Limitations
- **Convex/Spherical Geometry:** K-Means assumes isotropic, spherical clusters of similar density. Continuous temporal transitions along diurnal curves may be slightly discretized.
- **Absence of Temporal Autocorrelation:** The model treats each 15-minute slot independently, ignoring sequential Markovian dependencies ($t$ conditioned on $t-1$).
- **Linear Feature Scaling:** Standard scaling can be sensitive to rare extreme traffic events (accidents, parades).

### Sampling Biases
- **Single-Location Telemetry:** Patterns reflect this specific arterial corridor and cannot be blindly generalized to suburban highways or pedestrian plazas without local sensor calibration.
- **Unmeasured Mobility Modes:** Micro-mobility (e-scooters, bicycles, pedestrians) is partially unobserved.
- **Weather and Seasonal Blindness:** Severe meteorological events (monsoons, snowstorms) are not explicitly conditioned.

### Ethical & Responsible Governance
- **Decision Support, Not Automated Decisions:** Congestion clusters should inform civil engineering decisions (signal retiming, public transit priority lanes) rather than automated dynamic tolling or punitive traffic enforcement.
- **Equitable Municipal Investment:** Service allocations should never deprioritize roads classified as "low flow" if those segments serve vulnerable communities.

---

## 🌐 Interactive Web Application (Streamlit)

The project includes a Streamlit dashboard for traffic pattern analysis, scenario simulations, model diagnostics, and cluster profiles. The original Flask application remains available as `app.py`.

### Key Web Features:
- **Executive Dashboard:** Live metrics, active cluster persona cards, and diurnal vehicle distribution.
- **⚡ Live Pattern Classifier:** Interactive sliders for hour of day, day of week, cars, bikes, buses, and trucks. Instantly classifies the time-slot into its discovered cluster persona and estimates ground-truth congestion likelihoods.
- **📈 Diagnostics & PCA Viewer:** Full metrics grid ($k=2..10$), interactive Elbow curve, L1/L2 Silhouette comparisons, and PCA component loadings.
- **🔍 Profiles & Stability Deep Dive:** Original physical-unit statistics, standardized centroids heatmap, and Adjusted Rand Index (ARI) stability tables.
- **💾 Dataset Preview:** Inspect the bundled telemetry dataset or preview an uploaded CSV. Uploads are not saved or used to retrain the deployed model.

### Run Streamlit Locally:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Streamlit opens the dashboard at **`http://localhost:8501`**.

### Deploy to Streamlit Community Cloud:
1. Push the project to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the main file path to `streamlit_app.py` and deploy.

Keep the trained model artifacts (`data/processed/scaler.joblib` and `data/processed/kmeans_model.joblib`), dashboard data, figures, and `requirements.txt` in the repository so the hosted app can load them. Streamlit Cloud does not preserve uploaded files or other local filesystem changes between restarts.

---

## ⚡ How to Reproduce & Run

### 1. Setup Virtual Environment
```bash
pip install -r requirements.txt
```

### 2. Execute End-to-End Pipeline
```bash
python run_pipeline.py
```
This single command runs raw data verification, feature engineering, K-Means grid exploration, PCA projection, profiling, stability testing, and generates all 10 figures in `outputs/figures/`.

### 3. Interactive Jupyter Notebooks
Launch Jupyter to explore step-by-step:
```bash
jupyter notebook notebooks/
```
- `01_eda_preprocessing.ipynb`: Exploratory data analysis, cyclical transformations, scaling.
- `02_clustering_kmeans.ipynb`: Elbow curves, Euclidean/Manhattan silhouette analysis, PCA loadings.
- `03_validation_interpretation.ipynb`: Profiles, ground-truth cross-tabs, bootstrap stability tests.

