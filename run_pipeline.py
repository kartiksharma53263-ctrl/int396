

"""
End-to-End Execution Pipeline
Road Segment Congestion Pattern Discovery with K-Means
Executes full preprocessing, modeling grid, visualization export,
profiling, and stability testing.
"""

import os
import sys

# Prevent joblib/loky subprocess calls for CPU core detection in sandbox
os.environ["LOKY_MAX_CPU_COUNT"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

# Configure UTF-8 encoding for standard outputs on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Set Matplotlib temp directory to avoid permissions warnings
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), ".mpl_cache")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import load_and_inspect_raw_data, engineer_features, prepare_clustering_matrix
from clustering import run_kmeans_experiments, compute_pca_projection
from evaluation import (
    generate_cluster_profiles,
    cross_tabulate_traffic_situation,
    test_random_seed_stability,
    test_subsampling_stability
)

RANDOM_STATE = 42
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

def main():
    print("=================================================================")
    print("ROAD SEGMENT CONGESTION PATTERN DISCOVERY PIPELINE")
    print("=================================================================")
    
    # 0. Output directory setups
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # 1. Load Data
    raw_csv = "data/raw/Traffic.csv"
    if not os.path.exists(raw_csv):
        print(f"[WARN] {raw_csv} not found! Generating realistic dataset...")
        from generate_sample_data import generate_traffic_data
        generate_traffic_data(output_path=raw_csv)
        
    df_raw, inspect_info = load_and_inspect_raw_data(raw_csv)
    print(f"[OK] Loaded raw data: {inspect_info['shape'][0]} rows, {inspect_info['shape'][1]} columns")
    print(f"   Missing values detected: {sum(inspect_info['missing_counts'].values())}")
    print(f"   Duplicate rows detected: {inspect_info['duplicate_rows']}")
    
    # 2. Preprocessing & Feature Engineering
    print("\n--- Phase 1: Feature Engineering & Preprocessing ---")
    df_eng = engineer_features(df_raw)
    X_scaled_df, feature_cols, held_out_labels, scaler = prepare_clustering_matrix(df_eng)
    
    # Save processed clean dataframe
    processed_df = df_eng.copy()
    processed_df.to_csv("data/processed/traffic_clean.csv", index=False)
    X_scaled_df.to_csv("data/processed/X_scaled.csv", index=False)
    print(f"[OK] Preprocessed matrix prepared: {X_scaled_df.shape[0]} samples x {X_scaled_df.shape[1]} features")
    print(f"   Clustering features: {feature_cols}")
    print("   Held-out label preserved: 'Traffic Situation'")
    
    # EDA Visualizations
    # Figure 1: Diurnal hourly vehicle distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    hourly_avg = df_eng.groupby(np.floor(df_eng["hour_float"]).astype(int))[["CarCount", "BikeCount", "BusCount", "TruckCount"]].mean()
    hourly_avg.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_title("Figure 1: Mean Vehicle Composition by Hour of Day", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour of Day (0-23)")
    ax.set_ylabel("Average Count per 15-Minute Interval")
    ax.legend(["Cars", "Bikes/Motorcycles", "Buses", "Trucks"])
    plt.tight_layout()
    plt.savefig("outputs/figures/01_diurnal_hourly_pattern.png", dpi=300)
    plt.close()
    
    # Figure 2: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = X_scaled_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title("Figure 2: Feature Correlation Heatmap (Selected Features)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/figures/02_correlation_heatmap.png", dpi=300)
    plt.close()
    
    # 3. K-Means Grid Experiments (k = 2 to 10)
    print("\n--- Phase 2: K-Means Experiments (k=2 to 10) ---")
    results_df, models = run_kmeans_experiments(X_scaled_df.values, k_range=range(2, 11), n_init=25)
    results_df.to_csv("outputs/kmeans_metrics_grid.csv", index=False)
    print(results_df.to_string(index=False))
    
    # Figure 3: Elbow Plot (WCSS vs k)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df["k"], results_df["wcss"], marker="o", color="#1f77b4", linewidth=2.5, markersize=7)
    ax.set_title("Figure 3: K-Means Elbow Curve (Inertia / WCSS vs. k)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
    ax.set_xticks(range(2, 11))
    ax.axvline(x=4, color="crimson", linestyle="--", alpha=0.8, label="Selected k = 4")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/03_elbow_curve.png", dpi=300)
    plt.close()
    
    # Figure 4: Silhouette (Euclidean vs Manhattan) & Davies-Bouldin
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(results_df["k"], results_df["silhouette_euclidean"], marker="o", color="#2ca02c", label="Silhouette (Euclidean)", linewidth=2)
    ax1.plot(results_df["k"], results_df["silhouette_manhattan"], marker="s", color="#ff7f0e", linestyle="--", label="Silhouette (Manhattan)", linewidth=2)
    ax1.set_title("Silhouette Scores vs. k (Higher is Better)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Silhouette Score")
    ax1.set_xticks(range(2, 11))
    ax1.axvline(x=4, color="crimson", linestyle=":", alpha=0.7)
    ax1.legend()
    
    ax2.plot(results_df["k"], results_df["davies_bouldin"], marker="^", color="#d62728", linewidth=2)
    ax2.set_title("Davies-Bouldin Index vs. k (Lower is Better)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Davies-Bouldin Score")
    ax2.set_xticks(range(2, 11))
    ax2.axvline(x=4, color="crimson", linestyle=":", alpha=0.7, label="Selected k = 4")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("outputs/figures/04_silhouette_and_db_comparison.png", dpi=300)
    plt.close()
    
    # 4. Chosen Model Evaluation & PCA Projection
    chosen_k = 4
    print(f"\n--- Defensible Model Choice: k = {chosen_k} ---")
    best_model = models[chosen_k]
    labels = best_model.labels_
    df_eng["cluster"] = labels
    
    # Save the trained optimal KMeans model
    import joblib
    joblib.dump(best_model, "data/processed/kmeans_model.joblib")
    print("   Saved trained KMeans model to data/processed/kmeans_model.joblib")
    
    # PCA
    X_pca, exp_var, loadings, pca_obj = compute_pca_projection(X_scaled_df.values, feature_cols, n_components=2)
    loadings.to_csv("outputs/pca_loadings.csv")
    print(f"PCA Explained Variance: PC1 = {exp_var[0]*100:.2f}%, PC2 = {exp_var[1]*100:.2f}% (Total = {sum(exp_var)*100:.2f}%)")
    
    # Figure 5: PCA 2D Scatter colored by Cluster
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.65, edgecolors="none", s=25)
    # Plot PCA transformed cluster centers
    centers_pca = pca_obj.transform(best_model.cluster_centers_)
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c="black", marker="X", s=180, edgecolors="white", linewidth=1.5, label="Cluster Centroids")
    ax.set_title(f"Figure 5: PCA 2D Cluster Projection (k={chosen_k})", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({exp_var[0]*100:.1f}% explained variance)")
    ax.set_ylabel(f"PC2 ({exp_var[1]*100:.1f}% explained variance)")
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(chosen_k))
    cbar.set_label("Cluster Assignment")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("outputs/figures/05_pca_2d_clusters.png", dpi=300)
    plt.close()
    
    # Figure 6: PCA Loadings Heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(loadings, annot=True, fmt=".3f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Figure 6: PCA Component Loadings", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/figures/06_pca_loadings.png", dpi=300)
    plt.close()
    
    # Figure 7: Silhouette Per-Sample Analysis for k=4
    sample_sils = silhouette_samples(X_scaled_df.values, labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    colors = plt.cm.tab10(np.linspace(0, 1, chosen_k))
    for i in range(chosen_k):
        ith_cluster_sil = sample_sils[labels == i]
        ith_cluster_sil.sort()
        size_cluster_i = ith_cluster_sil.shape[0]
        y_upper = y_lower + size_cluster_i
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil, facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f"C{i}")
        y_lower = y_upper + 10
    
    mean_sil = results_df.loc[results_df["k"] == chosen_k, "silhouette_euclidean"].values[0]
    ax.axvline(x=mean_sil, color="red", linestyle="--", label=f"Average Silhouette ({mean_sil:.3f})")
    ax.set_title(f"Figure 7: Silhouette Analysis per Sample (k={chosen_k})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster Sample Index")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("outputs/figures/07_silhouette_samples.png", dpi=300)
    plt.close()
    
    # 5. Cluster Profiling and Interpretation
    print("\n--- Phase 3: Cluster Profiling & Ground Truth Cross-Tabulation ---")
    profile_mean, profile_median = generate_cluster_profiles(df_eng, labels)
    profile_mean.to_csv("outputs/cluster_profiles_mean.csv")
    profile_median.to_csv("outputs/cluster_profiles_median.csv")
    print("\nCluster Mean Profiles (Original Units):")
    print(profile_mean[["sample_count", "percentage", "total_volume", "CarCount", "TruckCount", "hour_float", "is_weekend"]])
    
    # Standardized Centroids Heatmap
    centroids_df = pd.DataFrame(best_model.cluster_centers_, columns=feature_cols, index=[f"Cluster {i}" for i in range(chosen_k)])
    centroids_df.to_csv("outputs/standardized_centroids.csv")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(centroids_df, annot=True, fmt=".2f", cmap="Spectral", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Figure 8: Standardized Feature Centroids Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/figures/08_standardized_centroids_heatmap.png", dpi=300)
    plt.close()
    
    # Cross-tabulation with held-out Traffic Situation
    if held_out_labels is not None:
        counts_ct, pct_ct = cross_tabulate_traffic_situation(held_out_labels, labels)
        counts_ct.to_csv("outputs/crosstab_counts.csv")
        pct_ct.to_csv("outputs/crosstab_percentages.csv")
        print("\nCross-Tabulation with Held-Out 'Traffic Situation' (%):")
        print(pct_ct)
        
        # Figure 9: Stacked bar chart of Traffic Situation by Cluster
        fig, ax = plt.subplots(figsize=(9, 5))
        # Ensure ordered columns
        order = [c for c in ["low", "normal", "high", "heavy"] if c in pct_ct.columns]
        pct_ct[order].plot(kind="bar", stacked=True, ax=ax, colormap="RdYlGn_r", edgecolor="black")
        ax.set_title("Figure 9: Held-Out 'Traffic Situation' Composition per Cluster", fontsize=13, fontweight="bold")
        ax.set_xlabel("Discovered Cluster")
        ax.set_ylabel("Percentage (%)")
        ax.legend(title="Traffic Situation", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig("outputs/figures/09_traffic_situation_by_cluster.png", dpi=300)
        plt.close()
        
    # 6. Stability Analysis (Multi-seed & Subsampling)
    print("\n--- Phase 4: Stability Analysis (Multi-seed & Subsampling) ---")
    stability_records = []
    
    for test_k in [3, 4, 5, 6, 7]:
        seed_stats, _ = test_random_seed_stability(X_scaled_df.values, k=test_k, num_seeds=15)
        sub_stats, _ = test_subsampling_stability(X_scaled_df.values, k=test_k, sample_ratio=0.8, n_iterations=20)
        
        stability_records.append({
            "k": test_k,
            "seed_mean_ari": seed_stats["mean_ari"],
            "seed_std_ari": seed_stats["std_ari"],
            "seed_min_ari": seed_stats["min_ari"],
            "bootstrap_mean_ari": sub_stats["mean_subsample_ari"],
            "bootstrap_std_ari": sub_stats["std_subsample_ari"],
            "bootstrap_min_ari": sub_stats["min_subsample_ari"]
        })
        
    stab_df = pd.DataFrame(stability_records)
    stab_df.to_csv("outputs/stability_analysis.csv", index=False)
    print(stab_df.to_string(index=False))
    
    # Figure 10: Stability Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(stab_df["k"], stab_df["seed_mean_ari"], yerr=stab_df["seed_std_ari"], fmt="o-", capsize=5, label="Random Seed Stability (Pairwise ARI)", color="#1f77b4", linewidth=2)
    ax.errorbar(stab_df["k"], stab_df["bootstrap_mean_ari"], yerr=stab_df["bootstrap_std_ari"], fmt="s--", capsize=5, label="80% Subsample Stability (ARI)", color="#ff7f0e", linewidth=2)
    ax.set_title("Figure 10: Cluster Stability Across Varying k (Adjusted Rand Index)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Mean ARI (+/- 1 Std Dev)")
    ax.set_ylim(0.4, 1.05)
    ax.axvline(x=4, color="crimson", linestyle=":", label="Selected k = 4")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("outputs/figures/10_stability_ari_comparison.png", dpi=300)
    plt.close()
    
    print("\n=================================================================")
    print("[SUCCESS] PIPELINE RUN COMPLETE! All figures and tables exported to:")
    print("   - outputs/figures/ (10 figures generated)")
    print("   - outputs/ (metrics tables, profiles, crosstab, loadings)")
    print("   - data/processed/ (scaled matrices and clean data)")
    print("=================================================================")

if __name__ == "__main__":
    main()
