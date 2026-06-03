import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================
# EXPERIMENT 8: Fault Detection using K-Means Clustering
# Subject: AI in Mechanical Engineering (ONT406)
# Sharda University
# ============================================================

class DatasetError(ValueError):
    """Raised for invalid sensor dataset parameters."""
    pass

class ClusteringError(RuntimeError):
    """Raised when K-Means clustering fails."""
    pass

class PlotError(RuntimeError):
    """Raised when visualization fails."""
    pass

# -------------------------------------------------------
# Input Helpers
# -------------------------------------------------------
def get_positive_float(prompt):
    while True:
        try:
            value = float(input(prompt))
        except ValueError:
            print("  Error: Enter a numeric value.")
            continue
        if value <= 0:
            print("  Error: Value must be greater than zero.")
            continue
        return value

def get_positive_int(prompt, minimum=1):
    while True:
        try:
            value = int(input(prompt))
        except ValueError:
            print("  Error: Enter a whole number.")
            continue
        if value < minimum:
            print(f"  Error: Value must be at least {minimum}.")
            continue
        return value

def get_range(label):
    """Get validated min-max range (both positive)."""
    while True:
        try:
            low  = float(input(f"  {label} min: "))
            high = float(input(f"  {label} max: "))
        except ValueError:
            print("  Error: Enter numeric values.")
            continue
        if low <= 0:
            print("  Error: Minimum must be positive.")
            continue
        if high <= low:
            print("  Error: Maximum must be greater than minimum.")
            continue
        return low, high

def get_float_or_nan(prompt):
    while True:
        raw = input(prompt).strip().lower()
        if raw in ['nan', 'n', 'missing', '']:
            return np.nan
        try:
            return float(raw)
        except ValueError:
            print("  Error: Enter a number or 'nan'.")

# -------------------------------------------------------
# Dataset Builders
# -------------------------------------------------------
def preset_dataset():
    np.random.seed(0)
    return pd.DataFrame({
        "Vibration":   np.random.uniform(0.5, 5.0, 150),
        "Temperature": np.random.uniform(30, 120, 150),
    })

def custom_dataset(n, vib_r, temp_r):
    if n < 6:
        raise DatasetError("Need at least 6 samples.")
    for label, (lo, hi) in [("Vibration", vib_r), ("Temperature", temp_r)]:
        if hi <= lo:
            raise DatasetError(f"{label}: max must be greater than min.")
    np.random.seed(0)
    return pd.DataFrame({
        "Vibration":   np.random.uniform(*vib_r,  n),
        "Temperature": np.random.uniform(*temp_r, n),
    })

def manual_dataset():
    n = get_positive_int("  Number of sensor readings (min 6): ", minimum=6)
    vibs, temps = [], []
    print(f"\n  Enter {n} sensor readings:")
    for i in range(n):
        print(f"\n  Reading {i+1}:")
        v = get_positive_float("    Vibration (g)    : ")
        t = get_positive_float("    Temperature (°C) : ")
        vibs.append(v)
        temps.append(t)
    df = pd.DataFrame({"Vibration": vibs, "Temperature": temps})
    if df.isnull().any().any():
        raise DatasetError("Manual data contains missing values.")
    return df

# -------------------------------------------------------
# Clustering
# -------------------------------------------------------
def run_clustering(data, k, label_map):
    """Scale data, run K-Means, display results."""
    required = {"Vibration", "Temperature"}
    missing  = required - set(data.columns)
    if missing:
        raise DatasetError(f"Dataset missing columns: {missing}")
    if k < 2:
        raise ClusteringError("k must be at least 2.")
    if k >= len(data):
        raise ClusteringError(
            f"k ({k}) must be less than number of data points ({len(data)})."
        )

    # Scale
    try:
        scaler     = StandardScaler()
        scaled     = scaler.fit_transform(data[["Vibration", "Temperature"]])
    except (ValueError, TypeError) as e:
        raise ClusteringError(f"Scaling failed: {e}")

    # K-Means
    try:
        kmeans   = KMeans(n_clusters=k, random_state=0, n_init=10)
        clusters = kmeans.fit_predict(scaled)
    except (ValueError, TypeError) as e:
        raise ClusteringError(f"K-Means failed: {e}")

    # Cluster centers in original scale
    try:
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
    except (ValueError, TypeError) as e:
        raise ClusteringError(f"Center inversion failed: {e}")

    print(f"\n{'='*60}")
    print("  K-MEANS CLUSTERING RESULTS")
    print(f"{'='*60}")
    print(f"  Number of clusters : {k}")
    print(f"  Total data points  : {len(data)}")
    print(f"\n  {'Cluster':<10}{'Vibration(g)':<18}{'Temp(°C)':<16}{'Count':<10}{'Label'}")
    print(f"  {'-'*60}")
    for i in range(k):
        vib   = float(centers[i][0])
        temp  = float(centers[i][1])
        count = int((clusters == i).sum())
        label = label_map.get(i, f"Cluster {i}")
        print(f"  {i:<10}{vib:<18.2f}{temp:<16.2f}{count:<10}{label}")
    print(f"{'='*60}")

    return clusters, scaler

# -------------------------------------------------------
# Plotting
# -------------------------------------------------------
def plot_clusters(data, clusters, k, filename="fault_detection_clusters.png"):
    """Visualise clusters using PCA reduction to 2D."""
    try:
        scaler  = StandardScaler()
        scaled  = scaler.fit_transform(data[["Vibration", "Temperature"]])
        pca     = PCA(n_components=2)
        reduced = pca.fit_transform(scaled)
    except (ValueError, np.linalg.LinAlgError) as e:
        raise PlotError(f"PCA failed: {e}")

    colors = ["green", "orange", "red", "blue", "purple", "brown"]
    try:
        plt.figure(figsize=(8, 6))
        for i in range(k):
            mask = clusters == i
            plt.scatter(
                reduced[mask, 0], reduced[mask, 1],
                c=colors[i % len(colors)],
                label=f"Cluster {i}",
                alpha=0.7, edgecolors="black", linewidth=0.3
            )
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title(f"Machine Health Clusters  (K={k}, PCA)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"  ✓ Cluster plot saved: {filename}")
    except (TypeError, ValueError) as e:
        raise PlotError(f"Plot generation failed: {e}")
    except OSError as e:
        raise PlotError(f"Could not save plot: {e}")

# -------------------------------------------------------
# Label Map Helper
# -------------------------------------------------------
def get_label_map(k):
    print(f"\n  Assign health-state labels to each cluster:")
    print(f"  Examples: Normal, Warning, Fault, Critical")
    label_map = {}
    for i in range(k):
        raw = input(f"  Label for Cluster {i}: ").strip()
        label_map[i] = raw if raw else f"Cluster {i}"
    return label_map

# -------------------------------------------------------
# Main Program
# -------------------------------------------------------
def main():
    print("=" * 55)
    print("   EXPERIMENT 08: Fault Detection (K-Means Clustering)")
    print("   AI in Mechanical Engineering — ONT406")
    print("   Sharda University")
    print("=" * 55)

    data     = None
    clusters = None
    k        = None

    while True:
        print("\n--- MENU ---")
        print("1. Use Preset Gearbox Dataset")
        print("2. Generate Custom Dataset")
        print("3. Enter Data Manually")
        print("4. Run K-Means Clustering")
        print("5. Plot Clusters (PCA Visualization)")
        print("6. Exit")

        choice = input("\nEnter your choice (1-6): ").strip()

        if choice in ["4", "5"] and data is None:
            print("  Error: Load data first (options 1, 2, or 3).")
            continue
        if choice == "5" and clusters is None:
            print("  Error: Run clustering first (option 4).")
            continue

        if choice == "1":
            data = preset_dataset()
            print(f"  ✓ Preset dataset loaded: {len(data)} samples.")

        elif choice == "2":
            try:
                n      = get_positive_int(
                    "  Number of samples (min 10): ", minimum=10)
                print("  Vibration range (g):")
                vib_r  = get_range("  Vibration")
                print("  Temperature range (°C):")
                temp_r = get_range("  Temperature")
                data   = custom_dataset(n, vib_r, temp_r)
                print(f"  ✓ Custom dataset created: {len(data)} samples.")
            except DatasetError as e:
                print(f"  Dataset Error: {e}")

        elif choice == "3":
            try:
                data = manual_dataset()
                print(f"  ✓ Manual data entered: {len(data)} samples.")
            except DatasetError as e:
                print(f"  Data Error: {e}")

        elif choice == "4":
            try:
                k         = get_positive_int(
                    "  Number of clusters k (min 2): ", minimum=2)
                label_map = get_label_map(k)
                clusters, _ = run_clustering(data, k, label_map)
            except (DatasetError, ClusteringError) as e:
                print(f"  Clustering Error: {e}")
                clusters = None

        elif choice == "5":
            try:
                plot_clusters(data, clusters, k)
            except PlotError as e:
                print(f"  Plot Error: {e}")

        elif choice == "6":
            print("\nExiting. Goodbye!")
            break

        else:
            print("  Error: Invalid choice. Please enter 1 through 6.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Program interrupted by user. Goodbye!")
