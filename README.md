# Experiment 08: Fault Detection in Machinery using K-Means Clustering

**Subject:** AI in Mechanical Engineering (ONT406)
**Sharda University, Greater Noida**

---

## Aim
To categorize machinery health states (Normal, Warning, Critical) using Unsupervised Learning (K-Means Clustering).

---

## Concepts Covered
- Unsupervised Machine Learning
- K-Means Clustering algorithm
- Feature scaling using StandardScaler
- Dimensionality reduction using PCA
- Machine health state identification

---

## Formulas Used

| Formula | Description |
|---|---|
| Σ||x - μi||² | K-Means objective (minimize intra-cluster variance) |
| Z = (X - μ) / σ | StandardScaler normalization |
| PCA | Reduce dimensions for 2D visualization |

---

## Cluster Interpretation

| Cluster | Vibration | Temperature | State |
|---|---|---|---|
| 0 | Low | Low | Normal |
| 1 | Medium | Medium | Warning |
| 2 | High | High | Fault/Critical |

---

## Software Required

| Software | Purpose | Download Link |
|---|---|---|
| Python 3.x | Programming language | https://www.python.org/downloads/ |
| VS Code | Code editor | https://code.visualstudio.com/ |
| Git | Version control | https://git-scm.com/ |

---

## Installation Steps

### Step 1: Install Python
```
1. Go to https://www.python.org/downloads/
2. Download Python 3.11 or above
3. CHECK "Add Python to PATH"
4. Verify: python --version
```

### Step 2: Install Required Libraries
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Step 3: Verify Installation
```bash
python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"
python -c "import matplotlib; print('Matplotlib:', matplotlib.__version__)"
```

---

## How to Run

```bash
git clone https://github.com/2025514764himanshu-sudo1128/Exp08-KMeans-Fault-Detection.git
cd Exp08-KMeans-Fault-Detection
python fault_detection_kmeans.py
```

---

## Output Files Generated
```
fault_detection_clusters.png   - 2D PCA cluster visualization
```

---

## Expected Console Output
```
=== Cluster Centers (Original Scale) ===
Cluster   Vibration(g)      Temperature(°C)   State
0         1.45              48.23             Normal
1         2.87              74.56             Warning
2         4.21              102.34            Fault/Critical

Cluster Summary:
0    38
1    34
2    28
```

---

## Author
**Himanshu Kumar** (2025514764)
Department of Electrical, Electronics and Communication Engineering
Sharda University, Greater Noida
