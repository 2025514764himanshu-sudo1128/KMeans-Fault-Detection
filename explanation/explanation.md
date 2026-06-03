# Experiment 08 — Code Explanation
# Fault Detection using K-Means Clustering

---

## What is this program doing?

In Experiment 7 we used **supervised learning** — we knew
which data points were "worn" and which weren't (labeled data).

But in real factories, you often DON'T have labels.
You just have sensor readings and want to know:
"Is the machine normal or is something wrong?"

This is **Unsupervised Learning** — finding patterns
in data WITHOUT labels.

This program:
1. Takes raw vibration and temperature sensor data
2. Automatically groups similar readings into clusters
3. Each cluster represents a different health state
4. Visualizes the clusters in 2D using PCA

---

## K-Means Algorithm — Explained Simply

**Imagine you have 100 scattered points on a paper.**
K-Means does this:
1. Place K=3 random center points (centroids)
2. Assign each data point to the nearest centroid
3. Move each centroid to the average position of its assigned points
4. Repeat steps 2-3 until centroids stop moving

**Result:** 3 groups of similar points.

---

## Line by Line Explanation

---

### Lines 1-5 (Imports)
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```
**KMeans:** The clustering algorithm from scikit-learn.

**StandardScaler:** Normalizes data so all features
have equal weight in clustering.

**PCA (Principal Component Analysis):**
Reduces high-dimensional data to 2D for visualization.

---

### Lines 8-12 (Generate Data)
```python
data = pd.DataFrame({
    "Vibration":   np.random.uniform(0.5, 5.0, 100),
    "Temperature": np.random.uniform(30, 120, 100)
})
```
In real applications, this data comes from sensors.
Vibration in g-force, Temperature in °C.

High vibration + high temperature often means fault.
Low values mean normal operation.

---

### Lines 15-17 (Feature Scaling)
```python
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
```
**Why is scaling necessary?**

Vibration range: 0.5 to 5.0 (range ≈ 4.5)
Temperature range: 30 to 120 (range ≈ 90)

Without scaling, Temperature dominates the clustering
because its values are 20× larger than Vibration.
The algorithm would essentially ignore Vibration!

**What StandardScaler does:**
Converts each feature to have mean=0 and std=1.
Formula: Z = (X - mean) / std

After scaling, both features have equal influence.

**fit_transform():**
Two steps in one:
- `fit()` = calculate mean and std from data
- `transform()` = apply the scaling formula

---

### Lines 20-22 (K-Means Clustering)
```python
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
clusters = kmeans.fit_predict(scaled_data)
data["Cluster"] = clusters
```
**n_clusters=3:**
We want 3 groups: Normal, Warning, Fault.
(In real use, you may not know k — use the Elbow Method)

**random_state=0:**
K-Means starts with random centroids.
Setting this ensures same result every run.

**n_init=10:**
Runs K-Means 10 times with different starting points.
Picks the best result. Prevents getting stuck in bad solution.

**fit_predict():**
Two steps in one:
- `fit()` = runs K-Means algorithm
- `predict()` = assigns cluster label to each data point

Returns array like [0, 2, 1, 0, 2, 1, ...] — cluster IDs.

**data["Cluster"] = clusters:**
Adds cluster labels as a new column in DataFrame.
Now each row knows which group it belongs to.

---

### Lines 25-35 (Cluster Centers)
```python
centers = scaler.inverse_transform(kmeans.cluster_centers_)
```
**kmeans.cluster_centers_:**
The final centroid positions AFTER clustering.
But these are in SCALED units — not original units.

**scaler.inverse_transform():**
Converts scaled values BACK to original units (g and °C).
Now we can interpret: "Cluster 2 has avg vibration = 4.2g"

**The classification logic:**
```python
if vib < 2.0:   → Normal
elif vib < 3.5: → Warning
else:           → Fault
```
This is based on domain knowledge from the lab manual.
K-Means doesn't know which cluster is "normal" —
the engineer must interpret the cluster centers.

---

### Lines 38-42 (PCA Visualization)
```python
pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled_data)
```
**What is PCA?**
Principal Component Analysis reduces many dimensions
to fewer dimensions while preserving the most important variation.

**Why do we need it here?**
Our data has 2 features (Vibration, Temperature) — already 2D.
But in real systems you might have 10+ sensors.
PCA reduces 10D → 2D so you can plot it.

Here with 2D data, PCA doesn't change much —
it just rotates the axes to align with maximum variance.

**n_components=2:**
Reduce to 2 principal components (for 2D plot).

**fit_transform():**
Learns the transformation and applies it in one step.

---

### Lines 45-57 (Plotting Clusters)
```python
colors = ['green', 'orange', 'red']
labels = ['Normal', 'Warning', 'Fault']
for i in range(3):
    mask = clusters == i
    plt.scatter(reduced[mask, 0], reduced[mask, 1],
                c=colors[i], label=labels[i])
```
**What is `mask = clusters == i`?**
Creates a True/False array — True where cluster == i.

**`reduced[mask, 0]`:**
Selects only the rows where mask is True.
- `mask` selects rows
- `0` selects first column (PC1)
- `1` selects second column (PC2)

**Why different colors?**
Green = Normal (safe to see)
Orange = Warning (caution)
Red = Fault (danger)

This visual convention is used in dashboards worldwide.

---

## Comparison: Supervised vs Unsupervised

| Aspect | Experiment 7 (Regression) | Experiment 8 (K-Means) |
|---|---|---|
| Type | Supervised | Unsupervised |
| Labels needed | Yes | No |
| Output | Continuous number (wear mm) | Group assignment (0/1/2) |
| Use case | Predict exact failure time | Identify anomaly groups |
| Training data | Needs historical labeled data | Needs only sensor readings |

---

## Real World Application

K-Means clustering is used for:
- **Bearing fault detection** in wind turbines
- **Anomaly detection** in power plant data
- **Patient grouping** in medical devices
- **Network intrusion detection** in cybersecurity
