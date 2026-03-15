# Wine Dataset Anomaly Detection

Welcome to my anomaly detection project! This pipeline focuses on identifying anomalies within the standard Wine dataset by transforming a classification problem into an outlier detection task.

## The Problem

The original dataset categorizes data into three distinct classes of wine. To adapt this for anomaly detection, Class 0 was designated as the "Anomalies" (outliers) and restricted to only 10 instances, while Classes 1 and 2 were considered the "Normal Data". The main goal is to detect these global/cluster anomalies based on their unique chemical characteristics.

## Approach & Implementation

Here is a brief overview of the steps taken to solve this problem:
* **Exploratory Data Analysis (EDA):** Boxplots were applied to the first four chemical features to visually identify potential extreme values falling outside the normal distribution.
* **Data Preprocessing:** The `StandardScaler` was used to normalize the data, ensuring all features have a mean of 0 and a standard deviation of 1, which is a critical requirement for distance-based algorithms.
* **Model Implementation:** Two different anomaly detection models from the PyOD library were implemented and compared:
    * **KNN (K-Nearest Neighbors):** Identifies anomalies by measuring the average distance to the k-nearest neighbors.
    * **ABOD (Angle-Based Outlier Degree):** Detects outliers by analyzing the variance of angles between points, which is highly effective for isolating points at the edge of clusters.
* **Evaluation & Visualization:** A custom function for the Dunn Index was built to evaluate cluster compactness and separation. PCA was then used to reduce the dimensionality to 2D in order to visualize the decision boundaries and mark the detected anomalies.

## Results & Conclusion

Both models successfully detected all 10 anomalies within the dataset. However, a comparison of the validation metrics showed that:
* **KNN** achieved a Silhouette Score of 0.238 and a Dunn Index of 0.160.
* **ABOD** achieved a Silhouette Score of 0.219 and a Dunn Index of 0.139.

KNN slightly outperformed ABOD in finding cohesive and well-separated clusters. For this specific dataset, the distance-based method (KNN) proved to be more robust and easier to interpret than the probabilistic approach, primarily due to the clear chemical separations between the normal wine types and the anomalous ones.

## Tech Stack

* **Python**
* **PyOD**
* **Scikit-Learn**
* **Pandas** & **NumPy**
* **Matplotlib** & **Seaborn**
