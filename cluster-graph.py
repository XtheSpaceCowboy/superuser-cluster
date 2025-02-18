```python
import kagglehub
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Download latest version
path = kagglehub.dataset_download("blastchar/telco-customer-churn")

# Get the actual CSV file path
csv_path = os.path.join(path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Load the dataset
df = pd.read_csv(csv_path)

# Convert 'TotalCharges' to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

# Apply KMeans clustering for TotalCharges
X_totalcharges = df[['TotalCharges']].values
kmeans_totalcharges = KMeans(n_clusters=3, random_state=42, n_init=10)
df['TotalCharges_Cluster'] = kmeans_totalcharges.fit_predict(X_totalcharges)

# Sort clusters by TotalCharges centers
totalcharges_centers = kmeans_totalcharges.cluster_centers_.flatten()
sorted_totalcharges_clusters = sorted(zip(totalcharges_centers, range(3)), reverse=True)

totalcharges_labels = {
    sorted_totalcharges_clusters[0][1]: "high value customers",
    sorted_totalcharges_clusters[1][1]: "normal customers",
    sorted_totalcharges_clusters[2][1]: "low value customers"
}

df['TotalCharges_Category'] = df['TotalCharges_Cluster'].map(totalcharges_labels)

# Apply KMeans clustering for tenure
X_tenure = df[['tenure']].values
kmeans_tenure = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Retention_Cluster'] = kmeans_tenure.fit_predict(X_tenure)

# Sort clusters by tenure centers
retention_centers = kmeans_tenure.cluster_centers_.flatten()
sorted_retention_clusters = sorted(zip(retention_centers, range(3)), reverse=True)

retention_labels = {
    sorted_retention_clusters[0][1]: "high retention customers",
    sorted_retention_clusters[1][1]: "normal retention customers",
    sorted_retention_clusters[2][1]: "low retention customers"
}

df['Retention_Category'] = df['Retention_Cluster'].map(retention_labels)

# Combine categories
def combine_categories(row):
    return f"{row['TotalCharges_Category']} with {row['Retention_Category']}"

df['Customer_Segment'] = df.apply(combine_categories, axis=1)

# Scatter plot visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df["tenure"], 
    y=df["TotalCharges"], 
    hue=df["Customer_Segment"], 
    palette="viridis", 
    alpha=0.7
)

plt.xlabel("Tenure (Months)")
plt.ylabel("Total Charges ($)")
plt.title("Customer Segments: TotalCharges vs. Tenure")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()
```
