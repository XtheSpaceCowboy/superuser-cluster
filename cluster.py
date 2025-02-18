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

# Convert 'TotalCharges' to numeric, as it might contain non-numeric values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with missing TotalCharges
df = df.dropna(subset=['TotalCharges'])

# Remove customers who churned
df = df[df['Churn'] == 'No']

# Apply KMeans clustering with 3 clusters for TotalCharges
X_totalcharges = df[['TotalCharges']].values  
kmeans_totalcharges = KMeans(n_clusters=3, random_state=42, n_init=10)
df['TotalCharges_Cluster'] = kmeans_totalcharges.fit_predict(X_totalcharges)

# Sort clusters by average TotalCharges and assign labels
totalcharges_centers = kmeans_totalcharges.cluster_centers_.flatten()
sorted_totalcharges_clusters = sorted(zip(totalcharges_centers, range(3)), reverse=True)

totalcharges_labels = {
    sorted_totalcharges_clusters[0][1]: "high value customers",
    sorted_totalcharges_clusters[1][1]: "normal customers",
    sorted_totalcharges_clusters[2][1]: "low value customers"
}

df['TotalCharges_Category'] = df['TotalCharges_Cluster'].map(totalcharges_labels)

# Apply KMeans clustering with 3 clusters for tenure
X_tenure = df[['tenure']].values  
kmeans_tenure = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Retention_Cluster'] = kmeans_tenure.fit_predict(X_tenure)

# Sort clusters by average tenure and assign labels
retention_centers = kmeans_tenure.cluster_centers_.flatten()
sorted_retention_clusters = sorted(zip(retention_centers, range(3)), reverse=True)

retention_labels = {
    sorted_retention_clusters[0][1]: "high retention customers",
    sorted_retention_clusters[1][1]: "normal retention customers",
    sorted_retention_clusters[2][1]: "low retention customers"
}

df['Retention_Category'] = df['Retention_Cluster'].map(retention_labels)

# Combine TotalCharges and Retention categories
def combine_categories(row):
    category_map = {
        ('high value customers', 'high retention customers'): 'high-value customer with high tenure',
        ('high value customers', 'normal retention customers'): 'high-value customer with medium tenure',
        ('high value customers', 'low retention customers'): 'high-value customer with low tenure',
        ('normal customers', 'high retention customers'): 'normal customer with high tenure',
        ('normal customers', 'normal retention customers'): 'normal customer with medium tenure',
        ('normal customers', 'low retention customers'): 'normal customer with low tenure',
        ('low value customers', 'high retention customers'): 'low-value customer with high tenure',
        ('low value customers', 'normal retention customers'): 'low-value customer with medium tenure',
        ('low value customers', 'low retention customers'): 'low-value customer with low tenure'
    }
    return category_map.get((row['TotalCharges_Category'], row['Retention_Category']), 'Unknown Category')

df['Customer_Segment'] = df.apply(combine_categories, axis=1)

# Compute statistics for customer segments
customer_segment_stats = df.groupby('Customer_Segment').agg(
    min_totalcharges=('TotalCharges', 'min'),
    min_totalcharges_tenure=('tenure', lambda x: df.loc[x.idxmin(), 'tenure']),
    max_totalcharges=('TotalCharges', 'max'),
    max_totalcharges_tenure=('tenure', lambda x: df.loc[x.idxmax(), 'tenure']),
    count_customers=('customerID', 'size'),
    sum_totalcharges=('TotalCharges', 'sum')
)

# Calculate total customers and TotalCharges percentage
total_customers = df.shape[0]
total_sum_totalcharges = df['TotalCharges'].sum()
customer_segment_stats['percentage_of_total_customers'] = (customer_segment_stats['count_customers'] / total_customers) * 100
customer_segment_stats['percentage_of_totalcharges'] = (customer_segment_stats['sum_totalcharges'] / total_sum_totalcharges) * 100

# Display customer segment statistics
print("\nCustomer Segments Statistics:")
print(customer_segment_stats)

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['tenure'], y=df['TotalCharges'], hue=df['Customer_Segment'], palette="Set1", alpha=0.7)
plt.xlabel("Tenure")
plt.ylabel("Total Charges")
plt.title("Customer Clusters by Tenure and Total Charges")
plt.legend(loc='best')
plt.show()

# Find the top 5 customers by TotalCharges
top_totalcharges_customers = df[['customerID', 'TotalCharges', 'TotalCharges_Category']].sort_values(by='TotalCharges', ascending=False).head(5)
print("\nTop 5 Customers Based on TotalCharges:")
print(top_totalcharges_customers)

# Find the top 5 customers by tenure
top_retention_customers = df[['customerID', 'tenure', 'Retention_Category']].sort_values(by='tenure', ascending=False).head(5)
print("\nTop 5 Customers Based on Tenure:")
print(top_retention_customers)

### COMMONALITY ANALYSIS ###

# Select categorical columns for analysis
categorical_columns = ['Contract', 'PaymentMethod', 'InternetService', 
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies']

# Function to find common attributes in each segment
def get_commonalities(df, group_col):
    commonality_dict = {}

    for category, group in df.groupby(group_col):
        category_commonalities = {}
        for col in categorical_columns:
            most_common = group[col].mode()[0]
            most_common_pct = (group[col] == most_common).mean() * 100
            category_commonalities[col] = (most_common, round(most_common_pct, 2))
        
        commonality_dict[category] = category_commonalities

    return commonality_dict

# Compute commonalities for each segment
customer_segment_commonalities = get_commonalities(df, 'Customer_Segment')

# Print results
for segment, commonalities in customer_segment_commonalities.items():
    print(f"\nCommon Attributes for {segment}:")
    for feature, (value, pct) in commonalities.items():
        print(f"  {feature}: {value} ({pct}%)")

```
