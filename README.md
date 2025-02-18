# SuperUser-Cluster
{Data analysis} User clustering to identify superusers/supercustomers

# Background
In order to fully grasp the impact of this process, I recommend first reading [Chamath Palihapitiya on the growth principles that got Facebook to billions of users](https://www.startuparchive.org/p/chamath-palihapitiya-on-the-growth-principles-that-got-facebook-to-billions-of-users).
Chamath was the senior executive at Facebook that was responsible for turning the company around when the core user base was declining in 2007.

The purpose of clustering the users is to identify who the superusers are. This will allow a PM to understand the commonalities in their user-journey. By doing so, PMs can formulate strategies to turn new and existing users into superusers.

# Glossary
 - Super-user (super-customer): A user that has shown a deep NEED for a product. For a product manager, this is a user that maximizes the product's key metrics such as LTV, engagement, and retention.
 - New user: A user that has started using a product. This user has not been using the product for long, but has not become a churned user. Usually, these users have not yet completed the onboarding phase yet, and have not experienced the entirety of the core product. Sometimes, PMs put an arbitrary threshold to label new users: "Used the product for less than 7 days".
 - Churned user: A user that completely stopped using a product.
 - Existing users: All other users. It's important to note that a PM should further categorize existing users to better focus their product strategies.
 - Note: We can further create more user categories - but for now, let's just bucket all of them as existing users.

# Dataset
The public database used is [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). from Kaggle
Each row represents a customer, each column contains customer’s attributes described on the column Metadata. The raw data contains 7043 rows (customers) and 21 columns (features).

# Requirements
- python3
- kagglehub
- numpy
- pandas
- os
- sklearn / Kmeans

# Final Script
```python
import kagglehub
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
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

# Apply KMeans clustering with 3 clusters for TotalCharges
X_totalcharges = df[['TotalCharges']].values  # Use only the 'TotalCharges' column for clustering
kmeans_totalcharges = KMeans(n_clusters=3, random_state=42, n_init=10)
df['TotalCharges_Cluster'] = kmeans_totalcharges.fit_predict(X_totalcharges)

# Sort the clusters by the average TotalCharges for each cluster to label them correctly
totalcharges_centers = kmeans_totalcharges.cluster_centers_.flatten()
sorted_totalcharges_clusters = sorted(zip(totalcharges_centers, range(3)), reverse=True)

# Map clusters to customer categories based on the sorted centers
totalcharges_labels = {
    sorted_totalcharges_clusters[0][1]: "high value customers",  # Highest TotalCharges
    sorted_totalcharges_clusters[1][1]: "normal customers",      # Middle TotalCharges
    sorted_totalcharges_clusters[2][1]: "low value customers"    # Lowest TotalCharges
}

# Assign category labels for TotalCharges
df['TotalCharges_Category'] = df['TotalCharges_Cluster'].map(totalcharges_labels)

# Group by TotalCharges_Category and compute min, max, and count for TotalCharges
totalcharges_stats = df.groupby('TotalCharges_Category')['TotalCharges'].agg(['min', 'max', 'count'])

# Print the statistics for TotalCharges category
print("Total Charges Category Statistics:")
print(totalcharges_stats)

# Apply KMeans clustering with 3 clusters for tenure
X_tenure = df[['tenure']].values  # Use only the 'tenure' column for clustering
kmeans_tenure = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Retention_Cluster'] = kmeans_tenure.fit_predict(X_tenure)

# Sort the clusters by the average tenure for each cluster to label them correctly
retention_centers = kmeans_tenure.cluster_centers_.flatten()
sorted_retention_clusters = sorted(zip(retention_centers, range(3)), reverse=True)

# Map clusters to retention categories based on the sorted centers
retention_labels = {
    sorted_retention_clusters[0][1]: "high retention customers",  # Highest tenure
    sorted_retention_clusters[1][1]: "normal retention customers", # Middle tenure
    sorted_retention_clusters[2][1]: "low retention customers"     # Lowest tenure
}

# Assign retention category labels to each row
df['Retention_Category'] = df['Retention_Cluster'].map(retention_labels)

# Group by Retention_Category and compute min, max, and count for tenure
retention_stats = df.groupby('Retention_Category')['tenure'].agg(['min', 'max', 'count'])

# Print the statistics for Retention (tenure) category
print("\nRetention Category (Tenure) Statistics:")
print(retention_stats)

# Group by Retention_Category and compute min, max for both tenure and TotalCharges
retention_stats = df.groupby('Retention_Category').agg(
    min_tenure=('tenure', 'min'),
    max_tenure=('tenure', 'max'),
    min_totalcharges=('TotalCharges', 'min'),
    max_totalcharges=('TotalCharges', 'max')
)

# Create a new column to combine tenure stats with TotalCharges stats
retention_stats['Tenure_and_TotalCharges'] = retention_stats.apply(
    lambda row: f"{row['min_tenure']} ({row['min_totalcharges']}), {row['max_tenure']} ({row['max_totalcharges']})", axis=1
)

# Print the statistics with Tenure and TotalCharges in the desired format
print("\nRetention Category (Tenure) and TotalCharges Statistics:")
print(retention_stats[['Tenure_and_TotalCharges']])

# Define a new category combining TotalCharges and Retention Categories
def combine_categories(row):
    totalcharges_category = row['TotalCharges_Category']
    retention_category = row['Retention_Category']
    
    # Mapping of categories based on previous clustering
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
    
    # Return the combined category
    return category_map.get((totalcharges_category, retention_category), 'Unknown Category')

# Apply the function to create the new combined category
df['Customer_Segment'] = df.apply(combine_categories, axis=1)

# Group by Customer_Segment and calculate required statistics
customer_segment_stats = df.groupby('Customer_Segment').agg(
    min_totalcharges=('TotalCharges', 'min'),
    min_totalcharges_tenure=('tenure', lambda x: df.loc[x.idxmin(), 'tenure']),
    max_totalcharges=('TotalCharges', 'max'),
    max_totalcharges_tenure=('tenure', lambda x: df.loc[x.idxmax(), 'tenure']),
    count_customers=('customerID', 'size'),
    sum_totalcharges=('TotalCharges', 'sum')
)

# Calculate the percentage of total customers in each segment
total_customers = df.shape[0]
customer_segment_stats['percentage_of_total_customers'] = (customer_segment_stats['count_customers'] / total_customers) * 100

# Calculate the total sum of TotalCharges for all customers
total_sum_totalcharges = df['TotalCharges'].sum()

# Calculate the percentage of TotalCharges for each segment
customer_segment_stats['percentage_of_totalcharges'] = (customer_segment_stats['sum_totalcharges'] / total_sum_totalcharges) * 100

# Calculate the percentage of total customers in each segment
total_customers = df.shape[0]
customer_segment_stats['percentage_of_total_customers'] = (customer_segment_stats['count_customers'] / total_customers) * 100

# Print the statistics for each customer segment
print("\nCustomer Segments with Min/Max TotalCharges, Associated Tenure, Sum of TotalCharges, and Percentages:")
print(customer_segment_stats[['min_totalcharges', 'min_totalcharges_tenure', 
                              'max_totalcharges', 'max_totalcharges_tenure', 
                              'count_customers', 'percentage_of_total_customers',
                              'sum_totalcharges', 'percentage_of_totalcharges']])


# Print the top 5 customers based on TotalCharges
top_totalcharges_customers = df[['customerID', 'TotalCharges', 'TotalCharges_Category']].sort_values(by='TotalCharges', ascending=False).head(5)

print("\nTop 5 Customers Based on TotalCharges:")
print(top_totalcharges_customers)

# Print the top 5 customers based on tenure
top_retention_customers = df[['customerID', 'tenure', 'Retention_Category']].sort_values(by='tenure', ascending=False).head(5)

print("\nTop 5 Customers Based on Tenure:")
print(top_retention_customers)
```
