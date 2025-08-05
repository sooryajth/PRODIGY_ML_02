import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n_customers = 200

annual_spending = np.random.normal(1000, 300, n_customers)
visit_frequency = np.random.randint(1, 50, n_customers)
avg_basket_size = np.random.normal(50, 15, n_customers)
product_categories = np.random.randint(1, 10, n_customers)

data = pd.DataFrame({
    'AnnualSpending': annual_spending,
    'VisitFrequency': visit_frequency,
    'AvgBasketSize': avg_basket_size,
    'ProductCategories': product_categories
})

print(data.head())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

data['Cluster'] = clusters

print(data.head())

plt.figure(figsize=(10, 6))
plt.scatter(data['AnnualSpending'], data['VisitFrequency'], c=data['Cluster'], cmap='viridis')
plt.title('Customer Segments')
plt.xlabel('Annual Spending')
plt.ylabel('Visit Frequency')
plt.show()

cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_df = pd.DataFrame(cluster_centers, columns=data.columns[:-1])
print(cluster_df)

for i in range(4):
    cluster_data = data[data['Cluster'] == i]
    print(f"\nCluster {i} Characteristics:")
    print(cluster_data.describe())