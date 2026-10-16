
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Load dataset
df = pd.read_csv("Mall_Customers.csv")

# 2. Check the data
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

# 3. Select features for clustering
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# 4. Find the optimal number of clusters (Elbow Method)
inertia = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertia.append(model.inertia_)

plt.plot(range(1, 11), inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# 5. Train the K-Means model
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X)

# 6. Assign cluster labels
df["Cluster"] = kmeans.labels_

# 7. Display clustered data
print("\nClustered Data:")
print(df[["CustomerID", "Annual Income (k$)",
          "Spending Score (1-100)", "Cluster"]].head(10))

# 8. Show number of customers in each cluster
print("\nCustomers in each cluster:")
print(df["Cluster"].value_counts().sort_index())

# 9. Visualize clusters
plt.figure(figsize=(9, 6))

plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=df["Cluster"],
    cmap="viridis",
    s=60
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=250,
    color="red",
    label="Centroids"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.legend()
plt.show()

# 10. Save clustered dataset
df.to_csv("Customer_Segments.csv", index=False)

print("\nModel training completed!")
print("Clustered data saved to Customer_Segments.csv")
