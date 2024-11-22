import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Step 1: Read data from a CSV file
csv_file = "predictions.csv"
df = pd.read_csv(csv_file)

# Step 2: Extract relevant features for clustering
typing_predictions = df[['x_center', 'y_center', 'timestamp', 'confidence']].values

# Step 3: Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(typing_predictions)

# Step 4: Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=2)  # Tune these parameters
cluster_labels = dbscan.fit_predict(normalized_features)

# Step 5: Add cluster labels back to the DataFrame
df['cluster_label'] = cluster_labels
print(df)

# Step 6: Save the clustered data to a new CSV
output_file = "clustered_predictions.csv"
df.to_csv(output_file, index=False)

