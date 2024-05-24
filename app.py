import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/team')
def index1():
    return render_template('index1.html')

@app.route('/solo')
def index():
    return render_template('index.html')

@app.route('/calculate_t20', methods=['POST'])
def calculate_t20():
    data = request.get_json()
    number = data.get('number', 0)
    #result = number * 2  # Replace with your desired calculation


    file_path = 't20.csv'
    data = pd.read_csv(file_path)

# Clean the data by replacing '-' with -1 in the 'SR' column
    data['SR'] = data['SR'].replace('-', -1)

# Convert the 'SR' column to numeric
    data['SR'] = pd.to_numeric(data['SR'], errors='coerce')

# Remove rows with missing values in the 'SR' column (if any)
    data = data.dropna(subset=['SR'])

# Extract the SR attribute
    sr_data = data[['SR']]

# Determine the optimal number of clusters (K) using the Elbow method
    wcss = []  # Within-cluster sum of squares
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(sr_data)
        wcss.append(kmeans.inertia_)

# Based on the Elbow method plot, choose an appropriate K value
    k_value = int(3)

# Perform K-means clustering with the chosen K value
    kmeans = KMeans(n_clusters=k_value, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(sr_data)

# Add cluster labels to the original data
    data['Cluster'] = kmeans.labels_

# Assign labels to clusters based on values
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['Cluster Center'])
    cluster_centers_sorted = cluster_centers.sort_values(by='Cluster Center')

# Define labels for clusters based on values
    cluster_labels = {
        cluster_centers_sorted.index[0]: 'Defensive Player',
        cluster_centers_sorted.index[1]: 'Consistent Player',
        cluster_centers_sorted.index[2]: 'Aggressive Player'
    }

# Replace cluster labels in the DataFrame
    data['Cluster'] = data['Cluster'].map(cluster_labels)

# Print the cluster labels and their counts
    cluster_counts = data['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Player Type', 'Count']
    print(cluster_counts)

# Save the clustered data to a new CSV file
    output_file = 'clustered_t20_data.csv'
    data.to_csv(output_file, index=False)

# Plot the clusters
    # plt.scatter(sr_data['SR'], [0] * len(sr_data), c=kmeans.labels_, cmap='rainbow')
    # plt.xlabel('SR')
    # plt.title(f'K-means Clustering (K={k_value})')
    # plt.show()

    new_player_sr = number

# # Create a DataFrame with the new player's data
    new_player_data = pd.DataFrame({'SR': [new_player_sr]})

# # Use the K-means model to predict the cluster for the new player
    new_player_cluster = kmeans.predict(new_player_data)

# # Get the player type label for the new player's cluster
    new_player_type = cluster_labels[new_player_cluster[0]]

# # Print the player type for the new player
    #print(f"The new player is classified as '{new_player_type}'")

    return jsonify({'result': new_player_type})
@app.route('/calculate_odi', methods=['POST'])
def calculate_odi():
    data = request.get_json()
    number = data.get('number', 0)
    #result = number * 2  # Replace with your desired calculation


    file_path = 'ODI.csv'
    data = pd.read_csv(file_path)

# Clean the data by replacing '-' with -1 in the 'SR' column
    data['SR'] = data['SR'].replace('-', -1)

# Convert the 'SR' column to numeric
    data['SR'] = pd.to_numeric(data['SR'], errors='coerce')

# Remove rows with missing values in the 'SR' column (if any)
    data = data.dropna(subset=['SR'])

# Extract the SR attribute
    sr_data = data[['SR']]

# Determine the optimal number of clusters (K) using the Elbow method
    wcss = []  # Within-cluster sum of squares
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(sr_data)
        wcss.append(kmeans.inertia_)

# Based on the Elbow method plot, choose an appropriate K value
    k_value = int(3)

# Perform K-means clustering with the chosen K value
    kmeans = KMeans(n_clusters=k_value, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(sr_data)

# Add cluster labels to the original data
    data['Cluster'] = kmeans.labels_

# Assign labels to clusters based on values
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['Cluster Center'])
    cluster_centers_sorted = cluster_centers.sort_values(by='Cluster Center')

# Define labels for clusters based on values
    cluster_labels = {
        cluster_centers_sorted.index[0]: 'Defensive Player',
        cluster_centers_sorted.index[1]: 'Consistent Player',
        cluster_centers_sorted.index[2]: 'Aggressive Player'
    }

# Replace cluster labels in the DataFrame
    data['Cluster'] = data['Cluster'].map(cluster_labels)

# Print the cluster labels and their counts
    cluster_counts = data['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Player Type', 'Count']
    print(cluster_counts)

# Save the clustered data to a new CSV file
    output_file = 'clustered_ODI_data.csv'
    data.to_csv(output_file, index=False)

# Plot the clusters
    # plt.scatter(sr_data['SR'], [0] * len(sr_data), c=kmeans.labels_, cmap='rainbow')
    # plt.xlabel('SR')
    # plt.title(f'K-means Clustering (K={k_value})')
    # plt.show()

    new_player_sr = number

# # Create a DataFrame with the new player's data
    new_player_data = pd.DataFrame({'SR': [new_player_sr]})

# # Use the K-means model to predict the cluster for the new player
    new_player_cluster = kmeans.predict(new_player_data)

# # Get the player type label for the new player's cluster
    new_player_type = cluster_labels[new_player_cluster[0]]

# # Print the player type for the new player
    #print(f"The new player is classified as '{new_player_type}'")

    return jsonify({'result': new_player_type})
@app.route('/calculate_test', methods=['POST'])
def calculate_test():
    data = request.get_json()
    number = data.get('number', 0)
    #result = number * 2  # Replace with your desired calculation


    file_path = 'test.csv'
    data = pd.read_csv(file_path)

# Clean the data by replacing '-' with -1 in the 'SR' column
    data['SR'] = data['SR'].replace('-', -1)

# Convert the 'SR' column to numeric
    data['SR'] = pd.to_numeric(data['SR'], errors='coerce')

# Remove rows with missing values in the 'SR' column (if any)
    data = data.dropna(subset=['SR'])

# Extract the SR attribute
    sr_data = data[['SR']]

# Determine the optimal number of clusters (K) using the Elbow method
    wcss = []  # Within-cluster sum of squares
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(sr_data)
        wcss.append(kmeans.inertia_)

# Based on the Elbow method plot, choose an appropriate K value
    k_value = int(3)

# Perform K-means clustering with the chosen K value
    kmeans = KMeans(n_clusters=k_value, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(sr_data)

# Add cluster labels to the original data
    data['Cluster'] = kmeans.labels_

# Assign labels to clusters based on values
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['Cluster Center'])
    cluster_centers_sorted = cluster_centers.sort_values(by='Cluster Center')

# Define labels for clusters based on values
    cluster_labels = {
        cluster_centers_sorted.index[0]: 'Defensive Player',
        cluster_centers_sorted.index[1]: 'Consistent Player',
        cluster_centers_sorted.index[2]: 'Aggressive Player'
    }

# Replace cluster labels in the DataFrame
    data['Cluster'] = data['Cluster'].map(cluster_labels)

# Print the cluster labels and their counts
    cluster_counts = data['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Player Type', 'Count']
    print(cluster_counts)

# Save the clustered data to a new CSV file
    output_file = 'clustered_test_data.csv'
    data.to_csv(output_file, index=False)

# Plot the clusters
    # plt.scatter(sr_data['SR'], [0] * len(sr_data), c=kmeans.labels_, cmap='rainbow')
    # plt.xlabel('SR')
    # plt.title(f'K-means Clustering (K={k_value})')
    # #plt.show()

    new_player_sr = number

# # Create a DataFrame with the new player's data
    new_player_data = pd.DataFrame({'SR': [new_player_sr]})

# # Use the K-means model to predict the cluster for the new player
    new_player_cluster = kmeans.predict(new_player_data)

# # Get the player type label for the new player's cluster
    new_player_type = cluster_labels[new_player_cluster[0]]

# # Print the player type for the new player
    #print(f"The new player is classified as '{new_player_type}'")

    return jsonify({'result': new_player_type})
if __name__ == '__main__':
    app.run(debug=True)
