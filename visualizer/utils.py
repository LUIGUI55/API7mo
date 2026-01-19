import base64
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_classification
from sklearn.cluster import KMeans, DBSCAN
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def generate_kmeans_graph():
    plt.switch_backend('AGG')
    plt.figure(figsize=(10, 6))
    
    # Synthetic Data representing credit card transactions (PCA reduced)
    # 2 clusters: Normal vs Fraud (highly imbalanced)
    X, y = make_blobs(n_samples=500, centers=2, n_features=2, random_state=42, cluster_std=[1.0, 2.5])
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', alpha=0.6)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.9, marker='X', label='Centroides')
    plt.title('Clustering K-Means: Simulación de Fraude')
    plt.xlabel('Característica V1 (PCA)')
    plt.ylabel('Característica V2 (PCA)')
    plt.legend()
    plt.tight_layout()
    
    return get_graph()

def generate_dbscan_graph():
    plt.switch_backend('AGG')
    plt.figure(figsize=(10, 6))
    
    # DBSCAN is good for non-linear separators and noise
    # Generating moon-shaped data to demonstrate DBSCAN capability
    X, _ = make_moons(n_samples=500, noise=0.05, random_state=42)
    
    # Add some random noise points (outliers) to simulate fraud anomalies
    outliers = np.random.uniform(low=-1.5, high=2.5, size=(20, 2))
    X = np.vstack([X, outliers])
    
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    clusters = dbscan.fit_predict(X)
    
    # -1 indicates noise in DBSCAN
    unique_labels = set(clusters)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
            label = 'Ruido (Posible Fraude)'
        else:
            label = f'Cluster {k}'

        class_member_mask = (clusters == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6, label=label)

    plt.title('Clustering DBSCAN: Detección de Anomalías')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.legend()
    plt.tight_layout()
    
    return get_graph()

def generate_naive_bayes_graph():
    plt.switch_backend('AGG')
    plt.figure(figsize=(10, 6))
    
    # Synthetic Data for Email Spam Classification
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusión Naive Bayes: Detección de SPAM')
    plt.xlabel('Etiqueta Predicha (0: Ham, 1: Spam)')
    plt.ylabel('Etiqueta Real (0: Ham, 1: Spam)')
    plt.tight_layout()
    
    return get_graph()
