import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Fix: use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

# Path to the fixed dataset
DATA_PATH = r"C:\Users\Krishna\Desktop\Internship-Indolike\Customer-Segmentation\data\Clustered_Customers.csv"

def optimal_k(X, max_k=10):
    """Improved elbow method using KneeLocator to detect the elbow point."""
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Use KneeLocator to find elbow
    kl = KneeLocator(range(1, max_k + 1), wcss, curve='convex', direction='decreasing')
    elbow_k = kl.elbow

    return elbow_k if elbow_k is not None else 2

def perform_clustering(
    x_column,
    y_column,
    palette='Set2',
    gender_filter='All',
    age_min=None,
    age_max=None,
    output_plot_path='static/plots/cluster_plot.png',
    output_csv_path='results/clustered_output.csv'
):
    try:
        # Load dataset
        df = pd.read_csv(DATA_PATH)

        # Apply gender filter
        if gender_filter != 'All':
            df = df[df['Gender'].str.lower() == gender_filter.lower()]

        # Apply age filter
        if age_min is not None and age_max is not None:
            df = df[(df['Age'] >= int(age_min)) & (df['Age'] <= int(age_max))]

        # Select features
        if y_column == 'None':
            X = df[[x_column]]
            plot_2d = False
        else:
            X = df[[x_column, y_column]]
            plot_2d = True

        # Auto-determine k
        k = optimal_k(X)

        # Fit KMeans
        model = KMeans(n_clusters=k, init='k-means++', random_state=42)
        df['Cluster'] = model.fit_predict(X)

        # Silhouette Score
        silhouette = silhouette_score(X, df['Cluster']) if k > 1 else 0.0

        # Plot clusters
        plt.figure(figsize=(8, 5))
        if plot_2d:
            sns.scatterplot(x=x_column, y=y_column, hue='Cluster', data=df, palette=palette, s=100)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
        else:
            sns.stripplot(x=x_column, y=[''] * len(df), hue='Cluster', data=df, palette=palette, size=10, jitter=True)
            plt.xlabel(x_column)
            plt.ylabel("")

        plt.title(f'Customer Segments (Auto k = {k})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_plot_path)
        plt.close()

        # Save clustered data
        df.to_csv(output_csv_path, index=False)

        # Return results
        return {
            'success': True,
            'k': k,
            'silhouette': round(silhouette, 4),
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'k': None,
            'silhouette': None,
            'error': str(e)
        }
