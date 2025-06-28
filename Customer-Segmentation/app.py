from flask import Flask, render_template, request, send_file, redirect, url_for
import pandas as pd
import os
from clustering import perform_clustering

app = Flask(__name__)

# Load once and use to populate dropdowns
DATA_PATH = r"C:\Users\Krishna\Desktop\Internship-Indolike\Customer-Segmentation\data\Clustered_Customers.csv"
df = pd.read_csv(DATA_PATH)
all_columns = df.columns.tolist()
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
gender_options = ['All'] + df['Gender'].dropna().unique().tolist()
palette_options = ['Set1', 'Set2', 'coolwarm', 'viridis', 'plasma', 'Accent', 'Pastel1']

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    result = None

    if request.method == 'POST':
        # Get form inputs
        x_column = request.form.get('x_column')
        y_column = request.form.get('y_column', 'None')
        gender = request.form.get('gender', 'All')
        age_min = request.form.get('age_min')
        age_max = request.form.get('age_max')
        palette = request.form.get('palette', 'Set2')

        # Call clustering function
        result = perform_clustering(
            x_column=x_column,
            y_column=y_column,
            palette=palette,
            gender_filter=gender,
            age_min=age_min,
            age_max=age_max,
            output_plot_path='static/plots/cluster_plot.png',
            output_csv_path='results/clustered_output.csv'
        )

        if result['success']:
            return render_template('result.html',
                                   k=result['k'],
                                   silhouette=result['silhouette'],
                                   x_column=x_column,
                                   y_column=(y_column if y_column != 'None' else None))
        else:
            error = result['error']

    return render_template('index.html',
                           columns=numeric_columns,
                           genders=gender_options,
                           palette_options=palette_options,
                           min_age=int(df['Age'].min()),
                           max_age=int(df['Age'].max()),
                           error=error)


@app.route('/download')
def download():
    return send_file('results/clustered_output.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
