import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def generate_all_plots(df):
    print("Generating all visualization plots...\n")

    # Ensure TotalCharges is numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])

    # -------------------------
    # 1 Histogram
    # -------------------------
    plt.figure()
    sns.histplot(df['MonthlyCharges'], bins=30, kde=True)
    plt.title("Histogram - Monthly Charges Distribution")
    plt.show()

    # -------------------------
    # 2 Boxplot
    # -------------------------
    plt.figure()
    sns.boxplot(data=df[['tenure','MonthlyCharges','TotalCharges']])
    plt.title("Boxplot of Numerical Variables")
    plt.show()

    # -------------------------
    # 3 Bar Graph
    # -------------------------
    if 'Contract' in df.columns:
        plt.figure()
        df['Contract'].value_counts().plot(kind='bar')
        plt.title("Bar Graph - Contract Distribution")
        plt.xlabel("Contract Type")
        plt.ylabel("Count")
        plt.show()

    # -------------------------
    # 4 Scatter Plot
    # -------------------------
    plt.figure()
    sns.scatterplot(x='tenure', y='TotalCharges', data=df)
    plt.title("Scatter Plot - Tenure vs Total Charges")
    plt.show()

    # -------------------------
    # 5 Line Plot
    # -------------------------
    df_sorted = df.sort_values('tenure')
    plt.figure()
    plt.plot(df_sorted['tenure'], df_sorted['TotalCharges'])
    plt.title("Line Plot - Tenure vs Total Charges Trend")
    plt.xlabel("Tenure")
    plt.ylabel("Total Charges")
    plt.show()

    # -------------------------
    # 6 Grouped & 7 Stacked Bar Chart
    # -------------------------
    if 'Contract' in df.columns:
        grouped = df.groupby('Contract')[['MonthlyCharges','TotalCharges']].mean()
        grouped.plot(kind='bar')
        plt.title("Grouped Bar Chart - Avg Charges by Contract")
        plt.ylabel("Average Value")
        plt.show()

        grouped.plot(kind='bar', stacked=True)
        plt.title("Stacked Bar Chart - Charges by Contract")
        plt.ylabel("Charges")
        plt.show()

    # -------------------------
    # 8 Correlation Heatmap
    # -------------------------
    plt.figure()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

    # -------------------------
    # 9 Covariance Heatmap
    # -------------------------
    plt.figure()
    sns.heatmap(df.cov(numeric_only=True), annot=True, cmap='YlGnBu')
    plt.title("Covariance Heatmap")
    plt.show()

    # -------------------------
    # 10 PCA manually using numpy
    # -------------------------
    features = ['tenure','MonthlyCharges','TotalCharges']
    X = df[features].values.astype(float)

    # Standardize manually
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

    # Covariance matrix
    cov_matrix = np.cov(X_scaled.T)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]  # descending order
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Scree Plot
    explained_variance_ratio = eigenvalues / eigenvalues.sum()
    plt.figure()
    plt.plot(np.cumsum(explained_variance_ratio), marker='o')
    plt.title("Scree Plot")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.show()

    # 2D PCA Projection
    X_pca_2d = X_scaled @ eigenvectors[:, :2]
    plt.figure()
    plt.scatter(X_pca_2d[:,0], X_pca_2d[:,1], alpha=0.6)
    plt.title("2D PCA Projection Plot")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.show()

    print("All plots generated successfully!")