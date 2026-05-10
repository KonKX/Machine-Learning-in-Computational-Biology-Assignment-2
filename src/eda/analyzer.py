from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class EDAAnalyzer:
    def __init__(self, df):
        self.df = df

    def descriptive_statistics(self):
        return self.df.describe(include="all")

    def correlation_matrix(self):
        return self.df.corr(numeric_only=True)

    def run_pca(self, X, n_components=2):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(X_scaled)

        return transformed, pca.explained_variance_ratio_