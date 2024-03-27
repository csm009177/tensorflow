import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d

# 분류를 위한 데이터 생성
X_classification, y_classification = make_classification(n_samples=1000, n_features=20, n_classes=2)

# 회귀를 위한 데이터 생성
X_regression, y_regression = make_regression(n_samples=1000, n_features=20, noise=0.1)

# 군집화를 위한 데이터 생성
X_clustering, _ = make_classification(n_samples=1000, n_features=20, n_classes=2, n_clusters_per_class=1)

# 차원 축소를 위한 PCA 적용
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clustering)

# 유사도 계산 (코사인 유사도)
cosine_similarity = np.dot(X_clustering[:1], X_clustering.T)

# 보간을 위한 데이터 생성
x_interp = np.linspace(0, 10, 10)
y_interp = np.sin(x_interp)
f = interp1d(x_interp, y_interp, kind='linear')

# 근사를 위한 다항식 학습
poly_coeffs = np.polyfit(x_interp, y_interp, 3)

print("Classification data shape:", X_classification.shape, y_classification.shape)
print("Regression data shape:", X_regression.shape, y_regression.shape)
print("Clustering data shape:", X_clustering.shape)
print("PCA data shape:", X_pca.shape)
print("Cosine similarity:", cosine_similarity)
print("Interpolated value at x=2:", f(2))
print("Polynomial coefficients:", poly_coeffs)
