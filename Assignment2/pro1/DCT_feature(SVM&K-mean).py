import numpy as np
from scipy.fftpack import dct
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.neural_network import MLPClassifier
# ==============================
# Step 1: Create ReducedMNIST
# ==============================
def create_reduced_mnist(random_seed=1000):
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    X = X.to_numpy() / 255.0  # Convert to NumPy array and normalize
    y = y.astype(int)
    
    np.random.seed(random_seed)
    
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    for digit in range(10):
        digit_indices = np.where(y == digit)[0]
        selected = np.random.choice(digit_indices, 1200, replace=False)
        X_train_list.append(X[selected[:1000]])
        y_train_list.append(y[selected[:1000]])
        X_test_list.append(X[selected[1000:1200]])
        y_test_list.append(y[selected[1000:1200]])
    
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)
    
    return X_train, y_train, X_test, y_test

# ==============================
# Step 2: DCT Feature Extraction
# ==============================
def extract_dct_features(X, size=15):
    """
    Extract top-left DCT coefficients as features
    X: array of shape (n_samples, 784)
    size: the number of DCT coefficients per dimension (size x size = feature dim)
    """
    n_samples = X.shape[0]
    dct_features = np.zeros((n_samples, size*size))
    
    for i in range(n_samples):
        img = X[i].reshape(28,28)
        dct2 = dct(dct(img.T, norm='ortho').T, norm='ortho')
        dct_features[i] = dct2[:size,:size].flatten()
        
    return dct_features

# ==============================
# Step 3: K-Means Classifier
# ==============================
def kmeans_classifier(X_train, y_train, X_test, n_clusters_per_class):
    start_time = time.time()
    
    centroids = []
    labels = []
    for digit in range(10):
        digit_features = X_train[y_train == digit]
        kmeans = KMeans(n_clusters=n_clusters_per_class, random_state=42, n_init="auto").fit(digit_features)
        centroids.append(kmeans.cluster_centers_)
        labels.extend([digit]*n_clusters_per_class)
    
    centroids = np.vstack(centroids)
    labels = np.array(labels)
    
    predictions = []
    for x in X_test:
        dists = np.linalg.norm(centroids - x, axis=1)
        pred = labels[np.argmin(dists)]
        predictions.append(pred)
    predictions = np.array(predictions)
    
    end_time = time.time()
    total_time = end_time - start_time
    accuracy = accuracy_score(y_test, predictions) * 100
    return predictions, accuracy, total_time

# ==============================
# Step 4: SVM Classifier
# ==============================
def svm_classifier(X_train, y_train, X_test, kernel_type='linear'):
    start_time = time.time()
    clf = SVC(kernel=kernel_type, gamma='scale')
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    end_time = time.time()
    total_time = end_time - start_time
    accuracy = accuracy_score(y_test, predictions) * 100
    return predictions, accuracy, total_time
# ==============================
# Step 5: MLP Classifier
# ==============================
def mlp_classifier(X_train, y_train, X_test, y_test, hidden_layers):
    start_time = time.time()

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        max_iter=100,
        early_stopping=True,
        n_iter_no_change=10,
        learning_rate_init=0.001,
        random_state=42
    )

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    total_time = time.time() - start_time
    accuracy = accuracy_score(y_test, predictions) * 100

    return predictions, accuracy, total_time

# ==============================
# Main Execution
# ==============================
X_train, y_train, X_test, y_test = create_reduced_mnist()

print("Extracting DCT features...")
X_train_dct = extract_dct_features(X_train, size=15)  # 15x15=225
X_test_dct = extract_dct_features(X_test, size=15)

# ==============================
# K-Means Classification
# ==============================
n_clusters_list = [1, 4, 16, 32]
kmeans_results = []

for n_clusters in n_clusters_list:
    print(f"\nK-Means with {n_clusters} clusters per class...")
    preds, acc, t = kmeans_classifier(X_train_dct, y_train, X_test_dct, n_clusters)
    kmeans_results.append({'clusters': n_clusters, 'accuracy': acc, 'time': t, 'predictions': preds})
    print(f"Accuracy: {acc:.2f}% | Total Time: {t:.2f} sec")

best_kmeans = max(kmeans_results, key=lambda x: x['accuracy'])

# ==============================
# SVM Classification
# ==============================
svm_kernels = ['linear', 'rbf']
svm_results = []

for kernel in svm_kernels:
    print(f"\nSVM with {kernel} kernel...")
    preds, acc, t = svm_classifier(X_train_dct, y_train, X_test_dct, kernel_type=kernel)
    svm_results.append({'kernel': kernel, 'accuracy': acc, 'time': t, 'predictions': preds})
    print(f"Accuracy: {acc:.2f}% | Total Time: {t:.2f} sec")

best_svm = max(svm_results, key=lambda x: x['accuracy'])
# ==============================
# MLP Classification
# ==============================
mlp_configs = {
    "1-hidden": (128,),
    "3-hidden": (128, 64, 32),
    "4-hidden": (256, 128, 64, 32)
}

mlp_results = []

for name, layers in mlp_configs.items():
    print(f"\nMLP with {name} layers...")
    preds, acc, t = mlp_classifier(X_train_dct, y_train, X_test_dct, y_test, layers)
    
    mlp_results.append({
        'config': name,
        'accuracy': acc,
        'time': t,
        'predictions': preds
    })
    
    print(f"Accuracy: {acc:.2f}% | Total Time: {t:.2f} sec")

best_mlp = max(mlp_results, key=lambda x: x['accuracy'])


# ==============================
# Summary Table
# ==============================
print("\nSummary Table (DCT Features):")
print("Classifier     | Parameter       | Accuracy (%) | Total Time (s)")
print("---------------------------------------------------------------")

for res in kmeans_results:
    print(f"K-Means       | {res['clusters']} clusters  | {res['accuracy']:11.2f} | {res['time']:14.2f}")

for res in svm_results:
    print(f"SVM           | {res['kernel']} kernel    | {res['accuracy']:11.2f} | {res['time']:14.2f}")
for res in mlp_results:
    print(f"MLP           | {res['config']:12} | {res['accuracy']:11.2f} | {res['time']:14.2f}")