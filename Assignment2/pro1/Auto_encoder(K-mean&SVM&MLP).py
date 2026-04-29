import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time


# Autoencoder
import tensorflow as tf
from tensorflow.keras import layers, models

# ==============================
# Step 1: Create ReducedMNIST
# ==============================
def create_reduced_mnist(random_seed=1000):
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    X = X.to_numpy() / 255.0
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
# Step 2: Autoencoder Features
# ==============================
def build_autoencoder(input_dim=784, latent_dim=64):
    input_layer = layers.Input(shape=(input_dim,))

    dropout_rate = 0.4  # Dropout value, can be adjusted

    # Encoder
    encoded = layers.Dense(256, activation='relu')(input_layer)
    encoded = layers.Dropout(dropout_rate)(encoded)
    encoded = layers.Dense(latent_dim, activation='relu')(encoded)

    # Decoder
    decoded = layers.Dense(256, activation='relu')(encoded)
    decoded = layers.Dropout(dropout_rate)(decoded)
    decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = models.Model(input_layer, decoded)
    encoder = models.Model(input_layer, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder

# ==============================
# Step 3: K-Means Classifier
# ==============================
def kmeans_classifier(X_train, y_train, X_test, y_test, n_clusters_per_class):
    start_time = time.time()

    centroids = []
    labels = []

    for digit in range(10):
        digit_features = X_train[y_train == digit]
        kmeans = KMeans(n_clusters=n_clusters_per_class, random_state=42, n_init="auto").fit(digit_features)
        centroids.append(kmeans.cluster_centers_)
        labels.extend([digit] * n_clusters_per_class)

    centroids = np.vstack(centroids)
    labels = np.array(labels)

    predictions = []
    for x in X_test:
        dists = np.linalg.norm(centroids - x, axis=1)
        pred = labels[np.argmin(dists)]
        predictions.append(pred)

    predictions = np.array(predictions)

    total_time = time.time() - start_time
    accuracy = accuracy_score(y_test, predictions) * 100

    return predictions, accuracy, total_time

# ==============================
# Step 4: SVM Classifier
# ==============================
def svm_classifier(X_train, y_train, X_test, y_test, kernel_type='linear'):
    start_time = time.time()

    clf = SVC(kernel=kernel_type, gamma='scale')
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    total_time = time.time() - start_time
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

# ===== Autoencoder =====
print("\nTraining Autoencoder...")
autoencoder, encoder = build_autoencoder(latent_dim=64)

autoencoder.fit(
    X_train, X_train,
    epochs=20,
    batch_size=32,
    shuffle=True,
    verbose=1
)

print("\nExtracting Autoencoder features...")
X_train_ae = encoder.predict(X_train)
X_test_ae = encoder.predict(X_test)

# ==============================
# K-Means
# ==============================
n_clusters_list = [1, 4, 16, 32]
kmeans_results = []

for n_clusters in n_clusters_list:
    print(f"\nK-Means with {n_clusters} clusters...")
    preds, acc, t = kmeans_classifier(X_train_ae, y_train, X_test_ae, y_test, n_clusters)
    kmeans_results.append({'clusters': n_clusters, 'accuracy': acc, 'time': t, 'predictions': preds})
    print(f"Accuracy: {acc:.2f}% | Time: {t:.2f} sec")

best_kmeans = max(kmeans_results, key=lambda x: x['accuracy'])

# ==============================
# SVM
# ==============================
svm_kernels = ['linear', 'rbf']
svm_results = []

for kernel in svm_kernels:
    print(f"\nSVM with {kernel} kernel...")
    preds, acc, t = svm_classifier(X_train_ae, y_train, X_test_ae, y_test, kernel)
    svm_results.append({'kernel': kernel, 'accuracy': acc, 'time': t, 'predictions': preds})
    print(f"Accuracy: {acc:.2f}% | Time: {t:.2f} sec")

best_svm = max(svm_results, key=lambda x: x['accuracy'])

# ==============================
# MLP
# ==============================
mlp_configs = {
    "1-hidden": (128,),
    "3-hidden": (128, 64, 32),
    "4-hidden": (256, 128, 64, 32)
}

mlp_results = []

for name, layers in mlp_configs.items():
    print(f"\nMLP ({name})...")
    preds, acc, t = mlp_classifier(X_train_ae, y_train, X_test_ae, y_test, layers)
    
    mlp_results.append({
        'config': name,
        'accuracy': acc,
        'time': t,
        'predictions': preds
    })

    print(f"Accuracy: {acc:.2f}% | Time: {t:.2f} sec")

best_mlp = max(mlp_results, key=lambda x: x['accuracy'])

# ==============================
# Confusion Matrices
# ==============================
plt.figure(figsize=(18,5))

# KMeans
cm = confusion_matrix(y_test, best_kmeans['predictions'])
plt.subplot(1,3,1)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("KMeans")

# SVM
cm = confusion_matrix(y_test, best_svm['predictions'])
plt.subplot(1,3,2)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("SVM")

# MLP
cm = confusion_matrix(y_test, best_mlp['predictions'])
plt.subplot(1,3,3)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("MLP")

plt.tight_layout()
plt.show()

# ==============================
# Summary Table
# ==============================
print("\nSummary Table (Autoencoder Features):")
print("Classifier     | Parameter       | Accuracy (%) | Time (s)")
print("-----------------------------------------------------------")

for res in kmeans_results:
    print(f"K-Means       | {res['clusters']} clusters  | {res['accuracy']:11.2f} | {res['time']:10.2f}")

for res in svm_results:
    print(f"SVM           | {res['kernel']} kernel    | {res['accuracy']:11.2f} | {res['time']:10.2f}")

for res in mlp_results:
    print(f"MLP           | {res['config']:12} | {res['accuracy']:11.2f} | {res['time']:10.2f}")