import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_model(model, train_loader, test_loader,
                epochs=20, lr=0.001, description="Model"):

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    print(f"\n{'='*50}")
    print(f"TRAINING: {description}")
    print(f"{'='*50}")

    # ── TRAINING ──
    train_start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1:2d}/{epochs}] "
                  f"Loss: {avg_loss:.4f}")

    train_time = (time.time() - train_start) * 1000
    print(f"\nTraining time: {train_time:.1f} ms")

    # ── TESTING ──
    test_start = time.time()
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds   = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    test_time = (time.time() - test_start) * 1000
    accuracy  = accuracy_score(all_labels, all_preds) * 100

    print(f"Testing time : {test_time:.1f} ms")
    print(f"Accuracy     : {accuracy:.1f}%")

    return accuracy, train_time, test_time, train_losses

if __name__ == "__main__":
    print("✅ Step 4 Complete — Training function ready!")