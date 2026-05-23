import torch
import torch.nn as nn

from pm.model.model import AudioClassifier


def train_model(train_loader, val_loader=None, num_classes=2, max_epochs=1, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioClassifier(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total
        print(f"epoch {epoch + 1}/{max_epochs} - train_loss {train_loss:.4f} - train_acc {train_acc:.4f}")

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
            print(f"epoch {epoch + 1}/{max_epochs} - val_loss {val_loss:.4f} - val_acc {val_acc:.4f}")

    return model


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total
