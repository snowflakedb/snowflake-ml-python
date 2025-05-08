from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)


class IrisNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x) -> None:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model_name: Optional[str] = None) -> Any:
    # load data
    dataset = load_iris()
    X_train, _, y_train, _ = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=42, shuffle=False
    )
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = IrisNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 2
    for _ in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    return model


def predict_result(model) -> Any:
    dataset = load_iris()
    _, X_test, _, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=42)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    final_y_pred = []
    model.eval()
    with torch.no_grad():
        for X_batch, _ in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            final_y_pred.extend(predicted.cpu().numpy())

    return final_y_pred


if __name__ == "__main__":
    result = train_model()
    __return__ = result
