# Generate synthetic data (pretending it's medical data)
import torch
import numpy as np

# Generate synthetic data for two hospitals
np.random.seed(0)
hospital1_data = np.random.randn(100, 5)  # 100 samples, 5 features
hospital1_labels = (np.random.randn(100) > 0).astype(int)  # Binary labels for readmission risk

np.random.seed(1)
hospital2_data = np.random.randn(200, 5)  # 200 samples, 5 features
hospital2_labels = (np.random.randn(200) > 0).astype(int)  # Binary labels for readmission risk

# Implement the Federated Learning process
import syft as sy
from torch import nn, optim

# Hook PyTorch to PySyft
hook = sy.TorchHook(torch)

# Create virtual workers representing two hospitals
hospital1 = sy.VirtualWorker(hook, id="hospital1")
hospital2 = sy.VirtualWorker(hook, id="hospital2")

# Share synthetic data with respective hospitals
data_ptr_hospital1 = torch.tensor(hospital1_data).send(hospital1)
labels_ptr_hospital1 = torch.tensor(hospital1_labels).send(hospital1)

data_ptr_hospital2 = torch.tensor(hospital2_data).send(hospital2)
labels_ptr_hospital2 = torch.tensor(hospital2_labels).send(hospital2)

# Define the Federated Learning model (neural network)
class FederatedModel(nn.Module):
    def __init__(self):
        super(FederatedModel, self).__init__()
        self.fc = nn.Linear(5, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# Define the loss function and optimizer
model = FederatedModel()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Federated Learning training loop
epochs = 10

for epoch in range(epochs):
    # Train on hospital1 data
    model.send(hospital1)
    optimizer.zero_grad()
    pred_hospital1 = model(data_ptr_hospital1.float())
    loss_hospital1 = criterion(pred_hospital1.view(-1), labels_ptr_hospital1.float())
    loss_hospital1.backward()
    optimizer.step()
    model.get()

    # Train on hospital2 data
    model.send(hospital2)
    optimizer.zero_grad()
    pred_hospital2 = model(data_ptr_hospital2.float())
    loss_hospital2 = criterion(pred_hospital2.view(-1), labels_ptr_hospital2.float())
    loss_hospital2.backward()
    optimizer.step()
    model.get()

# Aggregate the models (average)
model_avg = (model.get() + model.copy().send(hospital1).get()) / 2

# Get the final model from any hospital (e.g., hospital1)
final_model = model_avg.get()
