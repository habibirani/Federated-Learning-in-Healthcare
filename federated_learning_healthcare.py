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
model_avg_params = (model.fc.weight.data + model.copy().send(hospital1).get().fc.weight.data) / 2
model_avg_bias = (model.fc.bias.data + model.copy().send(hospital1).get().fc.bias.data) / 2

# Update the model with averaged parameters
model_avg = FederatedModel()
model_avg.fc.weight.data = model_avg_params
model_avg.fc.bias.data = model_avg_bias

# Get the final model from any hospital (e.g., hospital1)
final_model = model_avg.get()

# For simplicity, let's use the same synthetic data from hospital1 as the test data
test_data = torch.tensor(hospital1_data)
test_labels = torch.tensor(hospital1_labels)

# Evaluate the federated model on the test data
with torch.no_grad():
    model_avg.eval()  # Set the model to evaluation mode
    predictions = model_avg(test_data.float()).round().squeeze().detach().numpy()

# Calculate accuracy
accuracy = (predictions == test_labels.numpy()).mean()
print("Accuracy:", accuracy)

