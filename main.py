import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from SCADA_data import T1Dataset
from Model.RNNs import LSTMModule


batch_size = 32
learning_rate = 0.001
num_epochs = 100

train_dataset = T1Dataset("archive/T1.csv")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

input_dim = train_dataset.data.shape[1]
output_dim = 1
model = LSTMModule(input_dim, output_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

input_dim = train_dataset.data.shape[1]
output_dim = 1 
model = LSTMModule(input_dim, output_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for data in train_loader:
        inputs, targets = data

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

