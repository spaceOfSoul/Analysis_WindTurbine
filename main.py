import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from SCADA_data import T1Dataset
from Model.RNNs import LSTMModule


batch_size = 32
learning_rate = 0.001
num_epochs = 100
save_dir = 'train_models'

if __name__ == "__main__":
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    train_dataset = T1Dataset("archive/T1.csv")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = train_dataset.data.shape[1]
    output_dim = 1
    model = LSTMModule(input_dim, output_dim)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []

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
    
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        torch.save(model.state_dict(), os.join(save_dir,f"model_epoch_{epoch+1}"))
        
        if epoch == 0 or epoch_loss < min(loss_history[:-1]):
            torch.save(model.state_dict(), os.join(save_dir,"best_model"))

    np.save(os.join(save_dir,'loss_history.npy'), np.array(loss_history))
    
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('training_loss.png')
    plt.show()