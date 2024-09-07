import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from SCADA_data import T1Dataset
from Model.RNNs import LSTMModule
from utility import weights_init

batch_size = 32
learning_rate = 0.001
num_epochs = 100
save_dir = 'train_models'

if __name__ == "__main__":
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    train_dataset = T1Dataset("archive/train_data.csv")
    val_dataset = T1Dataset("archive/val_data.csv")
    test_dataset = T1Dataset("archive/test_data.csv")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = train_dataset.data.shape[1]
    output_dim = 1
    
    model = LSTMModule(input_dim, output_dim)
    model.cuda()
    model.apply(weights_init)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # train
        for data in train_loader:
            inputs, targets = data
            inputs = torch.tensor(inputs).cuda() 
            targets = torch.tensor(targets).cuda() 
            
            optimizer.zero_grad()
    
            outputs = model(inputs)
    
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()

        # validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data in val_loader:
                inputs, targets = data
                inputs = torch.tensor(inputs).cuda() 
                targets = torch.tensor(targets).cuda() 
                
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))

                val_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)

        epoch_val_loss = val_loss / len(val_loader)        
        val_loss_history.append(epoch_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")
        
        torch.save(model.state_dict(), os.path.join(save_dir,f"model_epoch_{epoch+1}"))
        
        if epoch == 0 or epoch_val_loss < min(val_loss_history[:-1]):
            torch.save(model.state_dict(), os.path.join(save_dir,"best_model"))

    np.save(os.path.join(save_dir,'loss_history.npy'), np.array(loss_history))
    np.save(os.path.join(save_dir,'val_history.npy'), np.array(val_loss_history))
    
    plt.plot(loss_history, "Train Loss")
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('training_loss.png')
    plt.show()