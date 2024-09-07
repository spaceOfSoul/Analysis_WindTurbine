import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class T1Dataset(Dataset):
    def __init__(self, csv_file, input_dim=4):
        self.input_dim = input_dim

        data = pd.read_csv(csv_file)

        self.data = data[['LV ActivePower (kW)', 'Wind Speed (m/s)', 
                          'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']].values

        self.data = np.nan_to_num(self.data)

        # z 정규화
        self.scaler = StandardScaler()
        self.data_scaled = self.scaler.fit_transform(self.data)

    def __len__(self):
        return len(self.data_scaled)

    def __getitem__(self, idx):
        weather_data = self.data_scaled[idx, :]
        power_data = weather_data[2]

        weather_data = torch.tensor(weather_data, dtype=torch.float32)
        power_data = torch.tensor(power_data, dtype=torch.float32)

        return weather_data, power_data