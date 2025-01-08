import pandas as pd
import torch
import os
from preprocessing import HandleData
from modelling import GRU, Trainer

def mkdir():
    if not os.path.exists('resources/models/GRU'):
        os.makedirs('resources/models/GRU')

# Check folder
mkdir()

# Get data
data = pd.read_csv('resources/dataset.csv')

# Preprocessing
handle_data = HandleData(data).data

# Modelling
data = handle_data.values.tolist()

for row in data:
    fs_hash = row[0]
    fss_hash = row[1]
    nfeatures = row[2]
    features = torch.tensor(row[3])
    targets = torch.tensor(row[4])

    gru = GRU(input_size=nfeatures, hidden_size=16)
    trainer = Trainer(model=gru, fs_hash=fs_hash, fss_hash=fss_hash)
    trainer.fit(features, targets)
