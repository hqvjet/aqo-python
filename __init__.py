import pandas as pd
import torch
import os
from preprocessing import HandleData
from modelling import Trainer, ModelLibrabry

def mkdir():
    if not os.path.exists('resources/models/GRU'):
        os.makedirs('resources/models/GRU')

# Check folder
mkdir()

# Get model list in folder
model_names = [f[:-3] for f in os.listdir('modelling') if f.endswith('.py')]
model_names.remove('trainer')
model_names.remove('__init__')
model_names.remove('model_library')

# Get data
data = pd.read_csv('resources/dataset.csv')

# Get model list
model_list = ['gru'] # if there are any new model, add here
model_lib = ModelLibrabry(model_list)

# Preprocessing
handle_data = HandleData(data).data
data = handle_data.values.tolist()

# Modelling
for row in data:
    fs_hash = row[0]
    fss_hash = row[1]
    nfeatures = row[2]
    features = torch.tensor(row[3])
    targets = torch.tensor(row[4])

    print('Which model do you wanna use ? <type model name correctly>')
    for i, name in enumerate(model_names):
        print(f'{i+1}. {name}')

    key = input()

    model = model_lib.select_model(model_name=key, nfeatures=nfeatures)

    trainer = Trainer(model=model, fs_hash=fs_hash, fss_hash=fss_hash)
    trainer.fit(features, targets)
