import pandas as pd
import torch
import os
from preprocessing import HandleData
from modelling import Trainer, Tester, ModelLibrabry

def mkdir():
    if not os.path.exists('resources/models'):
        os.makedirs('resources/models')

    if not os.path.exists('resources/models/GRU'):
        os.makedirs('resources/models/GRU')

    if not os.path.exists('resources/models/XGB'):
        os.makedirs('resources/models/XGB')

# Check folder
mkdir()

# Get model list in folder
model_names = [f[:-3] for f in os.listdir('modelling') if f.endswith('.py')]
model_names.remove('trainer')
model_names.remove('tester')
model_names.remove('__init__')
model_names.remove('model_library')

# Get data
data = pd.read_csv('resources/dataset.csv')

# Get model list
model_list = ['gru', 'xgboost'] # if there are any new model, add here
DL_model = ['gru']
model_lib = ModelLibrabry(model_list)

# Preprocessing
handle_data = HandleData(data).data

def train():
    print('Which model do you wanna train ? <type model name correctly>')
    for i, name in enumerate(model_names):
        print(f'{i+1}. {name}')

    key = input()

    # Modelling
    for row in handle_data:
        fs_hash = row[0]
        fss_hash = row[1]
        nfeatures = row[2]
        features = torch.tensor(row[3])
        targets = torch.tensor(row[4])
        isML = True

        if key in DL_model:
            isML=False

        model = model_lib.select_model(model_name=key, nfeatures=nfeatures)

        trainer = Trainer(model=model, fs_hash=fs_hash, fss_hash=fss_hash, isML=isML)
        trainer.fit(features, targets)

def eval():
    print('Which model do you wanna test ? <type model name correctly>')
    for i, name in enumerate(model_names):
        print(f'{i+1}. {name}')

    key = input()

    for row in handle_data:
        fs_hash = row[0]
        fss_hash = row[1]
        nfeatures = row[2]
        features = torch.tensor(row[3])
        targets = torch.tensor(row[4])
        isML = True

        if key in DL_model:
            isML=False

        model = model_lib.select_pretrained_model(model_name=key, fs=fs_hash, fss=fss_hash, nfeatures=nfeatures)

        tester = Tester(model=model, fs_hash=fs_hash, fss_hash=fss_hash, isML=isML)
        tester.test(features, targets)

print("What do you wanna do ?<train/test>")
key = input()

if key == 'train':
    train()
elif key == 'test':
    eval()
