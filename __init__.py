import pandas as pd
import torch
import matplotlib.pyplot as plt
import math
import os
from preprocessing import HandleData
from modelling import Trainer, Tester, ModelLibrary

def mkdir():
    if not os.path.exists('resources/models'):
        os.makedirs('resources/models')

    if not os.path.exists('resources/models/GRU'):
        os.makedirs('resources/models/GRU')

    if not os.path.exists('resources/models/XGB'):
        os.makedirs('resources/models/XGB')

    if not os.path.exists('resources/models/LSTM'):
        os.makedirs('resources/models/LSTM')

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
model_list = ['gru', 'xgboost', 'lstm'] # if there are any new model, add here
DL_model = ['gru', 'lstm']
model_lib = ModelLibrary(model_list)

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

        threshold = math.ceil(len(row[3]) * 0.5)
        features = torch.tensor(row[3][:threshold])
        targets = torch.tensor(row[4][:threshold])
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
    statistic = {
        'fs': [],
        'fss': [],
        'loss': []
    }

    for row in handle_data:
        fs_hash = row[0]
        fss_hash = row[1]
        nfeatures = row[2]

        if len(row[3]) == 1:
            threshold = 0
        else:
            threshold = math.ceil(len(row[3]) * 0.5)

        trained_features = torch.tensor(row[3][:threshold])
        trained_targets = torch.tensor(row[4][:threshold])
        features = torch.tensor(row[3][threshold:])
        targets = torch.tensor(row[4][threshold:])
        isML = True

        if key in DL_model:
            isML=False

        model = model_lib.select_pretrained_model(model_name=key, fs=fs_hash, fss=fss_hash, nfeatures=nfeatures)

        tester = Tester(model=model, fs_hash=fs_hash, fss_hash=fss_hash, isML=isML)
        loss = tester.test(features, targets, trained_features, trained_targets)

        statistic['fs'].append(fs_hash)
        statistic['fss'].append(fss_hash)
        statistic['loss'].append(loss)

    statistic = pd.DataFrame(statistic)
    statistic = statistic.groupby('fs', as_index=False)['loss'].mean()
    print(statistic)
    statistic.to_csv(f'resources/statistic.csv', index=False)


def visualize():
    def normalize(data):
        result = {
            'fs': [],
            'loss': []
        }

        for row in data:
            result['fs'].append(row[0])
            last_loss = [float(i) for i in row[5][1:-1].split(',')][-1]
            result['loss'].append(last_loss)

        return pd.DataFrame(result)

    stat = pd.read_csv('resources/statistic.csv')
    res = pd.read_csv('resources/result.csv')
    res = normalize(res.values)
    stat = pd.merge(stat, res, on='fs', suffixes=('_xgb', '_knn'))

    plt.figure(figsize=(15,8))
    plt.plot(stat['fs'].apply(str), stat['loss_xgb'], color='red', marker='o', markersize=3, label='MAE on XGBoost')
    plt.plot(stat['fs'].apply(str), stat['loss_knn'], color='blue', marker='s', markersize=3, label='MAE on Online 3NN')
    plt.xticks(fontsize=8, rotation=90)
    plt.title('Comparision of Loss between XGBoost and Online 3NN')
    plt.xlabel('Feature Space Hash')
    plt.ylabel('MAE Loss on the last iter')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('resources/chart.png')

print("What do you wanna do ?<train/test/visualize>")
key = input()

if key == 'train':
    train()
elif key == 'test':
    eval()
elif key == 'visualize':
    visualize()
else:
    print('Wrong command, try again')
