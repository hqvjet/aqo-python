from modelling.gru import GRU
from modelling.xgboost import XGB
from modelling.lstm import LSTM
import torch

class ModelLibrary:
    def __init__(self, model_list:str=None):
        self.model_list = model_list

    def select_model(self, model_name=None, nfeatures=None):
        '''
            Get model by name
            Init model here if you have any new model following the rule below
        '''
        if model_name == self.model_list[0]:
            return GRU(input_size=nfeatures, hidden_size=16)

        elif model_name == self.model_list[1]:
            return XGB()

        elif model_name == self.model_list[2]:
            return LSTM(input_size=nfeatures, hidden_size=16)

        # elif .....
        #   return .....

        # If the name not in model list
        else:
            print('Name not in model list, please confirm that the input is correct')
            return None

    def select_pretrained_model(self, model_name=None, fs=None, fss=None, nfeatures=None):
        '''
            Get pretrained model by name
            Init model here if you have any new model following the rule below
        '''
        if model_name == self.model_list[0]:
            model = GRU(input_size=nfeatures, hidden_size=16)
            return model.load_state_dict(torch.load(f'resources/models/GRU/{fs}_{fss}.pt'))

        elif model_name == self.model_list[1]:
            model = XGB()
            model.load_model(f'resources/models/XGB/{fs}_{fss}.json')
            return model

        if model_name == self.model_list[2]:
            model = LSTM(input_size=nfeatures, hidden_size=16)
            return model.load_state_dict(torch.load(f'resources/models/LSTM/{fs}_{fss}.pt'))

        # elif .....
        #   return .....

        # If the name not in model list
        else:
            print('Name not in model list, please confirm that the input is correct')
            return None

