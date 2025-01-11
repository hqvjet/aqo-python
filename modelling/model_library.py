from modelling.gru import GRU
from modelling.xgboost import XGB

class ModelLibrabry:
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
        # elif .....
        #   return .....

        # If the name not in model list
        else:
            print('Name not in model list, please confirm that the input is correct')
            return None

