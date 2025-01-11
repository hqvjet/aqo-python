import torch
import torch.nn as nn
import xgboost as xgb

class XGB(nn.Module):
    def __init__(self, output_size=1):
        super(XGB, self).__init__()
        self.name = 'XGB'
        self.params = {
                "objective": "reg:squarederror",
                "max_depth": 8,
                "eta": 0.01,
                "subsample": 0.9,
                "colsample_bytree": 1,
                "tree_method": "hist",
                "device": "cuda",
        }
        self.num_rounds = 100
    
    def forward(self, x, y=None):
        dtrain = xgb.DMatrix(x, label=y)
        model = xgb.train(self.params, dtrain, num_boost_round=self.num_rounds)
        predict = model.predict(xgb.DMatrix(x))
        return model, predict
