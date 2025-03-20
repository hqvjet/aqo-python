from sklearn.svm import SVR as SVR_lib
import torch
import torch.nn as nn
import joblib

class SVR(nn.Module):
    def __init__(self):
        super(SVR, self).__init__()
        self.name = 'SVR'
        self.model = SVR_lib(kernel='rbf')

    def forward(self, x, y=None):
        if y is None:
            return self.model.predict(x)
        else:
            self.model.fit(x, y)

        return self.model.predict(x)

    def load_model(self, fs, fss):
        self.model = joblib.load(f'resources/models/{self.name}/{fs}_{fss}.joblib')

    def save_model(self, fs, fss):
        joblib.dump(self.model, f'resources/models/{self.name}/{fs}_{fss}.joblib')
