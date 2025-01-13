import torch
import torch.nn as nn

class Tester():
    def __init__(self, model=None, fs_hash=None, fss_hash=None, isML=True):
        self.model = model
        self.fsh = fs_hash
        self.fssh = fss_hash
        self.isML = isML
        self.loss = nn.L1Loss()

    def test(self, features, targets):
        if not self.isML:
            self.model.eval()

            with torch.no_grad():
                predict = self.model(features)

            return output

        else:
            predict = self.model(features, targets, train=False)
            predict = torch.tensor(predict)

        loss = self.loss(predict, targets)
        print(f'MAE on fs: {self.fsh}, fss: {self.fssh}: {loss.item()}')
