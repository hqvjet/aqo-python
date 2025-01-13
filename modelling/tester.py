import torch
import torch.nn as nn

def L2_distance_compute(x: torch.Tensor, y: torch.Tensor) -> float:
    dist = torch.dist(x, y, p=2).item()
    return dist

class Tester():
    def __init__(self, model=None, fs_hash=None, fss_hash=None, isML=True):
        self.model = model
        self.fsh = fs_hash
        self.fssh = fss_hash
        self.isML = isML
        self.loss = nn.L1Loss()

    def test(self, features, targets, trained_features, trained_targets):
        if not self.isML:
            self.model.eval()

            with torch.no_grad():
                predict = self.model(features)

            return output

        else:
            predict = []
            added = False
            for i in range(features.size()[0]):
                for j in range(trained_features.size()[0]):
                    if L2_distance_compute(features[i], trained_features[j]) == 0.0:
                        predict.append(trained_targets[j])
                        added = True
                        break
                if not added:
                    predict.append(self.model([features[i]], [targets[i]], train=False))
                added = False

            predict = torch.tensor(predict)

        loss = self.loss(predict, targets)
        print(f'MAE on fs: {self.fsh}, fss: {self.fssh}: {loss.item()}')

        return loss.item()
