import torch
import torch.nn as nn

class Trainer():
    def __init__(self, epoch=1000, lr=0.001, batch_size=32, fs_hash=None, fss_hash=None, model=None, isML=True):
        self.model = model
        self.loss = nn.L1Loss()
        self.isML = isML
        if not isML:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.epochs = epoch
            self.lr = lr
        self.fsh = fs_hash
        self.fssh = fss_hash

    def fit(self, features, targets):
        self.model.train()
        max_loss = 1000000000

        data = torch.cat((features, targets.unsqueeze(1)), axis=1)
        unique_data = torch.unique(data, dim=0)

        unique_features = unique_data[:, :-1]
        unique_targets = unique_data[:, -1]

        print(f'This space has {data.size()[0]} datapoints, {unique_data.size()[0]} unique datapoints, fs: {self.fsh}, fss: {self.fssh}')

        if not self.isML:
            for epoch in range(self.epochs):
                self.opt.zero_grad()
                output = self.model(unique_features)
                loss = self.loss(output, unique_targets)
                loss.backward()
                self.opt.step()

                if loss < max_loss:
                    max_loss = loss
                    torch.save(self.model.state_dict(), f'resources/models/{self.model.name}/{self.fsh}_{self.fssh}.pt')

                if (epoch+1) % 500 == 0:
                    print(f'Epoch {epoch+1}/{self.epochs}, Global MAE Loss: {loss.item()}')

        else:
            output = self.model(unique_features, unique_targets)
            model = self.model.model
            model.save_model(f'resources/models/{self.model.name}/{self.fsh}_{self.fssh}.json')
            output = torch.tensor(output)
            loss = self.loss(output, unique_targets)
            print(f'Global MAE Loss: {loss.item()}')

