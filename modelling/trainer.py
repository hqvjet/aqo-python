import torch
import torch.nn as nn

class Trainer():
    def __init__(self, epoch=1000, lr=0.001, batch_size=32, fs_hash=None, fss_hash=None, model=None):
        self.model = model
        self.loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.epochs = epoch
        self.lr = lr
        self.batch_size = batch_size
        self.fsh = fs_hash
        self.fssh = fss_hash

    def fit(self, features, targets):
        self.model.train()
        max_loss = 1000000000

        print(f'This space has {features.size()[0]} datapoints')

        for epoch in range(self.epochs):
            self.opt.zero_grad()
            output = self.model(features)
            loss = self.loss(output, targets)
            loss.backward()
            self.opt.step()
            
            if loss < max_loss:
                max_loss = loss
                torch.save(list(self.model.parameters()), f'resources/models/{self.model.name}/{self.fsh}_{self.fssh}.pt')

            if (epoch+1) % 250 == 0:
                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}')



