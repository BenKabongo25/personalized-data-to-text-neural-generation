import numpy as np
import os
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class RatingDataset(Dataset):

    def __init__(self, data_df, users, movies):
        super().__init__()
        self.data_df = data_df
        self.users = data_df.userID.tolist()
        self.movies = data_df.movieID.tolist()
        self.ratings = data_df.rating.tolist()

        u_label_encoder = LabelEncoder()
        m_label_encoder = LabelEncoder()
        u_label_encoder.fit(users)
        m_label_encoder.fit(movies)

        self.users_ = u_label_encoder.transform(self.users)
        self.movies_ = m_label_encoder.transform(self.movies)


    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users_[idx], self.movies_[idx], self.ratings[idx]


class RecommenderNetwork(nn.Module):

    def __init__(
                self, 
                n_users, 
                n_movies,
                n_features=150, 
                um_dropout_p=0.02,
                hidden_sizes=[100, 200, 300],
                dropouts_p=[0.25, 0.50]
        ):
        
        super().__init__()
        assert len(dropouts_p) >= len(hidden_sizes)-1

        hidden_sizes.insert(0, 2 * n_features)
        layers = []
        for i in range(1, len(hidden_sizes)):
            l = nn.Linear(hidden_sizes[i-1], hidden_sizes[i])
            torch.nn.init.xavier_uniform_(l.weight)
            l.bias.data.fill_(0.01)

            layers.extend([l, nn.ReLU()])
            if i != len(hidden_sizes)-1:
                layers.append(nn.Dropout(dropouts_p[i-1]))

        self.u = nn.Embedding(n_users, n_features)
        self.m = nn.Embedding(n_movies, n_features)
        self.drop = nn.Dropout(um_dropout_p)
        self.hidden = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_sizes[-1], 1)

        self.u.weight.data.uniform_(-0.05, 0.05)
        self.m.weight.data.uniform_(-0.05, 0.05)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.01)

        
    def forward(self, users, movies):
        features = torch.cat([self.u(users), self.m(movies)], dim=1)
        x = self.drop(features)
        x = self.hidden(x)
        out = torch.sigmoid(self.fc(x))
        out = 6 * out  - 0.5
        return out
    


def train_loop(dataloader, model, loss_fn, optimizer, device):
    running_loss = .0
    size = len(dataloader.dataset)

    model.train()
    for batch, (U, M, R) in enumerate(dataloader):
        U, M, R = U.to(device).int(), M.to(device).int(), R.to(device).float()
        R_pred = model(U, M)
        loss = loss_fn(R_pred, R)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(R)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return running_loss / size
    

def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for (U, M, R) in dataloader:
            U, M, R = U.to(device).int(), M.to(device).int(), R.to(device).float()
            R_pred = model(U, M)
            test_loss += loss_fn(R_pred, R).item()
            correct += (R_pred.argmax(1) == R).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


def Trainer(
        model,
        device,
        users,
        movies,
        train_df,
        test_df,
        lr,
        weight_decay,
        n_epochs,
        batch_size,
        output_dir='./',
    ):
    test_every_epoch = max(1, n_epochs // 10)
    writer = SummaryWriter(os.path.join(output_dir, 'runs'))

    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = RatingDataset(train_df, users, movies)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataset = RatingDataset(test_df, users, movies)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    for epoch in range(n_epochs):
        running_loss = train_loop(train_dataloader, model, loss_fn, optimizer, device)
        writer.add_scalar('Loss/train', running_loss, epoch)
        if epoch % test_every_epoch == 0:
            test_loss, correct = test_loop(test_dataloader, model, loss_fn, device)
            writer.add_scalar('Loss/test', test_loss, epoch//10)
            writer.add_scalar('Accuracy/test', correct, epoch//10)
    
    torch.save(model.state_dict(), os.path.join(output_dir, 'model_weights.pth'))


def main():
    print('[Recommender] START')
    seed = 42
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    lr = 1e-3
    weight_decay = 1e-5
    batch_size = 64
    n_epochs = 100
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[Device] Device : {device}')

    print('[Loading data]')
    df = pd.read_csv('users_output.csv', index_col=0)[['movieID', 'userID', 'rating']].dropna()
    users = df.userID.unique().tolist()
    movies = df.movieID.unique().tolist()
    n_users = len(users)
    n_movies = len(movies)

    train_df = df.sample(frac=0.6, replace=True)
    test_df = df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    print(f'[Data] Train shape \t: {train_df.shape}')
    print(f'[Data] Test shape \t: {test_df.shape}')

    print('\n[Data]')
    print(df.head())

    model = RecommenderNetwork(n_users, n_movies)
    print('\n[Training] ...')
    Trainer(model, device, users, movies, train_df, test_df, lr, weight_decay, n_epochs, batch_size)

    print('\n[Recommender] END')

main()