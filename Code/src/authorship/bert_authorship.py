# Ben Kabongo
# Personnalized data-to-text neural generation
# ISIR/MLIA, 2023
# Crédits : https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f

# Authorship attribution with BERT


import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


def load_data(n_authors=2, data_path='./users_output.csv'):
    users_output_df = pd.read_csv(data_path, index_col=0)
    users_output_df = users_output_df.rename({'target': 'review'}, axis=1)
    users_output_df = users_output_df[['review', 'movieID', 'userID']]
    users_output_df = users_output_df[users_output_df['userID'] != 'u/nan']
    users_output_df = users_output_df.dropna()

    occurrences = users_output_df['userID'].value_counts()[:n_authors]
    mask = users_output_df.userID.isin(occurrences.index)
    df = users_output_df[mask]
    
    sample_size = occurrences.min()
    train_dfs, test_dfs, eval_dfs = [], [], []

    for ui, u in enumerate(df.userID.unique()):
        user_df = df[df['userID'] == u]
        sample_df = user_df.sample(n=sample_size, replace=True)
        sample_df['userNum'] = [ui]*sample_size
        
        train_df = sample_df.sample(frac=.8)
        test_eval_df = sample_df.drop(train_df.index).reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)
        test_df = test_eval_df.sample(frac=.5)
        eval_df = test_eval_df.drop(test_df.index).reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        train_dfs.append(train_df)
        test_dfs.append(test_df)
        eval_dfs.append(eval_df)

    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)
    eval_df = pd.concat(eval_dfs)
        
    return train_df, test_df, eval_df


class AuthorshipDataset(Dataset):

    def __init__(self, data_df, tokenizer):
        self.data_df = data_df
        self.tokenizer = tokenizer

        def tokenize(text):
            return self.tokenizer(
                text, 
                padding='max_length', 
                max_length = 512, 
                truncation=True, 
                return_tensors="pt"
            )
        self.labels = data_df.userNum.tolist()
        self.texts = data_df.review.apply(tokenize).tolist()

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class BertClassifier(nn.Module):

    def __init__(self, n_authors=80, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, n_authors)
        self.relu = nn.ReLU()
        self._n_authors = n_authors

    def forward(self, input_id, mask):
        _, cls_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(cls_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


def train(model, train_df, eval_df, lr=1e-6, epochs=5, batch_size=8, model_path='bert_authorship.pt'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    train_dt = AuthorshipDataset(train_df, tokenizer)
    eval_dt = AuthorshipDataset(eval_df, tokenizer)

    train_ld = DataLoader(train_dt, batch_size=batch_size, shuffle=True)
    eval_ld = DataLoader(eval_dt, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        device = torch.device("cpu")
    print(f'[Device] Device :\t {device}\n')

    print('Training...\n')
    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_ld, f'[Train Epoch {epoch + 1}/{epochs}]'):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            loss = criterion(output, train_label.long())
            total_loss_train += loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            loss.backward()
            optimizer.step()

        total_acc_eval = 0
        total_loss_eval = 0

        with torch.no_grad():
            for eval_input, eval_label in tqdm(eval_ld, "[Validation]"):
                eval_label = eval_label.to(device)
                mask = eval_input['attention_mask'].to(device)
                input_id = eval_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                loss = criterion(output, eval_label.long())
                total_loss_eval += loss.item()

                acc = (output.argmax(dim=1) == eval_label).sum().item()
                total_acc_eval += acc
            
        print(f'[Training] Epochs: {epoch + 1} ' +
            f'| Train Loss: {total_loss_train / len(train_dt): .3f} ' +
            f'| Train Accuracy: {total_acc_train / len(train_dt): .3f} ' +
            f'| Val Loss: {total_loss_eval / len(eval_dt): .3f} ' +
            f'| Val Accuracy: {total_acc_eval / len(eval_dt): .3f}')

        torch.save(model.state_dict(), model_path)
        print(f'[Model] Saving model at {model_path}')


def evaluate(model, test_df, batch_size=8):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    test_dt = AuthorshipDataset(test_df, tokenizer)
    test_ld = DataLoader(test_dt, batch_size=batch_size)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.cuda()
    else:
        device = torch.device("cpu")

    total_acc_test = 0
    print('Test...')
    with torch.no_grad():

        for test_input, test_label in tqdm(test_ld, '[Test]'):
              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
    
    print(f'[Test] Test Accuracy: {total_acc_test / len(test_dt): .3f}')


def main(args):
    lr = 1e-6
    epochs = args.epochs
    batch_size = args.batch_size
    n_authors = args.n_authors
    model_path = args.model_path
    data_path = args.data_path
    evaluation = args.evaluation

    model = BertClassifier(n_authors=n_authors)

    if model_path != '.':
        model.load_state_dict(torch.load(model_path))
        print(f'[Model] Loading model from {model_path}')
    else:
        model_path = f'bert_authorship_{model._n_authors}.pt'

    if not evaluation:
        train_df, test_df, eval_df = load_data(n_authors, data_path)
        train(model, train_df, eval_df, lr, epochs, batch_size, model_path)
        torch.save(model.state_dict(), model_path)
        print(f'[Model] Saving model at {model_path}')
        evaluate(model, test_df, batch_size)
    else:
        data_df = pd.read_csv(data_path, index_col=0)
        evaluate(model, data_df, batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--n_authors', default=2, type=int)
    parser.add_argument('--model_path', default='.', type=str)
    parser.add_argument('--data_path', default='./', type=str)
    parser.add_argument('--evaluation', default=False, type=bool)
    args = parser.parse_args()
    main(args)
