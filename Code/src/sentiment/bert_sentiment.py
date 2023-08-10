# Ben Kabongo
# Personalized data-to-text neural generation
# ISIR/MLIA, 2023
# Crédits : https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f

# Sentiment analysis with BERT

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


def binarize(n, s=2):
    return 1 if n > s else 0


def load_data(data_path='../../../Data/appdata/rotten/review_to_rating.csv'):
    data_df = pd.read_csv(data_path, index_col=0)

    data_df = data_df[['input', 'output']].rename({'input': 'text', 'output': 'label'}, axis=1)
    data_df['text'] = data_df['text'].apply(str)
    data_df['label_id'] = data_df['label'].apply(binarize)

    train_size = 0.75
    train_df = data_df.sample(frac=train_size, random_state=42)
    test_df = data_df.drop(train_df.index).reset_index(drop=True)

    return train_df, test_df


class SentimentDataset(Dataset):

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
        self.labels = data_df.label_id.tolist()
        self.texts = data_df.text.apply(tokenize).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, cls_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(cls_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


def train(model, train_df, eval_df, lr=1e-6, epochs=5, batch_size=8):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    train_dt = SentimentDataset(train_df, tokenizer)
    eval_dt = SentimentDataset(eval_df, tokenizer)

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

        torch.save(model.state_dict(), 'bert_sentiment.pt')
        print('[Model] Saving model at bert_sentiment.pt')


def evaluate(model, test_df, batch_size=8):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    test_dt = SentimentDataset(test_df, tokenizer)
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


def main():
    lr=1e-6
    epochs=5
    batch_size=8
    (train_df, test_df) = load_data()
    model = BertClassifier()
    #model.load_state_dict(torch.load('bert_sentiment.pt'))
    #print('[Model] Loading model from bert_sentiment.pt')
    train(model, train_df, test_df, lr, epochs, batch_size)
    torch.save(model.state_dict(), 'bert_sentiment.pt')
    print('[Model] Saving model at bert_sentiment.pt')
    evaluate(model, test_df, batch_size)


if __name__ == '__main__':
    main()
