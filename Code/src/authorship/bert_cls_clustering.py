# Ben Kabongo
# Personalized data-to-text neural generation
# ISIR/MLIA, 2023

# Clustering with BERT

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.manifold import TSNE
from torch import nn
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
    sample_dfs = []

    for ui, u in enumerate(df.userID.unique()):
        user_df = df[df['userID'] == u]
        sample_df = user_df.sample(n=sample_size, replace=True)
        sample_df['userNum'] = [ui]*sample_size
        sample_dfs.append(sample_df)

    data_df = pd.concat(sample_dfs)
    return data_df


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

    
    def get_cls(self, input_id, mask):
        _, cls_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        return cls_output


def visualize(args):
    data_df = load_data(args.n_authors, args.data_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    data_dt = AuthorshipDataset(data_df, tokenizer)
    data_ld = DataLoader(data_dt, batch_size=args.batch_size)
    print('[Data] Data loaded')
    
    model = BertClassifier(n_authors=args.n_authors)
    model.load_state_dict(torch.load(args.model_path))
    print(f'[Model] Loading model from {args.model_path}')

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.cuda()
    else:
        device = torch.device("cpu")

    cls_outputs = []
    labels = []

    with torch.no_grad():
        for data_input, data_label in tqdm(data_ld):
            data_label = data_label.to(device)
            mask = data_input['attention_mask'].to(device)
            input_id = data_input['input_ids'].squeeze(1).to(device)
            cls_output = model.get_cls(input_id, mask).tolist()
            cls_outputs.extend(cls_output)
            labels.extend(data_label.tolist())

    cls_outputs = np.array(cls_outputs)
    labels = np.array(labels)

    n_authors = args.n_authors

    kmeans = KMeans(n_clusters=n_authors)
    _ = kmeans.fit(cls_outputs)
    predictions = kmeans.predict(cls_outputs)
    ari = adjusted_rand_score(labels, predictions)
    print(f'[Clustering] KMeans ARI = {ari}')

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_data = tsne.fit_transform(cls_outputs)

    plt.figure()
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels)
    plt.savefig(args.out_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--n_authors', default=2, type=int)
    parser.add_argument('--data_path', default='./users_output.csv', type=str)
    parser.add_argument('--out_filename', default='clustering_authors.png', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    args = parser.parse_args()

    visualize(args)
