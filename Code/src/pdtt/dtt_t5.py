# Ben Kabongo
# Personalized data-to-text neural generation
# ISIR/MLIA, 2023

# Personalized Data-to-text with T5

import ast
import argparse
import json
import numpy as np
import os
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn

from datetime import datetime
from nltk.tokenize import TreebankWordTokenizer
from parent.parent import parent
from style_paraphrase.evaluation.similarity.sim_models import WordAveraging
from style_paraphrase.evaluation.similarity.sim_utils import Example
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertModel
from transformers.optimization import Adafactor


class PDTTDataset(Dataset):
    """ Dataset de data-to-text personnalisé
    inputs    : données semi-structurées
    user_id   : identifiant de l'utilisateur
    targets   : description textuelle personnalisée
    dtt_refs  : description textuelle non personnalisée
    pdtt_refs : descriptions textuelles personnalisées 
    parents   : representation pour la métrique parent
    """

    def __init__(self, data_df, tokenizer, args):
        self.inputs = data_df[args.input_col]
        self.targets = data_df[args.pdtt_refs_col].apply(lambda l: l[-1])
        self.user_ids = data_df[args.user_id_col].tolist()
        self.dtt_refs = data_df[args.dtt_ref_col].tolist()
        self.pdtt_refs = data_df[args.pdtt_refs_col].tolist()
        self.parents = data_df[args.parent_col].tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.inputs[idx],
            self.targets[idx],
            self.user_ids[idx],
            self.dtt_refs[idx],
            self.pdtt_refs[idx],
            self.parents[idx],
        )


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


def save(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


def empty_cache():
    with torch.no_grad(): 
        torch.cuda.empty_cache()


def evaluate_batch(
        model, 
        tokenizer, 
        device, 
        batch, 
        args, 
        authorship_model=None, 
        authorship_tokenizer=None,
        similarity_fn=None
    ):

    empty_cache()
    inputs, targets, user_ids, dtt_refs, pdtt_refs, parents = batch
    tokenized_inputs = tokenizer.batch_encode_plus(
        inputs,
        padding='max_length',
        max_length=args.model_input_length,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = tokenized_inputs["input_ids"].to(device)
    outputs = model.generate(input_ids, max_length=args.model_output_length)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    hypotheses = [h.split(' ') for h in outputs]
    references = [[r.split(' ')] for r in targets]

    #references = [[r.split(' ') for r in p] for p in pdtt_refs]
    parent_tables = [json.loads(p) for p in parents]

    all_similarity = []
    for input, target in zip(inputs, targets):
        similarity = similarity_fn([input], [target])
        similarity = np.mean(similarity)
        all_similarity.append(similarity)
    similarity = np.mean(all_similarity)

    parent_precision, parent_recall, parent_f_score = parent(
        hypotheses,
        references,
        parent_tables,
        avg_results=True,
    )

    acc = 0
    if authorship_model is not None and authorship_tokenizer is not None:
        bert_tokenized_inputs = [
            authorship_tokenizer(
                o, 
                padding='max_length', 
                max_length=512, 
                truncation=True, 
                return_tensors="pt"
            ) for o in outputs
        ]
        attention_mask = torch.cat([item['attention_mask'] for item in bert_tokenized_inputs], dim=0).to(device)
        input_ids = torch.cat([item['input_ids'] for item in bert_tokenized_inputs], dim=0).to(device)
        authorship_outputs = authorship_model(input_ids, attention_mask)
        acc = (authorship_outputs.argmax(dim=1).to(device) == user_ids.to(device)).sum().item()
 
    return {
        'similarity': similarity, 
        'parent': [parent_precision, parent_recall, parent_f_score],
        'authorship': acc,
    }


def evaluate(
        model, 
        tokenizer, 
        device, 
        eval_df, 
        args,
        authorship_model=None,
        authorship_tokenizer=None,
        similarity_fn=None
    ):

    eval_dt = PDTTDataset(eval_df, tokenizer, args)
    eval_ld = DataLoader(eval_dt, batch_size=args.batch_size, shuffle=False)

    similarity, parent_precision, parent_recall, parent_f_score, acc = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(eval_ld, desc="Validation"):
            scores = evaluate_batch(
                model, 
                tokenizer, 
                device, 
                batch, 
                args,
                authorship_model,
                authorship_tokenizer,
                similarity_fn
            )
            similarity.append(scores['similarity'])
            parent_precision.append(scores['parent'][0])
            parent_recall.append(scores['parent'][1])
            parent_f_score.append(scores['parent'][2])
            acc.append(scores['authorship'])

    return {
        'similarity': np.mean(similarity),
        'parent': [np.mean(parent_precision), np.mean(parent_recall), np.mean(parent_f_score)],
        'authorship': np.mean(acc)
    }


def train_batch(model, tokenizer, optimizer, device, batch, args):
    empty_cache()
    inputs, targets, _, _, _, _, = batch

    tokenized_inputs = tokenizer.batch_encode_plus(
        inputs,
        padding='max_length',
        max_length=args.model_input_length,
        truncation=True,
        return_tensors="pt"
    )
    tokenized_targets = tokenizer.batch_encode_plus(
        targets,
        padding='max_length',
        max_length=args.model_output_length,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = tokenized_inputs["input_ids"].to(device)
    attention_mask = tokenized_inputs["attention_mask"].to(device)
    labels = tokenized_targets["input_ids"].to(device)
    labels = labels.clone().detach()
    labels[labels == tokenizer.pad_token_id] = -100
    optimizer.zero_grad()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    loss = outputs.loss
    running_loss = loss.item()
    loss.backward()
    optimizer.step()
    return running_loss


def train(
        model, 
        tokenizer, 
        optimizer, 
        device, 
        train_df, 
        eval_df, 
        args, 
        authorship_model=None, 
        authorship_tokenizer=None,
        similarity_fn=None
    ):

    print('Training ...')
    train_dt = PDTTDataset(train_df, tokenizer, args)
    train_ld = DataLoader(train_dt, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        model.train()

        running_loss = .0
        for batch in tqdm(train_ld, desc=f"Epochs {epoch} / {args.epochs}"):
            loss = train_batch(model, tokenizer, optimizer, device, batch, args)
            running_loss += loss

        running_loss = running_loss/len(train_dt)
        print(f"[Training] Epochs {epoch} / {args.epochs} : loss={running_loss}")
        save(model, optimizer, args.model_path)

        model.eval()
        scores = evaluate(
            model, 
            tokenizer, 
            device, 
            eval_df, 
            args, 
            authorship_model,
            authorship_tokenizer,
            similarity_fn
        )
        print(f"[Evaluation] scores = {scores}")


def load_data(args):
    print('[Data] Configuration ...')

    data_config = None
    with open(args.data_config_path, "r") as f:
        data_config = json.load(f)

    data_to_text_path = data_config['data_to_text_path']
    users_configs = data_config['users']

    data_to_text_df = pd.read_csv(data_to_text_path)
    print(f"[Data] [Data-to-text] Examples")
    print(data_to_text_df.head(2))

    train_dfs, eval_dfs = [], []

    print('[Data] User processing ...')
    for user_configs in users_configs:
        print(f"[Data] [User] {user_configs['username']} ({user_configs['userid']})")
        user_data_df = pd.read_csv(user_configs['user_data_path'])[[args.pdtt_refs_col]]
        print(user_data_df.head(2))
        n_samples = len(user_data_df)
        user_data_df[args.user_id_col] = [user_configs['userid']] * n_samples
        merged_df = data_to_text_df.merge(user_data_df, left_index=True, right_index=True)
        train_df = merged_df.sample(frac=args.train_size)
        eval_df = merged_df.drop(train_df.index).reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)
        train_dfs.append(train_df)
        eval_dfs.append(eval_df)

    train_df = pd.concat(train_dfs)
    eval_df = pd.concat(eval_dfs)
        
    print(f"[Data] Train shape : {train_df.shape}")
    print(f"[Data] Eval shape : {eval_df.shape}")
    print(f"[Data] Examples : \n{train_df.head(2)}\n")
    print("[Data] Successful configuration")

    return train_df, eval_df


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    print('[Data] Loading data ...')
    train_df, eval_df = load_data(args)
    print('[Data] Data loaded')

    train_df[args.pdtt_refs_col] = train_df[args.pdtt_refs_col].apply(ast.literal_eval)
    eval_df[args.pdtt_refs_col]  = eval_df[args.pdtt_refs_col].apply(ast.literal_eval)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'[Device] Device : {device}')

    print(f'[Model] Model : {args.model_name}')
    print(f'[Model] Loading model ...')
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, model_max_length=args.model_input_length)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name, return_dict=True)
    model = model.to(device)

    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
    )

    if args.model_path != "":
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        model_dir = f"./models/{args.model_name}/"
        os.makedirs(model_dir, exist_ok=True)
        args.model_path = f"{model_dir}/model.pt"
    print('[Model] Model loaded')

    authorship_model = None
    authorship_tokenizer = None
    if args.authorship_model_path != "":
        print(f'[Authorship Model] Loading model from {args.authorship_model_path}')
        authorship_model = BertClassifier(n_authors=args.authorship_n_authors)
        authorship_model.load_state_dict(torch.load(args.authorship_model_path))
        authorship_model.to(device)
        authorship_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        print(f'[Authorship Model] Model loaded')

    def fn(*args):
        return 0
    similarity_fn = fn 

    if args.similarity_path != "":
        tok = TreebankWordTokenizer()

        sim_model = torch.load(args.similarity_path + 'sim.pt')
        state_dict = sim_model['state_dict']
        vocab_words = sim_model['vocab_words']
        sim_model_args = sim_model['args']

        sim_model = WordAveraging(sim_model_args, vocab_words)
        sim_model.load_state_dict(state_dict, strict=True)
        sp = spm.SentencePieceProcessor()
        sp.Load(args.similarity_path + 'sim.sp.30k.model')
        sim_model.eval()

        def make_example(sentence, sim_model):
            sentence = sentence.lower()
            sentence = " ".join(tok.tokenize(sentence))
            sentence = sp.EncodeAsPieces(sentence)
            wp1 = Example(" ".join(sentence))
            wp1.populate_embeddings(sim_model.vocab)
            return wp1

        def find_similarity(s1, s2):
            with torch.no_grad():
                s1 = [make_example(x, sim_model) for x in s1]
                s2 = [make_example(x, sim_model) for x in s2]
                wx1, wl1, wm1 = sim_model.torchify_batch(s1)
                wx2, wl2, wm2 = sim_model.torchify_batch(s2)
                scores = sim_model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
                return [x.item() for x in scores]

        similarity_fn = find_similarity

    train(
        model, 
        tokenizer, 
        optimizer, 
        device, 
        train_df, 
        eval_df, 
        args, 
        authorship_model,
        authorship_tokenizer,
        similarity_fn
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_input_length', default=1024, type=int)
    parser.add_argument('--model_output_length', default=512, type=int)

    parser.add_argument('--input_col', default='input', type=str)
    parser.add_argument('--user_id_col', default='user_id', type=str)
    parser.add_argument('--dtt_ref_col', default='dtt_ref', type=str)
    parser.add_argument('--pdtt_refs_col', default='pdtt_refs', type=str)
    parser.add_argument('--parent_col', default='parent', type=str)

    parser.add_argument('--data_config_path', default='dtt_t5_config_example.json', type=str)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--train_size', default=.8, type=float)
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--model_name', default='t5-small', type=str)
    parser.add_argument('--model_path', default='', type=str)

    parser.add_argument('--authorship_model_path', default="", type=str)
    parser.add_argument('--authorship_n_authors', default=2, type=int)

    parser.add_argument('--similarity_path', default='style_paraphrase/evaluation/similarity/sim/', type=str)

    args = parser.parse_args()
    main(args)
