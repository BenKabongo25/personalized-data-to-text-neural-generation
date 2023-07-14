# -*- coding: utf-8 -*-

import gc
import os
import math
import numpy as np
import pandas as pd
import random
import sentencepiece
import sys
import torch
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torch import nn
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor


def discretise(n, s=2):
    match s:
        case 2:
            return 'positive' if n > 2.5 else 'negative'
        case 3:
            if n < 2: return 'negative'
            if n < 4: return 'neutral'
            return 'positive'
        case 4:
            if n < 1.25: return 'very negative'
            if n < 2.50: return 'negative'
            if n < 3.75: return 'positive'
            return 'very positive'
        case 5:
            if n < 1: return 'very negative'
            if n < 2: return 'negative'
            if n < 3: return 'neutral'
            if n < 4: return 'positive'
            return 'very positive'
        case _:
            return math.floor(n)


class RatingDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __getitem__(self, idx):
        return [self.inputs[idx], self.outputs[idx]]

    def __len__(self):
        return len(self.outputs)


class PromptOnlyModel(nn.Module):
    def __init__(self, prompt_length, dim_out):
        super(PromptOnlyModel, self).__init__()
        self.embedder = nn.Linear(dim_out, prompt_length, bias=False)
        self.prompt_length = prompt_length

    def forward(self):
        return self.embedder.weight

    def len(self):
        return self.prompt_length


def save(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path + 'prompt_optimizer.pt')


def cleancuda():
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()


def train(model,
          tokenizer,
          optimizer,
          prompt,
          writer,
          device,
          train_dt,
          eval_dt,
          batch_size,
          epochs,
          model_path):

    if model_path is not None and not os.path.exists(model_path):
        os.makedirs(model_path)

    train_ld = DataLoader(train_dt, batch_size=batch_size, shuffle=True)
    eval_ld = DataLoader(eval_dt, batch_size=batch_size, shuffle=True)

    for epoch in tqdm(range(epochs), 'Train'):
        print(f'\n[Training] EPOCH {epoch}/{epochs}:')

        tqdm_util = tqdm(enumerate(train_ld))
        for i, (x, y) in tqdm_util:
            cleancuda()

            with torch.no_grad():
                in_ = tokenizer(list(x),
                                padding=True,
                                truncation=True,
                                max_length=512,
                                return_tensors='pt')
                out_ = tokenizer(list(y),
                                padding=True,
                                truncation=True,
                                max_length=2,
                                return_tensors='pt')

                input_batch, attention_batch = in_.input_ids, in_.attention_mask
                output_batch = out_.input_ids
                output_batch = output_batch.clone().detach()
                output_batch[output_batch == tokenizer.pad_token_id] = -100
                attention_batch = torch.hstack([
                    torch.tensor([[1] * prompt.len()] * input_batch.shape[0]),
                    attention_batch
                ])

                attention_batch = attention_batch.to(device)
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)

                embed_batch = model.encoder.embed_tokens(input_batch)

            prompt_batch = torch.stack([prompt()] * embed_batch.shape[0])
            stacked_batch = torch.hstack([prompt_batch,embed_batch])
            stacked_batch = stacked_batch.to(device)

            res = model(inputs_embeds=stacked_batch,
                        labels=output_batch,
                        attention_mask=attention_batch)
            loss = res.loss
            loss_num = loss.item()/len(x)
            writer.add_scalar('Loss/train', loss_num, epoch*len(train_ld)+i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                tqdm_util.set_description('[Training] train loss: {:.5f}'.format(loss_num))
                if model_path is not None:
                    save(prompt, optimizer, model_path)

        if model_path is not None:
            save(prompt, optimizer, model_path)

        with torch.no_grad():
            cleancuda()

            sample_id = random.sample(range(len(eval_dt)), 10)
            test_gen = [eval_dt[n][0] for n in sample_id]
            target = [eval_dt[n][1] for n in sample_id]

            input_ids = tokenizer(test_gen,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=512
                                ).input_ids
            input_ids = input_ids.to(device)

            embed_batch = model.encoder.embed_tokens(input_ids)
            prompt_batch = torch.stack([prompt()] * embed_batch.shape[0])
            stacked_batch = torch.hstack([prompt_batch,embed_batch])
            output = model.generate(inputs_embeds=stacked_batch, max_length=2)
            output = tokenizer.batch_decode(output, skip_special_tokens=True)

            print("[Examples] ")
            for i in range(len(test_gen)):
                print('Input :\t', test_gen[i])
                print('Generated :\t', output[i])
                print('Output :\t', target[i])
                print(f"{'='*100}\n")

            tqdm_util = tqdm(enumerate(eval_ld))
            for i,(x,y) in tqdm_util:
                cleancuda()

                in_ = tokenizer(list(x),
                                padding=True,
                                truncation=True,
                                max_length=512,
                                return_tensors='pt')
                out_ = tokenizer(list(y),
                                padding=True,
                                truncation=True,
                                max_length=2,
                                return_tensors='pt')

                input_batch, attention_batch = in_.input_ids, in_.attention_mask
                output_batch = out_.input_ids
                output_batch = output_batch.clone().detach()
                output_batch[output_batch == tokenizer.pad_token_id] = -100
                attention_batch = torch.hstack([
                    torch.tensor([[1] * prompt.len()] * input_batch.shape[0]),
                    attention_batch
                ])

                attention_batch = attention_batch.to(device)
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)

                embed_batch = model.encoder.embed_tokens(input_batch)
                prompt_batch = torch.stack([prompt()] * embed_batch.shape[0])
                stacked_batch = torch.hstack([prompt_batch, embed_batch])
                stacked_batch = stacked_batch.to(device)

                res = model(inputs_embeds=stacked_batch,
                            labels=output_batch,
                            attention_mask=attention_batch)

                loss = res.loss
                loss_num = loss.item()/len(x)
                writer.add_scalar('Loss/eval', loss_num, epoch*len(eval_ld)+i)

                if i % 10 == 0:
                    tqdm_util.set_description('[Validation] Validation loss: {:.5f}'.format(loss_num))


def generate(model,
            tokenizer,
            prompt,
            device,
            data_dt,
            batch_size,
            model_path=None):
    
    if model_path is not None and not os.path.exists(model_path):
        os.makedirs(model_path)

    data_ld = DataLoader(data_dt, batch_size=batch_size, shuffle=True)

    inputs = []
    targets = []
    outputs = []

    with torch.no_grad():
        cleancuda()

        for i, (x, y) in tqdm(enumerate(data_ld)):
            input = list(x)
            target = list(y)

            input_ids = tokenizer(input,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=512
                                ).input_ids
            input_ids = input_ids.to(device)

            embed_batch = model.encoder.embed_tokens(input_ids)
            prompt_batch = torch.stack([prompt()] * embed_batch.shape[0])
            stacked_batch = torch.hstack([prompt_batch,embed_batch])
            output = model.generate(inputs_embeds=stacked_batch, max_length=2)
            output = tokenizer.batch_decode(output, skip_special_tokens=True)

            inputs.extend(input)
            targets.extend(target)
            outputs.extend(list(output))

        res_df = pd.DataFrame()
        res_df['input'] = inputs
        res_df['target'] = targets
        res_df['output'] = outputs

        now = datetime.now().strftime("%d%m%y_%H%M")
        res_df.to_csv(model_path + f'predicted_{now}.csv')


def main(
        train_flag,
        model_type,
        prompt_length,
        batch_size,
        model_path,
        data_path,
        n_classes
    ):

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print('[Device] Device :\t', device)
    cleancuda()

    input_text = "input"
    target_text = "output"

    print("[Data] Loading data ...")
    data_df = pd.read_csv(data_path, index_col=0)
    data_df = data_df[[input_text, target_text]]
    data_df[input_text] = data_df[input_text].apply(str)
    data_df[target_text] = data_df[target_text].apply(lambda x: discretise(x, n_classes))
    data_df[target_text] = data_df[target_text].apply(str)

    sizes = {2: 15_000, 3: 10_000, 4: 8_000, 5: 4_000}

    train_cl_size = sizes.get(n_classes, 1_000)
    eval_cl_size  = sizes.get(n_classes, 1_000)
    train_cl_dfs  = []
    eval_cl_dfs   = []
    test_cl_dfs   = []

    for cl in data_df[target_text].unique():
        data_cl_df = data_df[data_df[target_text] == cl]
        train_cl_df = data_cl_df.sample(n=train_cl_size, random_state=42)
        rest_df = data_cl_df.drop(train_cl_df.index).reset_index(drop=True)
        train_cl_df = train_cl_df.reset_index(drop=True)
        eval_cl_df = rest_df.sample(n=eval_cl_size, random_state=42)
        test_cl_df = rest_df.drop(eval_cl_df.index).reset_index(drop=True)
        eval_cl_df = eval_cl_df.reset_index(drop=True)

        train_cl_dfs.append(train_cl_df)
        eval_cl_dfs.append(eval_cl_df)
        test_cl_dfs.append(test_cl_df)

        print(f"[Data] Classe {cl}")
        print(f"[Data] Train size : {train_cl_df.shape}")
        print(f"[Data] Eval size : {eval_cl_df.shape}")
        print(f"[Data] Test size : {test_cl_df.shape}\n")

    train_df = pd.concat(train_cl_dfs)
    eval_df  = pd.concat(eval_cl_dfs)
    test_df  = pd.concat(test_cl_dfs)

    print(f"[Data] Full data")
    print(f"[Data] Train size : {train_df.shape}")
    print(f"[Data] Eval size : {eval_df.shape}")
    print(f"[Data] Test size : {test_df.shape}\n")

    print(f"[Data] Examples :")
    print(train_df.head(3))

    train_dt = RatingDataset(train_df[input_text].tolist(), train_df[target_text].tolist())
    test_dt  = RatingDataset(test_df[input_text].tolist(), test_df[target_text].tolist())
    eval_dt  = RatingDataset(eval_df[input_text].tolist(), eval_df[target_text].tolist())

    tokenizer = T5Tokenizer.from_pretrained(model_type, model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained(model_type, return_dict=True)
    model = model.to(device)

    prompt = PromptOnlyModel(prompt_length, model.shared.embedding_dim)

    optimizer = Adafactor(
        prompt.parameters(),
        lr=0.001,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
    )

    if model_path != '.':
        checkpoint = torch.load(model_path + 'prompt_optimizer.pt', map_location='cpu')
        prompt.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        model_path = f'./prompt_tuning/review_to_rating_{n_classes}/{model_type}/{datetime.now().strftime("%d%m%y_%H%M")}/'
    print('[Model] Model path :\t', model_path)

    prompt = prompt.to(device)

    for p in model.parameters():
        p.requires_grad = False

    writer = SummaryWriter()

    if train_flag:
        train(model,
            tokenizer,
            optimizer,
            prompt,
            writer,
            device,
            train_dt,
            eval_dt,
            batch_size=batch_size,
            epochs=100,
            model_path=model_path,
        )
    
    generate(model,
        tokenizer,
        prompt,
        device,
        test_dt,
        batch_size,
        model_path
    )


if __name__ == '__main__':
    main(
        sys.argv[1] == 'train',
        model_type=sys.argv[2],
        prompt_length=int(sys.argv[3]),
        batch_size=int(sys.argv[4]),
        model_path=sys.argv[5],
        data_path=sys.argv[6],
        n_classes=2
    )
