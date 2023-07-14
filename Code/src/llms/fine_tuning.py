import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from rich import box
from rich.console import Console
from rich.table import Column, Table
from torch import cuda
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer


def binarize(n, s=2):
    return "positive" if n > s else "negative"


def display_df(df):
    console = Console()
    table = Table(
        Column("input_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )
    for _, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])
    console.print(table)


def save(model, tokenizer, output_dir):
    console.log(f"[Saving Model]...\n")
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


class PDataset(Dataset):

    def __init__(self, data_df, tokenizer, input_len, target_len, input_text, target_text):
        self.tokenizer = tokenizer
        self.data_df = data_df
        self.input_len = input_len
        self.summ_len = target_len
        self.input_text = self.data_df[input_text]
        self.target_text = self.data_df[target_text]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        input_text = str(self.input_text[index])
        target_text = str(self.target_text[index])

        input_text = " ".join(input_text.split())
        target_text = " ".join(target_text.split())

        input = self.tokenizer.batch_encode_plus(
            [input_text],
            max_length=self.input_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = input["input_ids"].squeeze()
        input_mask = input["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "input_ids": input_ids.to(dtype=torch.long),
            "input_mask": input_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


def train(
        epoch, 
        tokenizer, 
        model, 
        device, 
        loader, 
        optimizer, 
        output_dir
    ):

    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["input_ids"].to(device, dtype=torch.long)
        mask = data["input_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]

        if _ % 10 == 0:
            training_logger.add_row(str(epoch), str(_), str(loss.item()))
        if _ % 100 == 0:
            console.print(training_logger)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    save(model, tokenizer, output_dir)


def validate(tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['input_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask, 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
                for g in generated_ids
            ]
            target = [
                tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for t in y
            ]
            if _ % 10 == 0:
                console.print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


def T5Trainer(data_df, input_text, target_text, model_params, output_dir):
    torch.manual_seed(model_params["SEED"])
    np.random.seed(model_params["SEED"])
    torch.backends.cudnn.deterministic = True

    console.log(f"""[Model]: Loading {model_params["MODEL_PATH"]}...\n""")
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL_NAME"])
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL_PATH"])
    model = model.to(device)

    console.log(f"[Data]: Reading data...\n")
    data_df = data_df[[input_text, target_text]]
    data_df[input_text] = data_df[input_text].apply(str)
    data_df[target_text] = data_df[target_text].apply(binarize)
    data_df[target_text] = data_df[target_text].apply(str)
    display_df(data_df.head(3))

    train_size = model_params['TRAIN_SIZE']
    train_df = data_df.sample(frac=train_size, random_state=model_params["SEED"])
    val_df = data_df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    console.print(f"FULL Dataset: {data_df.shape}")
    console.print(f"TRAIN Dataset: {train_df.shape}")
    console.print(f"TEST Dataset: {val_df.shape}\n")

    training_set = PDataset(
        train_df,
        tokenizer,
        model_params["MAX_INPUT_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        input_text,
        target_text,
    )

    val_set = PDataset(
        val_df,
        tokenizer,
        model_params["MAX_INPUT_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        input_text,
        target_text,
    )

    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
         "num_workers": 0,
        }

    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=model_params["LEARNING_RATE"])

    def trainer_validate(tokenizer, model, device, val_loader):
        console.log(f"[Initiating Validation]...\n")
        for _ in range(model_params["VAL_EPOCHS"]):
            predictions, actuals = validate(tokenizer, model, device, val_loader)
            final_df = pd.DataFrame({"yhat": predictions, "y": actuals})
            final_df.to_csv(os.path.join(output_dir, "predictions.csv"))
        console.log(f"[Validation Completed.]\n")

    console.log(f"[Initiating Fine Tuning]...\n")
    for epoch in range(model_params["START_TRAIN_EPOCHS"], model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer, output_dir)
        if epoch % model_params["VALIDATION_EVERY_TRAIN_EPOCHS"] == 0:
            trainer_validate(tokenizer, model, device, val_loader)
    save(model, tokenizer, output_dir)

    trainer_validate(tokenizer, model, device, val_loader)
    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.print(f"[Model] Model saved @ {os.path.join(output_dir, 'model_files')}\n")
    console.print(f"[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n")
    console.print(f"[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n")


if __name__ == "__main__":
    console = Console(record=True)

    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )

    device = 'cuda' if cuda.is_available() else 'cpu'
    console.log(f"[Device] Device : {device}")

    model_params = {
        "MODEL_NAME": sys.argv[3],
        "MODEL_PATH": sys.argv[4],
        "TRAIN_BATCH_SIZE": int(sys.argv[5]),
        "VALID_BATCH_SIZE": int(sys.argv[6]),
        "START_TRAIN_EPOCHS": int(sys.argv[7]),
        "TRAIN_EPOCHS": int(sys.argv[8]),
        "TRAIN_SIZE": float(sys.argv[9]),
        "VAL_EPOCHS": 1,
        "VALIDATION_EVERY_TRAIN_EPOCHS": 5,
        "LEARNING_RATE": 1e-4,
        "MAX_INPUT_TEXT_LENGTH": 512,
        "MAX_TARGET_TEXT_LENGTH": 50,
        "SEED": 42,
    }

    data_path = sys.argv[1]
    task_name = sys.argv[2]
    model_name = sys.argv[3]
    model_path = sys.argv[4]

    data_df = pd.read_csv(data_path, index_col=0)

    if model_path == '.':
        output_dir = f'./fine_tuning/{task_name}/{model_name}/{datetime.now().strftime("%d%m%y%H%M")}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_params['MODEL_PATH'] = model_params['MODEL_NAME']
    else:
        output_dir = model_path
        model_params['MODEL_PATH'] = os.path.join(output_dir, "model_files")

    T5Trainer(data_df=data_df, input_text=sys.argv[10], target_text=sys.argv[11], model_params=model_params, output_dir=output_dir)