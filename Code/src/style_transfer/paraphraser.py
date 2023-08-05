# Ben Kabongo
# Personnalized data-to-text neural generation
# ISIR/MLIA, 2023
# Crédits : https://github.com/martiansideofthemoon/style-transfer-paraphrase/blob/master/demo_paraphraser.py

# Paraphraser

import argparse
import logging
import pandas as pd
import sys
import torch
from tqdm import tqdm

from style_paraphrase.inference_utils import GPT2Generator

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default="paraphraser_gpt2_large", type=str)
parser.add_argument('--top_p_value', default=0.6, type=float)
parser.add_argument('--n_samples', default=5, type=int)
parser.add_argument('--input_dataset_path', default='in.csv', type=str)
parser.add_argument('--input_feature_name', default='input', type=str)
parser.add_argument('--output_dataset_path', default='out.csv', type=str)
args = parser.parse_args()

if not torch.cuda.is_available():
    print("Please check if a GPU is available or your Pytorch installation is correct.")
    sys.exit()

inputs = pd.read_csv(args.input_dataset_path)[args.input_feature_name]
outputs_greedy  = []
outputs_samples = []

print("Loading paraphraser...")
paraphraser = GPT2Generator(args.model_dir, upper_length="same_5")

print("Paraphrasing...")
for input_sentence in tqdm(inputs, desc='Paraphrasing'):
    paraphraser.modify_p(top_p=0.0)
    greedy_decoding = paraphraser.generate(input_sentence)

    paraphraser.modify_p(top_p=args.top_p_value)
    top_p_samples, _ = paraphraser.generate_batch([input_sentence] * args.n_samples)
    
    outputs_greedy.append(greedy_decoding)
    outputs_samples.append(top_p_samples)

print('Results recording...')
pd.DataFrame(
    {'input': inputs, 'greedy': outputs_greedy, 'outputs': outputs_samples}
).to_csv(args.output_dataset_path)
