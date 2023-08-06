# Ben Kabongo
# Personnalized data-to-text neural generation
# ISIR/MLIA, 2023
# Crédits : 
# https://github.com/martiansideofthemoon/style-transfer-paraphrase/blob/master/style_paraphrase/evaluation/similarity/test_sim.py

# Similarity evaluation

import argparse
import ast
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from style_paraphrase.evaluation.similarity.sim_models import WordAveraging
from style_paraphrase.evaluation.similarity.sim_utils import Example
from nltk.tokenize import TreebankWordTokenizer


tok = TreebankWordTokenizer()

model = torch.load('style_paraphrase/evaluation/similarity/sim/sim.pt')
state_dict = model['state_dict']
vocab_words = model['vocab_words']
args = model['args']

model = WordAveraging(args, vocab_words)
model.load_state_dict(state_dict, strict=True)
sp = spm.SentencePieceProcessor()
sp.Load('style_paraphrase/evaluation/similarity/sim/sim.sp.30k.model')
model.eval()


def make_example(sentence, model):
    sentence = sentence.lower()
    sentence = " ".join(tok.tokenize(sentence))
    sentence = sp.EncodeAsPieces(sentence)
    wp1 = Example(" ".join(sentence))
    wp1.populate_embeddings(model.vocab)
    return wp1


def find_similarity(s1, s2):
    with torch.no_grad():
        s1 = [make_example(x, model) for x in s1]
        s2 = [make_example(x, model) for x in s2]
        wx1, wl1, wm1 = model.torchify_batch(s1)
        wx2, wl2, wm2 = model.torchify_batch(s2)
        scores = model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
        return [x.item() for x in scores]


def main(inputs, outputs):
    all_similarity = []

    print("Compute similarity ...")
    for i, o in zip(inputs, outputs):
        similarity = find_similarity([i], o)
        similarity = np.mean(similarity)
        all_similarity.append(similarity)

    print(f"Similarity : mean = {np.mean(all_similarity)}, std = {np.std(all_similarity)}")
    with open("similarity_res.txt", "a") as f:
        f.write(f"mean = {np.mean(all_similarity)}, std = {np.std(all_similarity)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--input_feature", default="input", type=str)
    parser.add_argument("--output_feature", default="outputs", type=str)
    args = parser.parse_args()

    data_df = pd.read_csv(args.data_path, index_col=0)
    try: 
        data_df[args.output_feature] = data_df[args.output_feature].apply(ast.literal_eval)
    except:
        data_df[args.output_feature] = data_df[args.output_feature].apply(lambda x: [str(x)])

    inputs = data_df[args.input_feature].apply(str).tolist()
    outputs = data_df[args.output_feature].tolist()

    with open("similarity_res.txt", "a") as f:
        f.write(f"{args.data_path} : ")

    main(inputs, outputs)

