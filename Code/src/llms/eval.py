import numpy as np
import pandas as pd
import re
import sys
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate import meteor_score
from rouge import Rouge

def calculate_bleu(reference, candidate):
    smoothie = SmoothingFunction().method1
    reference = [reference.split()]
    candidate = candidate.split()
    return sentence_bleu(reference, candidate, smoothing_function=smoothie)

def calculate_rouge(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)[0]
    return scores['rouge-l']['f']

def calculate_meteor(reference, candidate):
    return meteor_score.meteor_score([reference], candidate)

def clean_output(output):
    match = re.search(r'\d+\.\d+', output)

    if match:
        cleaned_output = float(match.group())
    else:
        cleaned_output = -1.0

    return cleaned_output

def compute_mae(target, prediction):
    absolute_errors = np.abs(target - prediction)
    mae = np.mean(absolute_errors)
    return mae

def compute_mse(target, prediction):
    squared_errors = np.square(target - prediction)
    mse = np.mean(squared_errors)
    return mse

def compute_rmse(target, prediction):
    mse = compute_mse(target, prediction)
    rmse = np.sqrt(mse)
    return rmse


def main():
    res_df = pd.read_csv(sys.argv[1], index_col=0)
    res_df['target'] = res_df['target'].apply(str)
    res_df['BLEU'] = [calculate_bleu(t, o) for t, o in zip(res_df['target'], res_df['output'])]
    res_df['ROUGE'] = [calculate_rouge(t, o) for t, o in zip(res_df['target'], res_df['output'])]
    res_df['METEOR'] = [calculate_meteor(t, o) for t, o in zip(res_df['target'], res_df['output'])]
    res_df['target_float'] = res_df['target'].apply(float)
    res_df['output_clean'] = res_df['output'].apply(clean_output)
    correct_output_df = res_df[res_df['output_clean'] != -1]

    mae = compute_mae(res_df['target_float'], res_df['output_clean'])
    mse = compute_mse(res_df['target_float'], res_df['output_clean'])
    rmse = compute_rmse(res_df['target_float'], res_df['output_clean'])

    mae_ = compute_mae(correct_output_df['target_float'], correct_output_df['output_clean'])
    mse_ = compute_mse(correct_output_df['target_float'], correct_output_df['output_clean'])
    rmse_ = compute_rmse(correct_output_df['target_float'], correct_output_df['output_clean'])

    print(sys.argv[1])
    print('BLEU \t:', np.mean(res_df['BLEU']))
    print('ROUGE \t:', np.mean(res_df['ROUGE']))
    print('METEOR \t:', np.mean(res_df['METEOR']))
    print('MAE :\t', mae)
    print('MSE :\t', mse)
    print('RMSE :\t', rmse)
    print('MAE 2 :\t', mae_)
    print('MSE 2 :\t', mse_)
    print('RMSE 2 :\t', rmse_)
