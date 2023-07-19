# Ben Kabongo
# Personnalized data-to-text neural generation
# ISIR/MLIA, 2023

# Sentiment analysis with 'sentiment-analysis' pipeline

import pandas as pd
import warnings
from sklearn import metrics
from transformers import pipeline
warnings.filterwarnings('ignore')

def binarize(n, s=2):
    return "positive" if n > s else "negative"

def main():
    data_path = '../../../Data/appdata/rotten/review_to_rating.csv'
    data_df = pd.read_csv(data_path, index_col=0)

    data_df = data_df[['input', 'output']].rename({'input': 'text', 'output': 'label'}, axis=1)
    data_df['text'] = data_df['text'].apply(str)
    data_df['label'] = data_df['label'].apply(binarize)

    sentiment_pipeline = pipeline("sentiment-analysis")
    data_df['sentiment_analysis'] = sentiment_pipeline(data_df['text'].tolist())
    data_df['pred_label'] = data_df['sentiment_analysis'].apply(lambda x : x['label'])
    data_df['pred_score'] = data_df['sentiment_analysis'].apply(lambda x : x['score'])

    data_df.to_csv('sentiment_analysis.csv', index=True)

    res = {
        'accuracy': metrics.accuracy_score(data_df['pred_label'], data_df['label']),
        'f1-score': metrics.f1_score(data_df['pred_label'], data_df['label'], average="weighted"),
    }

    print(res)

if __name__ == '__main__':
    main()
    