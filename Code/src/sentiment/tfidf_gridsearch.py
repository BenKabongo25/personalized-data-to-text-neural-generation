# Ben Kabongo
# Personnalized data-to-text neural generation
# ISIR/MLIA, 2023

# Sentiment analysis with TF-IDF grid search


import json
import pandas as pd
import re
import warnings
from itertools import product
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import (linear_model,
                     metrics,
                     preprocessing)
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings('ignore')

def binarize(n, s=2):
    return "positive" if n > s else "negative"

def load_data():
    data_path = '../../../Data/appdata/rotten/'
    data_df = pd.read_csv(data_path, index_col=0)

    data_df = data_df[['input', 'output']].rename({'input': 'text', 'output': 'label'}, axis=1)
    data_df['text'] = data_df['text'].apply(str)
    data_df['label'] = data_df['label'].apply(binarize)

    train_size = 0.75
    train_df = data_df.sample(frac=train_size, random_state=42)
    test_df = data_df.drop(train_df.index).reset_index(drop=True)

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(train_df.label)

    train_df['label_id'] = label_encoder.transform(train_df.label)
    test_df['label_id'] = label_encoder.transform(test_df.label)

    return train_df, test_df


def delete_punctuation(text):
    punctuation = r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\n\t]"
    text = re.sub(punctuation, " ", text)
    text = re.sub('( )+', ' ', text)
    return text


def delete_digit(text):
    return re.sub('[0-9]+', '', text)


def delete_balise(text):
    return re.sub("<.*?>", "", text)


def stem(text):
    stemmer = EnglishStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmed_text = " ".join(stemmed_tokens)
    return stemmed_text

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_text = " ".join(lemmatized_tokens)
    return lemmatized_text


def TfIdfClassification(vectorizer, train_df, test_df, clf, clf_args):
    X_train = vectorizer.fit_transform(train_df.text)
    X_test = vectorizer.transform(test_df.text)

    y_train = train_df.label_id
    y_test = test_df.label_id

    model = clf(**clf_args)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return {
        'train': {
            'accuracy': metrics.accuracy_score(y_train_pred, y_train),
            'f1-score': metrics.f1_score(y_train_pred, y_train, average="weighted"),
        },
        'test': {
            'accuracy': metrics.accuracy_score(y_test_pred, y_test),
            'f1-score': metrics.f1_score(y_test_pred, y_test, average="weighted"),
        }
    }


def grid_search():
    train_df, test_df = load_data()

    preprocessors = [
        None,
        delete_punctuation,
        delete_digit,
        stem,
        lemmatize,
    ]

    tfidf_params = {
        'stop_words' : ['english'],
        'strip_accents': [None, 'ascii', 'unicode'],
        'lowercase' : [False, True],
        'analyzer': ['word', 'char'],
        'tokenizer': [None, word_tokenize],
        'ngram_range': [(1, 1), (1, 2), (1, 3)],
        'max_df': [.5, .8, .9, 1.],
        'min_df': [.0, .1, .2, .3, .4],
        'max_features': [None, 2_000, 5_000, 10_000],
        'binary': [False, True],
    }

    lr_params = {
        'C': [.01, 1, 10],
    }

    best_params = None
    best_scores = None

    for preprocessor in preprocessors:
        for tfidf_param in product(*tfidf_params.values()):
            tfidf_param_dict = dict(zip(tfidf_params.keys(), tfidf_param))

            for C in lr_params['C']:
                scores = TfIdfClassification(
                    TfidfVectorizer(preprocessor=preprocessor, **tfidf_param_dict),
                    train_df,
                    test_df,
                    linear_model.LogisticRegression,
                    {'C': C}
                )

                params = dict(tfidf_param_dict)
                params.update({'C': C, 'preprocessor': preprocessor})

                print(f"{pd.DataFrame({'value':params})}\n")
                print(f"{pd.DataFrame(scores)}\n")
                print('=======================================\n')

            
                if (best_scores is None or 
                    (best_scores['test']['f1-score'] < scores['test']['f1-score'])):
                    params_dict = dict(tfidf_param_dict)
                    params_dict['tokenizer'] = str(params_dict['tokenizer'])
                    best_params = {
                        'tfidf_params': params_dict,
                        'lr_params': {'C': C},
                        'preprocessor': str(preprocessor)
                    }
                    best_scores = scores
                    json.dump(best_params, open(f'best_params_save.json', 'w'))

    json.dump(best_params, open(f'best_params_save.json', 'w'))

if __name__ == '__main__':
    grid_search()
