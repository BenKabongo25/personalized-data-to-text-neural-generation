# Ben Kabongo
# Personalized data-to-text neural generation
# ISIR/MLIA, 2023

# Création des datasets

import numpy as np
import os
import pandas as pd
import sys

BASE_DIR = '../../../Data/'



def movies_critics():
    """ Création des datasets movies_input et users_output 
    movies_input : formattage des données de films de Rotten Tomatoes pour du data-to-text
    users_output : formattage des critiques utilisateurs de Rotten Tomatoes
    """

    # chargement des datasets de Rotten Tomatoes
    critic_reviews_filename = BASE_DIR + 'rotten/rotten_tomatoes_critic_reviews.csv'
    movies_filename = BASE_DIR + 'rotten/rotten_tomatoes_movies.csv'

    critic_reviews_df = pd.read_csv(critic_reviews_filename)
    movies_df = pd.read_csv(movies_filename)

    # Création du dataset movies
    movies_input_df = pd.DataFrame()
    movies_input_df['movieID'] = movies_df['rotten_tomatoes_link']
    movies_input_df['input1'] = movies_df.apply(movie2inputformat1, axis=1)
    movies_input_df['input2'] = movies_df.apply(movie2inputformat2, axis=1)
    movies_input_df['input3'] = movies_df.apply(movie2inputformat3, axis=1)
    movies_input_df.to_csv(BASE_DIR + 'appdata/rotten/movies_input.csv', index=True)

    # Création du dataset des users
    users_output_df = pd.DataFrame()
    users_output_df['movieID'] = critic_reviews_df['rotten_tomatoes_link']
    users_output_df['userID'] = critic_reviews_df['critic_name'].apply(to_userID)
    users_output_df['rating'] = critic_reviews_df['review_score'].apply(to_rating)
    users_output_df['target'] = critic_reviews_df['review_content']
    users_output_df.to_csv(BASE_DIR + 'appdata/rotten/users_output.csv', index=True)



def wikiroto():
    """ Merge du jeu de données WikiRoto et Rotten Tomatoes 
    Pour le data-to-text personnalisé
    """
    wikiroto_dir = BASE_DIR + 'wikiroto/'

    train_df = pd.read_csv(wikiroto_dir + 'data/wikiroto_train_with_table.csv')
    test_df = pd.read_csv(wikiroto_dir + 'data/wikiroto_test_with_table.csv')
    eval_df = pd.read_csv(wikiroto_dir + 'data/wikiroto_eval_with_table.csv')

    wikiroto_df = pd.concat([train_df, test_df, eval_df])
    wikiroto_df = wikiroto_df[['new_input', 'target', 'table', 'movie_info']]
    wikiroto_df = wikiroto_df.rename({'new_input': 'input', 'table':'parent'}, axis=1)
    wikiroto_df = wikiroto_df.reset_index(drop=True)
    wikiroto_df = wikiroto_df.drop_duplicates()
    wikiroto_df['title'] = wikiroto_df['input'].apply(get_title)
    wikiroto_df['movie_info'] = wikiroto_df['movie_info'].str.strip()
    wikiroto_df = wikiroto_df.dropna(axis=0)

    movies_rotten_df = pd.read_csv(BASE_DIR + 'rotten/rotten_tomatoes_movies.csv', index_col=0)
    movies_rotten_df = movies_rotten_df.drop_duplicates()
    movies_rotten_df = movies_rotten_df[['movie_title', 'movie_info']]
    movies_rotten_df = movies_rotten_df.rename({'movie_title': 'title'}, axis=1)
    movies_rotten_df['title'] = movies_rotten_df['title'].str.strip()
    movies_rotten_df['movie_info'] = movies_rotten_df['movie_info'].str.strip()
    movies_rotten_df = movies_rotten_df.dropna(axis=0)
    movies_rotten_df = movies_rotten_df.reset_index()

    final_wikiroto_df = movies_rotten_df.merge(wikiroto_df, on=['title', 'movie_info'])
    final_wikiroto_df = final_wikiroto_df.rename({'rotten_tomatoes_link': 'movieID'}, axis=1)
    final_wikiroto_df = final_wikiroto_df.set_index('movieID')[['title', 'input', 'target', 'parent']]
    final_wikiroto_df.to_csv(BASE_DIR + 'appdata/rotten/wikiroto_all.csv', index=True)



def reviews_authors_split(n_authors=2):
    """ Création de jeu de données (train, test, eval) de reviews avec leurs auteurs
    Utile pour la tâche d'authorship attribution

    :param n_authors: nombre d'auteurs
    """
    users_output_df = pd.read_csv(BASE_DIR + 'appdata/rotten/users_output.csv', index_col=0)
    users_output_df = users_output_df[['userID', 'movieID', 'target']]
    users_output_df = users_output_df[users_output_df['userID'] != 'u/nan']
    users_output_df = users_output_df.dropna()

    occurrences = users_output_df['userID'].value_counts()[:n_authors]
    mask = users_output_df.userID.isin(occurrences.index)
    df = users_output_df[mask]
    df = df.rename({'target': 'review'}, axis=1)
    df = df[['review', 'movieID', 'userID']]
    
    sample_size = occurrences.min()
    train_dfs, test_dfs, eval_dfs = [], [], []

    for ui, u in enumerate(df.userID.unique()):
        user_df = df[df['userID'] == u]
        sample_df = user_df.sample(n=sample_size, replace=True)
        sample_df['userNum'] = [ui]*sample_size
        
        train_df = sample_df.sample(frac=.8)
        test_eval_df = sample_df.drop(train_df.index).reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)
        test_df = test_eval_df.sample(frac=.5)
        eval_df = test_eval_df.drop(test_df.index).reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        train_dfs.append(train_df)
        test_dfs.append(test_df)
        eval_dfs.append(eval_df)

    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)
    eval_df = pd.concat(eval_dfs)
        
    return train_df, test_df, eval_df



def to_strap_format(train_df, test_df, eval_df, n_authors=2):
    """ Mise sous format STRAP de jeu de données de review-auteur
    Utile pour le transfert de style

    STRAP : https://github.com/martiansideofthemoon/style-transfer-paraphrase/tree/master

    :param train_df
    :param test_df
    :param eval_df
    :param n_authors : nombre d'auteurs
    """
    dt_dir = f'{BASE_DIR}strap/{n_authors}/' 
    os.makedirs(dt_dir, exist_ok=True)

    data = {'train': train_df, 'test': test_df, 'dev': eval_df}
    for name, df in data.items():
        with open(f'{dt_dir}{name}.txt', "w") as txt:
            txt.write("\n".join(df["review"].tolist()))
        with open(f'{dt_dir}{name}.label', "w") as label:
            label.write("\n".join(df["userID"].tolist()))


# renommage des features pour les infos des films
FEATURES_NAMES = {'movie_title': 'title' , 'movie_info': 'info', 'critics_consensus': 'critic',
                  'content_rating': 'rating', 'genres': 'genres', 'directors': 'directors',
                  'authors': 'authors', 'actors': 'actors', 'production_company': 'production_company', 
                  'audience_status': 'audience_status', 'audience_rating': 'audience_rating','audience_count': 'audience_count'}



def to_userID(username):
    """ formattage du nom des utilisateurs """
    return 'u/' + '_'.join(str(username).split())



def cinema_score(notation):
    """ Normalisation des notes de cinéma """
    cinema_notations = {'A+': 5, 'A': 4, 'A-': 3.7, 'B+': 3.3, 'B': 3,
        'B-': 2.7, 'C+': 2.3, 'C': 2, 'C-': 1.7, 'D+': 1.3, 'D': 1, 'D-': 0.7, 'F': 0}
    return cinema_notations.get(notation, None)


def to_rating(s):
    """ Normalisation des notes de film de Rotten Tomatoes """
    try: 
        nd = s.split('/')
        if len(nd) == 2:
            n, d = nd
            n = float(n)
            d = float(d)
            v = 5*n/d
            if v > 5:
                return None
            return v
        if s[0].upper() in 'ABCDF':
            return cinema_score(s.strip())
        return None
    except:
        return None


def movie2inputformat1(movie):
    """ Format d'input pour le data-to-text """
    input_ = '<movie> '
    
    for att in FEATURES_NAMES.keys():
        input_ += '<' + FEATURES_NAMES[att] + '> '
        input_ += str(movie[att])
        input_ += ' </' + FEATURES_NAMES[att] + '> '
    
    input_ += '</movie>'
    return input_


def movie2inputformat2(movie):
    """ Format d'input pour le data-to-text """
    input_ = ''
    
    for att in FEATURES_NAMES.keys():
        input_ += FEATURES_NAMES[att] + ' : '
        input_ += str(movie[att])
        input_ += ' | '
    
    return input_


def movie2inputformat3(movie):
    """ Format d'input pour le data-to-text """
    input_ = dict()
    
    for att in FEATURES_NAMES.keys():
        input_[FEATURES_NAMES[att]] = str(movie[att])
    
    return input_


def get_title(input1):
    """ Récupération des titres dans les données wikiroto """
    start = input1.index('<movie_title>')
    end = input1.index('</movie_title>')
    if start > 0 and end > 0:
        start += len('<movie_title>')
        return input1[start:end].strip()
    return None


# autres datasets

def movie_n_reviews_to_review(n=10):
    movies_input_df = pd.read_csv(BASE_DIR + 'appdata/rotten/movies_input.csv', index_col=0)
    users_output_df = pd.read_csv(BASE_DIR + 'appdata/rotten/users_output.csv', index_col=0)

    users_output_filtered_df = users_output_df[['movieID', 'userID', 'target']]
    users_output_filtered_df.dropna(inplace=True)

    occurrences = users_output_filtered_df['userID'].value_counts()
    mask = users_output_filtered_df['userID'].isin(occurrences[occurrences >= n+1].index)
    users_output_filtered_df = users_output_filtered_df[mask]

    data_users_df = users_output_filtered_df.merge(movies_input_df, on='movieID')
    data_users_df = data_users_df[['movieID', 'userID', 'input1', 'target']]
    inputs  = []
    outputs = []
    
    for i, v in enumerate(data_users_df.iterrows()):
        dataID, userID, data, out_ = v
        user_examples_df = (data_users_df[data_users_df['userID']==userID]).sample(n=n+1)
        user_examples_df = user_examples_df[user_examples_df['movieID']!=dataID]
        user_corpus = list(user_examples_df['target'])[:n] 
        in_ = 'summarize: ' + ' ' + data + ' ||| '.join(user_corpus)
        inputs.append(in_)
        outputs.append(out_)
        
    df = pd.DataFrame()
    df['input'] = inputs
    df['output'] = outputs
    return df


def movie_review_to_rating():
    movies_input_df = pd.read_csv(BASE_DIR + 'appdata/rotten/movies_input.csv', index_col=0)
    users_output_df = pd.read_csv(BASE_DIR + 'appdata/rotten/users_output.csv', index_col=0)

    users_output_filtered_df = users_output_df[['movieID', 'userID', 'target', 'rating']]
    users_output_filtered_df.dropna(inplace=True)

    data_users_df = users_output_filtered_df.merge(movies_input_df, on='movieID')
    data_users_df = data_users_df[['movieID', 'userID', 'input1', 'target', 'rating']]
    
    def to_input(row):
        return row['input1'] + ' ||| ' + row['target']
    
    df = pd.DataFrame()
    df['input'] = data_users_df.apply(to_input, axis=1)
    df['output'] = data_users_df['rating'].apply(str)
    return df


def review_to_rating():
    users_output_df = pd.read_csv(BASE_DIR + 'appdata/rotten/users_output.csv', index_col=0)

    users_output_filtered_df = users_output_df[['target', 'rating']]
    users_output_filtered_df.dropna(inplace=True)

    df = pd.DataFrame()
    df['input'] = users_output_filtered_df['target']
    df['output'] = users_output_filtered_df['rating']
    return df


def n_review_n_1_rating_user_to_rating(n):
    users_output_df = pd.read_csv(BASE_DIR + 'appdata/rotten/users_output.csv', index_col=0)

    users_output_filtered_df = users_output_df[['userID', 'target', 'rating']]
    users_output_filtered_df.dropna(inplace=True)

    occurrences = users_output_filtered_df['userID'].value_counts()
    mask = users_output_filtered_df['userID'].isin(occurrences[occurrences >= n+1].index)
    users_output_filtered_df = users_output_filtered_df[mask]

    inputs  = []
    outputs = []
    
    for i, v in enumerate(users_output_filtered_df.iterrows()):
        _, (userID, target, rating) = v

        user_examples_df = users_output_filtered_df[users_output_filtered_df['target']!=target]
        user_examples_df = (user_examples_df[user_examples_df['userID']==userID]).sample(n=n)
        user_examples_df = user_examples_df[['target', 'rating']]

        user_examples_input_list = user_examples_df.apply(
            lambda row: f"<review>{row['target']}</review> <rating>{row['rating']}</rating>", axis=1
        ).tolist()
        user_examples_input_list.append(f"<review>{target}</review>")

        user_examples_input = ' ||| '.join(user_examples_input_list)
        inputs.append(user_examples_input)
        outputs.append(rating)
        
    df = pd.DataFrame()
    df['input'] = inputs
    df['output'] = outputs
    return df



if __name__ == '__main__':
    movies_critics()
    wikiroto()
