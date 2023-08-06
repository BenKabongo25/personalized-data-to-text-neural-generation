# Ben Kabongo
# Personnalized data-to-text neural generation
# ISIR/MLIA, 2023


import ast
import pandas as pd
import sys


def to_authorship_data(args):
    """ Mise en forme de données au format pour l'authorship attribution

    :param data_df: pd.DataFrame
    :param feature_name: colonne des textes
    :param output_data_path: chemin du dataset de sortie
    """
    data_df = pd.read_csv(args.data_path, index_col=0)

    try :
        data_df[args.feature_name] = data_df[args.feature_name].apply(ast.literal_eval)
    except:
        data_df[args.feature_name] = data_df[args.feature_name].apply(lambda x: [str(x)])

    texts = []
    for text_list in data_df[args.feature_name].tolist():
        texts.extend(text_list)

    pd.DataFrame({"review":texts}).to_csv(args.output_data_path, index=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, type=str)

    # to_authorship_data
    parser.add_argument("--data_path", type=str, default='')
    parser.add_argument("--feature_name", type=str, default='')
    parser.add_argument('--output_data_path', type=str, default='')

    args = parser.parse_args()

    if "to_authorship_data" in args.action.lower().strip():
        assert args.data_path != ""
        assert args.feature_name != ""
        assert args.output_data_path != ""
        to_authorship_data(args)

    else:
        print("Unknown action")
        sys.exit()
