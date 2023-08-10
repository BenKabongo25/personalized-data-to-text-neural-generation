## Personalized data-to-text neural generation : Transfert de style

### STRAP : Style Transfer via Paraphrasing

En utilisant la méthode **STRAP** 
([papier](https://arxiv.org/abs/2010.05700), [github](https://github.com/martiansideofthemoon/style-transfer-paraphrase/tree/master)),
il est possible d'entraîner pour différents utilisateurs des modèles capables de paraphraser n'importe quel texte dans le style
d'écriture de ces utilisateurs. Il faut diposer en entrée d'une base de textes pour les différents utilisateurs, comme par exemple
des reviews de films tel que dans le dataset [Rotten Tomatoes](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset).

Il est difficile de trouver un dataset de data-to-text personnalisé. L'idée pour créer des datasets de data-to-text est de partir
de datasets de data-to-text et de bases de textes d'utilisateurs, d'entraîner des modèles utilisateur de transfert de style avec STRAP,
puis enfin de dériver pour chaque utilisateur une version personnalisée de chaque description de données non personnalisée.

Vous retrouverez toutes les informations nécessaires à l'utilisation de STRAP dans le repository github associé.

Voici quelques indications d'utilisation :

- Récupérez le repository :
  ```
  git clone https://github.com/martiansideofthemoon/style-transfer-paraphrase.git
  ```
- Placez-vous dans le nouveau répertoire et suivez les indications d'installations des dépendances :
  ```
  pip install torch torchvision
  pip install -r requirements.txt
  pip install --editable .
  
  cd fairseq
  pip install --editable .
  ```
- Ensuite, il faut mettre votre jeu de données des textes des utilisateurs sous le format STRAP.
  - Diviser le jeu de données en ensembles d'apprentissage (`train`), test (`test`) et évaluation (`dev`)
  - Pour chaque ensemble, créer les fichiers d'exemples et de label (`train.txt`, `train.label`, ...) en respectant la consinge d'un exemple par ligne et que chaque fichier *.txt* et *.label* doivent avoir le même nombre de lignes.
  - Le format des fichiers *.txt* doit ainsi être le suivant :
    ```
    Texte 1
    Texte 2
    ...
    Texte N
    ```
  - Le format des fichiers *.label* doit ainsi être le suivant :
    ```
    auteur du texte 1
    auteur du texte 2
    ...
    auteur du texte N
    ```
- Mettez les fichiers *.txt* et *.label* dans un dossier, de préférence, un sous-dossier du dossier `datasets` du repository.
- La suite des étapes pour entraîner votre modèle de transfert de style par génération de paraphrase avec STRAP est ainsi décrite [ici](https://github.com/martiansideofthemoon/style-transfer-paraphrase/tree/master#custom-datasets)

A la fin, vous obtenez un modèle capable de paraphraser n'importe quelle phrase avec les styles pour lesquels il a été entraînés.
Les auteurs de STRAP mettent également à disposition différents outils d'évalutaion.

### Mise en application

Dans notre cas, nous avons utilisé le modèle STRAP pour entraîner pour quelques utilisateurs du dataset Rotten Tomatoes des modèles de génération paraphrases.

Vous pouvez ainsi retrouver dans le dossier [Code/src/style_transfer](Code/src/style_transfer) tous les utilitaires nécessaires à la génération de descriptions personnalisées par utilisateur ainsi que des utilitaires d'évaluation.

#### Génération des descriptions textuelles personnalisées
Une fois le modèle entraîné, nous générons pour notre dataset de data-to-text les descriptions personnalisées pour l'utilisateur associé au modèle.

```
python paraphraser.py\
  --model_dir [MODEL_PATH]\
  --top_p_value [TOP_P_VALUE]\
  --n_samples [N_SAMPLES]\
  --input_dataset_path [DATA_TO_TEXT_DATASET_PATH]\
  --input_feature_name [FEATURE_NAME]\
  --output_dataset_path [OUTPUT_DATASET_PATH]
```

- MODEL_PATH : contient le modèle fine-tuné de STRAP pour un utilisateur.
- TOP_P_VALUE : compris entre 0 et 1, est un hyper-paramètre STRAP qui fait varier l'aléatoire dans la génération. D  ns le papier STRAP, les valeurs couramment utilisées sont 0.0, 0.6 et 0.9.
- N_SAMPLES : le nombre de texte à générer pour chaque exemple.
- DATA_TO_TEXT_DATASET_PATH : le chemin vers votre dataset de data-to-text (sous format .csv).
- FEATURE_NAME : la colonne de votre dataset à paraphraser : il s'agit de la colonne de description textuelle non personnalisée de votre dataset.
- OUTPUT_DATASET_PATH : le chemin du jeu de données de sorties, qui renvoie pour chaque exemple de description non personnalisée, *N_SAMPLES* de descriptions personnalisées pour l'utilisateur du modèle.

#### Evaluation

Les auteurs de STRAP mettent à disposition différents outils et métriques d'évaluation du transfert de style. Nous les utilisons, en plus d'autres procédures d'évaluation.

#### Similarité
Les auteurs de STRAP utilisent la métrique de similarité proposée par [Wieting et al. 2019](https://aclanthology.org/P19-1427/).
Vous pouvez accéder au code pour la similarité avec STRAP [ici](https://github.com/martiansideofthemoon/style-transfer-paraphrase/tree/master/style_paraphrase/evaluation/similarity).

Nous mettons à disposition un script d'évaluation de transfert de style en utilisant cette métrique de similarité.
Vous pouvez y accéder depuis [Code/src/style_transfer/similarity_eval.py](Code/src/style_transfer/similarity_eval.py)

Voici comment lancer le script :
```
python similarity_eval.py --data_path [OUTPUT_DATASET_PATH]
```

#### Authorship attribution

La personnalisation de textes étant au centre de notre travail, en plus des outils d'évaluation proposés par les auteurs de STRAP,
nous proposons de mesurer la qualité du transfert de style avec un modèle d'authorship attribution avec BERT. 

Nous vous invitons à vous référer à la page [Code/src/authorship/README_authorship.md](../authorship/README_authorship.md) 
pour plus d'informations.

### Résultats

