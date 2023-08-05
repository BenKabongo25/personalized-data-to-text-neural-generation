# Personalized data-to-text neural generation

## Introduction

Dans ce stage, effectué au MLIA/ISIR, nous nous intéressons au **data-to-text**, un domaine du traitement automatique des langues où la tâche consiste à générer des phrases en langage naturel à partir de données structurées ou semi-structurées. 

L'objectif du stage est de développer un système neuronal de conversion de données en texte capable de personnaliser la génération de texte. 

### Notions

- **Data-to-text** : génération en langage naturel d’une description textuelle pour des données structurées ou semi-structurées (graphes, tables, etc.).
- **Personnalisation** : tenir compte des préférences de l’utilisateur.
- **Style d’écriture** : façon d’écrire d’un utilisateur : expressions, vocabulaire, figures de style, etc.
- **Description personnalisée** : description textuelle écrite avec le style d’un utilisateur.

- **Data-to-text personnalisé** : génération en langage naturel d’une description textuelle personnalisée pour des données (semi-)structurées
- **Dataset de data-to-text personnalisé** : {(xi,ui,yi)} i=1..N
  - *xi* : données (semi-)structurées
  - *ui* : informations sur l’utilisateur
  - *yi* : description textuelle personnalisée pour *ui* de l’exemple *xi*
*Il n’existe pas de dataset de data-to-text personnalisé.*

### Apports

- Présentation de méthologie de création de datasets de data-to-text personnalisé
- Proposition d'un framework générique pour le data-to-text personnalisé
- Mise à dispotion d'un dataset de data-to-text personnalisé (dérivé de WikiRoto et de Rotten Tomatoes)


## Datasets

Dans ce travail, nous utilisons les datasets [Rotten Tomatoes Movies](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) et [WikiRoto](https://github.com/tramy1258/wikiroto/tree/main/data). Nous vous invitons à vous référer à la page [Data/README_Data.md](Data/README_Data.md) qui explique les étapes pour les récupérer et pouvoir reproduire nos expérimentations.

Nous mettons également à disposition un jeu de données de data-to-text personnalisé [PersoWikiRoto](Data/appdata/perso_wiki_roto).

### Création de dataset

Voici les étapes à suivre pour la création automatique d'un dataset de data-to-text personnalisé.

- **Entrées** :
  - **Dataset de data-to-text** : Vous aurez besoin d'un dataset de data-to-text basique (ex: WikiRoto)
  - **Dataset de styles d’utilisateurs** : Vous aurez également besoin de bases de textes des utilisateurs (ex: Rotten Tomatoes)
- **Sortie** : Dataset de data-to-text personnalisé.
  L'idée est de dériver pour chaque exemple non personnalisé *(xi, yi)* un exemple personnalisé *(xi, u, yui)* pour chaque utilisateur *u*.

Ensuite, il faut faire du transfert de style non supervisé. Notre choix sur le modèle de transfert de style non supervisé s'est porté sur le modèle **STRAP** présenté dans le papier suivant [Reformulating Unsupervised Style Transfer as Paraphrase Generation](https://arxiv.org/abs/2010.05700). 

Vous pouvez dès à présent accéder au repository github associé depuis l'adresse suivante : https://github.com/martiansideofthemoon/style-transfer-paraphrase/tree/master

Ils décrivent les étapes nécessaires au fine-tuning d'un modèle GPT-2 sur vos données. Une fois le modèle appris, vous pourrez l'utiliser pour dériver pour chaque utilisateur une description personnalisée depuis les descriptions non personnalisées de votre dataset de data-to-text.

Voici cependant quelques indications :

- Installez le repository :
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
- Ensuite, il faut mettre le jeu de données des textes des utilisateurs sous le format STRAP.
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
    auteur du texte 1
    ...
    auteur du texte N
    ```
- Mettez les fichiers *.txt* et *.label* dans un dossier, de préférence, un sous-dossier du dossier `datasets` du repository.
- La suite des étapes pour entraîner votre modèle de transfert de style par génération de paraphrase avec STRAP est ainsi décrite ici : https://github.com/martiansideofthemoon/style-transfer-paraphrase/tree/master#custom-datasets

A la fin, vous obtenez un modèle capable de paraphraser n'importe quelle phrase avec les styles pour lesquels il a été entraînés.
Les auteurs de STRAP mettent également à disposition différents outils d'évalutaion de la méthode.

### Cas concret

Dans notre cas, nous avons utilisé le modèle STRAP pour entraîner pour quelques utilisateurs du dataset Rotten Tomatoes des modèles de génération paraphrases.


## Authorship attribution

Afin d'évaluer la qualité de la personnalisation pour les différentes étapes de création de datasets et modèles de data-to-text personnalisé, une des métriques que nous utilisons est l'accuracy donnée par un modèle fine-tuné de BERT qui fait de la classification des auteurs.

Nous avons entraînés des modèles d'attribution d'auteurs pour 2, 5, 10, 20, 50 et 80 auteurs.
Le code du modèle est accessible ici [Code/src/authorship/bert_autorship.py](Code/src/authorship/bert_autorship.py)

Pour entraîner le modèle sur vos données, vous pouvez utiliser le script suivant :

```
python bert_authorship.py\
  --model_path [. | MODEL_PATH]\
  --data_path [DATA_PATH]\
  --batch_size [BATCH_SIZE]\
  --n_authors [N_AUTHORS]\
  --epochs [N_EPOCHS]
```

Afin d'évaluer des données sur un modèle pré-entrainé, vous pouvez utiliser le script suivant : 
```
python bert_authorship.py\
  --model_path [MODEL_PATH]\
  --data_path [DATA_PATH]\
  --batch_size [BATCH_SIZE]\
  --n_authors [N_AUTHORS]\
  --evaluation True
```

## Data-to-text personnalisé








