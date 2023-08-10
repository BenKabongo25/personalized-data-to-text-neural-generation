# Personalized data-to-text neural generation

## Introduction

Dans ce stage, effectué au sein de l'équipe MLIA du laboratoire ISIR, nous nous intéressons au data-to-text, un domaine du traitement automatique des langues où la tâche consiste à générer des phrases en langage naturel à partir de données structurées ou semi-structurées. 

L'objectif du stage est de développer un système neuronal de conversion de données en texte capable de personnaliser la génération de texte. 

### Notions

- **Data-to-text** : génération en langage naturel d’une description textuelle pour des données structurées ou semi-structurées (graphes, tables, etc.).
- **Personnalisation** : tenir compte des préférences de l’utilisateur.
- **Style d’écriture** : expressions, vocabulaire, figures de style d'un utilisateur.
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

## Authorship attribution

Afin d'évaluer la qualité de la personnalisation pour les différentes étapes de création de datasets et modèles de data-to-text personnalisé, une des métriques que nous utilisons est l'accuracy donnée par un modèle fine-tuné de BERT qui fait de la classification des auteurs.

Nous avons entraînés le modèle pour quelques nombres d'auteurs : entre 2 et 80.
Le code du modèle est accessible ici [Code/src/authorship/bert_autorship.py](Code/src/authorship/bert_authorship.py)

Vous pouvez vous référer à la page [Code/src/authorship/README_authorship.md](Code/src/authorship/README_authorship.md) pour avoir plus d'infos sur nos expérimentations et des indications sur le lancement des scripts.

## Data-to-text personnalisé

### Création de dataset

Voici les étapes à suivre pour la création automatique d'un dataset de data-to-text personnalisé.

- **Entrées** :
  - **Dataset de data-to-text** : Vous aurez besoin d'un dataset de data-to-text basique (ex: WikiRoto)
  - **Dataset de styles d’utilisateurs** : Vous aurez également besoin de bases de textes des utilisateurs (ex: Rotten Tomatoes)
- **Sortie** : Dataset de data-to-text personnalisé.
  L'idée est de dériver pour chaque exemple non personnalisé *(xi, yi)* un exemple personnalisé *(xi, u, yui)* pour chaque utilisateur *u*.

Pour obtenir en sortie un dataset de data-to-text personnalisé, il faut faire du transfert de style non supervisé. Notre choix s'est porté sur un modèle de transfert de style non supervisé, le modèle **STRAP**, présenté dans le papier suivant [Reformulating Unsupervised Style Transfer as Paraphrase Generation](https://arxiv.org/abs/2010.05700). 

Vous pouvez également accéder au repository github associé depuis l'adresse suivante : https://github.com/martiansideofthemoon/style-transfer-paraphrase/tree/master

Ils décrivent les étapes nécessaires au fine-tuning d'un modèle GPT-2 sur vos données. Une fois le modèle appris, vous pourrez l'utiliser pour dériver pour chaque utilisateur une description personnalisée depuis les descriptions non personnalisées de votre dataset de data-to-text.

Vous trouverez également quelques indications ici : [Code/src/style_transfer/README_style_transfer.py](Code/src/style_transfer/README_style_transfer.py).

### Modèle générique pour le data-to-text personnalisé

Après la création d'un dataset de data-to-text personnalisé en suivant les étapes décrites dans les sections précédentes, on peut ensuite passer à l'entraînement d'un modèle de data-to-text personnalisé.

Dans notre cas, nous fine-tunons différentes variantes du modèle T5 (t5-small, t5-base, t5-large). Le processus d'apprentissage du modèle générique que nous proposons reste très proche du paradigme d'apprentissage supervisé avec les modèles de langue dans les tâches de data-to-text, de summurization et autres.

*Des modèles beaucoup plus spécifiques à la tâche du data-to-text personnalisé peuvent être imaginés, tout en faisant varier les paramètres, tenant par exemple compte de l'autorship attribution, des embeddings utilisateurs, de l'analyse des sentiments ou d'un système de recommandation. Ces pistes restent à explorer.*

Vous pouvez retrouver le code de l'approche générique que nous proposons ici [Code/src/pdtt/dtt_t5.py](Code/src/pdtt/dtt_t5.py)

Nous vous invitons à vous référer à la page [Code/src/pdtt/README_pdtt.md](Code/src/pdtt/README_pdtt.md) pour obtenir les informations relatives à la configuration des données et au lancement des scripts.
