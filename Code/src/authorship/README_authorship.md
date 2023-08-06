## Personalized data-to-text neural generation : Authorship attribution

Afin d'évaluer la qualité de la personnalisation pour les différentes étapes de création de datasets et modèles 
de data-to-text personnalisé, une des métriques que nous utilisons est l'accuracy donnée par un modèle fine-tuné 
de BERT qui fait de la classification des auteurs.

Le code du modèle est accessible ici [authorship/bert_autorship.py](authorship/bert_autorship.py)

### Apprentissage
Pour entraîner le modèle sur vos données, vous pouvez utiliser le script suivant :

```
python bert_authorship.py\
  [--model_path [MODEL_PATH]\]
  --data_path [DATA_PATH]\
  --batch_size [BATCH_SIZE]\
  --n_authors [N_AUTHORS]\
  --epochs [N_EPOCHS]
  [--evaluation [0/1]]
```

- MODEL_PATH : chemin vers le modèle à utiliser. (paramètre facultatif)
- DATA_PATH : chemin vers le jeu de données à utiliser.
- BATCH_SIZE : taille du batch.
- N_AUTHORS : nombre d'auteurs à considérer.
- N_EPOCHS : nombre d'époques si apprentissage.

### Evaluation
Afin d'évaluer des données sur un modèle pré-entrainé, vous pouvez rajouter l'option : `--evaluation 1`.

### Clustering
Dans [authorship/bert_cls_clustering.py](authorship/bert_cls_clustering.py) vous retrouverez du code pour faire 
du clustering des textes d'utilisateurs avec BERT.
Vous pouvez ainsi réutiliser vos modèles pré-entraînés.

Vous pouvez utiliser le script suivant pour visualiser le clustering :
```
python bert_cls_clustering.py\
	--model_path [MODEL_PATH]\
	--data_path [DATA_PATH]\
	--n_authors [N_AUTHORS]\
	--out_filename [OUT_FILENAME]
```

- OUT_FILENAME : fichier dans lequel sauvegarder l'image du clustering.

### Expérimentations

#### Résultats
Nous avons entraînés des modèles d'attribution d'auteurs pour 2, 3, 5, 10 auteurs.

Voici les résultats au bout de 20 époques d'apprentissage pour chacun des modèles.

| Nombre auteurs  | Accuracy |
| ------------- | ------------- |
| 2 | 0.885 |
| 3 | 0.772 |
| 5 | 0.717 |
| 10 | 0.712 |

#### Clustering
Voici quelques résultats du clustering sur les données de Rotten Tomatoes :

| Nombre auteurs  | ARI score |
| ------------- | ------------- |
| 2 | 0.842 |
| 3 | 0.709 |
| 5 | 0.717 |

##### **2 auteurs**
![Clustering sur les reviews de Rotten Tomatoes - 2 Auteurs](/Code/src/authorship/out_clustering/clustering_authors_2.png)
##### **3 auteurs**
![Clustering sur les reviews de Rotten Tomatoes - 3 Auteurs](/Code/src/authorship/out_clustering/clustering_authors_3.png)
##### **5 auteurs**
![Clustering sur les reviews de Rotten Tomatoes - 3 Auteurs](/Code/src/authorship/out_clustering/clustering_authors_5.png)

### Grid search avec TF-IDF
Dans [authorship/bert_cls_clustering.py](authorship/bert_cls_clustering.py), vous retrouvez du code pour faire de la recherche
exhaustive des paramètres optimaux pour la classification d'auteurs avec du TF-IDF.
