## Personalized data-to-text neural generation : Données

### Rotten Tomatoes movies and critic reviews dataset

Dans le cadre de ce stage, nous utilisons le jeu de données **Rotten Tomatoes movies and critic reviews dataset**. Il s'agit d'un jeu de données dans lequel on retrouve deux datasets :

- **Dataset d'informations sur des films** : identifiant du film, titre, auteurs, acteurs, directeurs, genres, audience, etc.
- **Dataset de reviews utilisateurs sur les films** : nom de l’utilisateur, identifiant du film, note, contenu, date, etc.

Le jeu de données est accessible depuis le lien suivant : https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset

Vous pouvez ainsi le télécharger et le mettre dans le dossier `Data/rotten/`.

### WikiRoto

Le stage ayant donné lieu à un travail précédent afin de résoudre la tâche de data-to-text personnalisé, le jeu de données **WikiRoto**, fusionnant Wikipédia et Rotten Tomatoes, a été proposé. 

Rotten Tomatoes, manquant de description textuelle des films, n’étant pas un dataset de data-to-text. WikiRoto ajoute donc à chaque film une description textuelle (qui est le premier paragraphe de la page Wikipedia du film), afin de faire du data-to-text.

Le jeu de données est accessible depuis le lien suivant : https://github.com/tramy1258/wikiroto/tree/main/data

Vous pouvez ainsi le télécharger et le mettre dans le dossier `Data/wikiroto/`.
Dans le cadre de notre travail, nous nous sommes uniquement intéressé aux fichiers `wikiroto_[split]_with_table.csv`de ce jeu de données.

### Autres datasets

Une fois que vous avez obtenus les jeux de données Rotten Tomatoes et Wikiroto, exécutez le programme [Code/src/datasets/datasets.py](https://github.com/BenKabongo25/personalized-data-to-text-neural-generation/blob/main/Code/src/datasets/datasets.py) pour ainsi créer différents jeux de données utilisées dans notre étude.

Depuis `Code/src/datasets/`, lancez donc `python datasets.py`

Modifier les paramètres par défaut des fonctions du programme vous permettra de produire d'autres variantes de ces jeux de données.
