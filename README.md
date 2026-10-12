# Neural Network From Scratch

Implémentation d'un réseau de neurones multicouche et de la rétropropagation (backpropagation) **à la main**, en Python pur : aucune librairie de deep learning, seulement `random`, `math` et `csv`.

Le travail est regroupé dans un seul notebook : [`Neural network from scratch.ipynb`](Neural%20network%20from%20scratch.ipynb) (Lab 2 de Deep Learning).

## Représentation du réseau

- un réseau est une **liste de couches** ;
- une couche est une **liste de neurones** ;
- un neurone est un **dictionnaire** contenant ses poids (`weights`, le biais est le dernier poids), puis `output` et `delta` après passage avant / arrière.

## Contenu du notebook

| Partie | Contenu |
|---|---|
| 1-2 | Initialisation du réseau (`initialize_network`), reproductibilité avec une graine |
| 3-4 | Activation d'un neurone et fonction de transfert (sigmoïde) |
| 5 | Propagation avant (`forward_propagate`) |
| 6-7 | Dérivée de la sigmoïde et rétropropagation de l'erreur (`backward_propagate_error`) |
| 8-9 | Mise à jour des poids et entraînement par descente de gradient (`train_network`) |
| 10 | Prédiction (`predict`) |
| 11 | Bonus : activation par couche (sigmoïde, tanh, relu, linéaire, softmax) et choix de la loss (MSE ou entropie croisée) |
| 11 (Q20) | Comparaison de configurations sur 3 problèmes : classification binaire, classification à 3 classes, régression |
| 12 | Application au **Seeds Dataset** (3 variétés de grains de blé) |
| 13 | Bonus : plusieurs couches cachées |

## Résultats principaux

- **Données jouet** : l'erreur diminue bien au fil des epochs et le réseau classe correctement le dataset d'exemple.
- **Comparaison de configurations** : sur le cercle (non linéaire), tanh converge plus vite que la sigmoïde + MSE du TD ; sur 3 classes qui se chevauchent, toutes les configurations sont proches ; en régression linéaire, un réseau linéaire atteint une RMSE du niveau du bruit ajouté.
- **Seeds Dataset** : environ **90,5 % d'accuracy sur le test** (baseline à 31,7 %), avec standardisation des features, 10 neurones cachés, sigmoïde et erreur quadratique.
- **Couches multiples** : `forward_propagate` et `backward_propagate_error` fonctionnent sans modification pour un nombre quelconque de couches, mais n'apportent rien de plus sur ce dataset.

## Lancer le notebook

```bash
pip install jupyter
jupyter notebook "Neural network from scratch.ipynb"
```

Le notebook n'utilise que la bibliothèque standard de Python. Pour la partie Seeds, le fichier de données du dataset (`seeds_dataset.csv`) doit être fourni à côté du notebook (ou uploadé sur Colab) ; il n'est pas inclus dans ce dépôt.

## Auteur

Baptiste Wehry
