<div style="text-align:center;font-size:30px">﷽</div>











<div style="text-align:center"><h1>NLP avec Spark</h1></div>
<div style="text-align:center"><h2>Analyse des sentiments dans les commentaires</h2></div>













Travail fait par  **CHEBBAH Mehdi** et **HAMMAS Ali Cherif**

<div class="page-break"></div>



<h1>Table de contenu</h1>

[TOC]

<div class="page-break"></div>

# C'est quoi NLP ?

Le NLP (*Natural Language Processing*) est une **branche de l’intelligence artificielle** qui s’occupe particulièrement du traitement du langage écrit aussi appelé avec le nom français TALN (traitement automatique du langage naturel) ou TLN. En bref, c’est tout ce qui est lié au langage humain et au traitement de celui-ci par des outils informatiques.

Le NLP peut être divisé en 2 grandes parties, le **NLU** (*Natural Language Understanding*) et le **NLG** (*Natural Language Generation*).

-  Le premier concerne toute la partie « compréhension » du texte, prendre un texte en entrée et pouvoir en ressortir des données. Ce type est très utilisé dans:

   -  Les filtres spam dans les émail.
   -  La collection des avis des utilisateur pour un produit.
   -   Les chat-bots.
   -  Les recommandations des produits dans le E-commerce.
   -  etc

-  Le second, est générer du texte à partir des données, pouvoir construire des phrases cohérentes de manière automatique. Ce type est très utilisé dans:
   -  La traduction automatique.
   -  Les chat-bots.
   -  Les applications de type *Siri* ou *Google Assistent*.
   -  La génération automatique des rapport financières.
   -  etc



# Le prétraitement des données en NLP

## Théoriquement: 

Il existe plusieurs sources de données que peuvent servir dans le process du NLP, par exemple **le web scraping**, **les réseaux sociaux**, **Les bases de données**, **Les données en temps réel (Streaming)**, ...etc

Et selon le source des données (donc ça qualité) on fait le prétraitement. Mais globalement il existe 3 phases:

+  **Traiter les valeurs manquantes**: C'est une phase très importante dans la préparation des données pour tous type de modèle (NLP ou autre). Il existe plusieurs approches pour régler le problème des valeurs manquantes sans les supprimer (car la suppression de ces valeur peut biaiser le modèle)
+  **Annoter les données**: Cette phase en général est faites en utilisant l'intelligence humain (plusieurs humains lisent les données et les classifient selon des classes prédéfinis), Ou en utilisant des algorithmes de machine Learning non-supervisé (ou semi-supervisé) pour faire l'annotation.
+  **nettoyage des données**: Cette phase dépend des sources de données et de ces qualité et aussi de l'objective de l'analyse. On peut (comme on peut pas) trouver les traitements suivantes:
   +  **Éliminer les tag HTML**: (pour les données scrapées)
   +  **Supprimer les espaces blancs supplémentaires** (Supprimer le bruit)
   +  **Convertir tous les caractères en minuscules**
   +  **Suppression de mots vides**: (dépend de la langue) *par exemple: pour, a, on, de, dans, ...*
   +  **Supprimer la ponctuation**
   +  **lemmatisation**: transformation de chaque mot a ça forme de base *par exemple (fait => faire), ...)*
   +  **Tokenisation**: découper les documents (textes) en en plus petits morceaux appelés jetons.
   +  **TF-IDF**: (de l'anglais *term frequency-inverse document frequency*) est une mesure statistique permet d'évaluer l'importance d'un terme contenu dans un document, relativement à une collection. Le poids augmente proportionnellement au nombre d'occurrences du mot dans le document. Il varie également en fonction de la fréquence du mot dans la collection.

## En pratique

On va tout d'abord créer une session Spark

```python
import findspark
findspark.init('/opt/spark')

from pyspark import SparkContext
sc = SparkContext("local", "NLP App")
```

On a utiliser `findspark` pour initialiser un environnement `Spark` dans l'environnement `conda`. Puis initialiser un `SparkContext`.

Maintenant on peut importer les data-sets

```python
dataset_path = '/path/to/datasets/files/folder/'
stopwords_path = '/path/to/stopwords/file/folder/'
data = sc.textFile(dataset_path + "*.txt").map(lambda line: line.split("\t"))
stopwords = sc.textFile(stopwords_path + "english").collect()
```

Puis on prépare les données pour le prétraitement (on sépare les documents des annotations)

```python
documents = data.map(lambda line: line[0])
labels = data.map(lambda line: line[1])
```

Le prétraitement des données suit le schéma suivant:

1.  Convertir les documents en minuscule.

2. Supprimer la ponctuation.

3. Supprimer les espaces blancs supplémentaires.

4. Tokenisation (en mots).

   ```python
   def lower_clean_str(x):
     punc='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
     lowercased_str = x.lower()
     for ch in punc:
       lowercased_str = lowercased_str.replace(ch, ' ')
     return lowercased_str.strip()
   
   import re
   documents = documents.map(lambda line: re.sub(" +", ' ', lower_clean_str(line)).split(" "))
   ```

5. Supprimer les mots vides.

   ```python
   def removeStopWords(words, stopwords):
       return [x for x in words if x not in stopwords]
   
   documents = documents.map(lambda line: removeStopWords(line, stopwords))
   documents.take(2)
   # Resultats:
   # [['slow','moving','aimless','movie','distressed','drifting','young','man'],
   # ['sure','lost','flat','characters','audience','nearly','half','walked']]
   ```

6. Appliquer TF-IDF

   ```python
   from pyspark.mllib.feature import HashingTF, IDF
   hashingTF = HashingTF()
   tf = hashingTF.transform(documents)
   
   tf.cache()
   idf = IDF().fit(tf)
   tfidf = idf.transform(tf)
   
   idfIgnore = IDF(minDocFreq=2).fit(tf)
   tfidfIgnore = idfIgnore.transform(tf)
   ```

7. Préparer les structures de données nécessaires pour la création du modèle: pour la phase *training* le modèle prend comme *input* la structure `RDD of LabeledPoints`

```python
from pyspark.mllib.regression import LabeledPoint
tfidfWithIndexes = tfidfIgnore.zipWithIndex().map(lambda x: (x[1], x[0]))
labelsWithIndexes = labels.zipWithIndex().map(lambda x: (x[1], x[0]))
labelsWithIndexes.take(5)
trainingData = tfidfWithIndexes.join(labelsWithIndexes).map(lambda x: LabeledPoint(x[1][1], x[1][0]))
training, test = trainingData.randomSplit([0.7, 0.3])
```



# Les types d'analyse NLP

Il existe deux grandes catégories du NLP (selon l’algorithme d'analyse):

>  ##### Remarque:
>
>  Dans ce travail on va utiliser (Pratiquement) un modèle statistique qui se base sur des algorithmes supervisés. Mais les méthodes de Deep Learning sont expliquées théoriquement seulement.  



## Basé Machine Learning (statistique)

Cette catégorie des algorithmes est la plus utiliseé vu sa simplicité, on peut trouver deux approches utilisées pour des objectives différents:

### Les méthodes supervisées

Nécessite que la data-set soit annotée. Ces méthodes sont utilisées pour l'analyse des sentiment, extraction des information des données, texte classification, ...etc. Les algorithmes les plus utilisés sont:

+  *Support Vector Machines (SVM)*
+  *Naïve Bayes*
+  *Champ aléatoire conditionnel (CRF)*
+  *Réseaux bayésien*
+  ... etc

### Les méthodes non-supervisées

Ne nécessite pas l'annotation du data-sets. Et elles sont utilisées pour la morphologie, la segmentation des phrases, classification des textes, désambiguïsation lexicale, traduction, ...etc. 



## Basé Deep Learning

Dernièrement l'apprentissage en profondeur (Deep Learning) est devenu l'une des méthodes les plus utilisées pour résoudre des problèmes d'apprentissages complexes puisqu'il permet non seulement d'étendre les limites des modèles précédemment vu (les modèles statistiques) mais également de donner parfois d’excellents résultats selon le contexte ou il est utilisé.

Même s'il existe plusieurs recherches établies jusqu’à maintenant mais il y a deux approches qui sont largement utilisées pour le NLP (`CNN` et `RNN`).

### `CNN`

**Les `CNN`** (Réseau neuronal convolutif) est un type de réseau de neurones artificiels (`ANN`) qui comme son nom l'indique est un ensemble de neurones (représentent des poids) qui sont classé en couches (`layers`). La principale différence entre le `CNN` et `ANN` est que contrairement au `ANN` qui repose sur des fonction d'activations pour passer d'une couche a une autre, le `CNN` applique des filtres sur les données en entrées pour extraire les features.

![](rapport.assets/0-NYqnYrLeC1J0Bon5.png)

Le principe de cette approche consiste tout d'abord a découper les phrases en mots qui sont par la suite transformés en une matrice d'intégrations de mots (la matrice qui est en entrée)  de dimension d, puis juste après découpe la matrice en entrée en plusieurs régions pour qu'ensuite on applique les différents filtres sur les matrices qui les correspondent, puis une étape cruciale appelé le `pooling` doit être lancée, cette dernière consiste a effectuer des transformations sur les matrices résultantes pour être égales a une taille déjà prédéfinie. On peut identifier deux principales raisons a cette méthode : 

1. Donner une taille fixe a la matrice de sorties
2. Réduire la taille de la matrice de sorties 

et ce quelque soit la taille de la matrice en entrée. Au final on aura la représentation de la phase finale qui représente un classificateur basé sur les features extraites.

![](rapport.assets/Screenshot from 2020-06-07 19-19-17.png)

En général, les CNN sont efficaces car ils peuvent extraire des indices sémantiques lorsqu'il s'agit du contexte globale , mais ils ont du mal à préserver l'ordre séquentiel et à modéliser les informations contextuelles à longue distance. Les modèles récurrents sont mieux adaptés à ce type d'apprentissage et ils sont abordés ci-après.

### `RNN`

**Les `RNN`s** (Réseau de neurones récurrents) sont des réseaux de neurones conçus spécialement pour être très performant lorsqu'il s'agit des données en séquences et ceci leur donne un très grand avantage pour le NLP. Les `RNN`s sont très bons pour le traitement des données en séquence puisqu'ils reposent sur un concept appelé ` Sequential Memory` , ce dernier consiste a apprendre les choses en se basant sur un mécanisme qu'on utilise nous les humains énormément dans notre vie, voici l'une des méthodes les plus efficace pour le modéliser.

Donc si on demande a quelqu'un de réciter les alphabets dans l'ordre normal 

` A B C D E F G H I J K L M N O P Q R S T U V W X Y Z `

Il n'aura aucune difficulté a le faire mais si on lui donne un alphabet au milieu et on lui demande de compléter la séquence il aura quelques difficultés mais juste après il va pouvoir les réciter très rapidement et ceci puisque cette personne avait appris les alphabets en séquence.

Voici un défis plus difficile qui consiste a réciter les alphabet dans l'ordre inverse 

` Z Y X W V U T S R Q P O N M L K J I H G F E D C B A `

cela devient encore plus difficile même si tout le monde connaît les alphabets, juste le fait que le séquencement n'est pas respecté rend la tache difficile et même parfois impossible, la même chose est appliqué pour les `RNN`s.

Pour pouvoir inclure ce concept dans ces réseau de neurones il suffit juste de prendre un `ANN` simple, puis dans chaque couche on crée un arcs qui permet de rattacher la sortie a l'entrée, grâce a cela les données de l'état précédent vont être ajoutées aux données de l’état courant

![](rapport.assets/Screenshot from 2020-06-07 17-45-52.png)

Donc le principale avantage du Réseau de neurones récurrents et la possibilité de donner un sens aux séquencements de mots pour savoir avec précision  le sujet et le contexte de cette phrase, le meilleur exemple ou l'on peut appliquer ce modèle est un chat-bot car il permet facilement de comprendre ce que l'utilisateur veut a travers la phrase qu'il a exprimé en entrée et par la suite le modèle pourra définir la meilleur réponse et la plus convenable par rapport a ce qui a été demandé.



# La construction et teste des modèles

On va construire 3 modèles

+  Le premier c'est naïve bayes

```python
from pyspark.mllib.classification import NaiveBayes
model1 = NaiveBayes.train(training, 5.0)
# le 2eme parametre c'est le parametre de lissage=5.0
predictionAndLabel_NB = test.map(lambda p: (model1.predict(p.features), p.label))
```

+  Le deuxième c'est SVM

```python
from pyspark.mllib.classification import SVMWithSGD
model2 = SVMWithSGD.train(training, iterations=100)
predictionAndLabel_SVM = test.map(lambda p: (model2.predict(p.features), p.label))
```

+  Le troisième c'est RF

```python
from pyspark.mllib.tree import RandomForest
model = RandomForest.trainClassifier(training, numClasses=2, numTrees=5, categoricalFeaturesInfo={}, featureSubsetStrategy="auto", maxDepth=4, maxBins=32)
predictions = model.predict(test.map(lambda x: x.features))
predictionAndLabel_RF = test.map(lambda lp: lp.label).zip(predictions)
```

>  ##### Remarque:
>
>  Les paramètres utilisées dans la création des modèles sont choisis de tel sort que **l'accuracy** des modèles soit la meilleure possible.



# Comparaison entre les modèles

On peut comparer **l'accuracy** des modèles

```python
def accuracy(predictionAndLabel):
    return 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()

# NB
print('model accuracy {}'.format(accuracy(predictionAndLabel_NB)))
# Resultats: model accuracy 0.7817982456140351

# SVM
print('model accuracy {}'.format(accuracy(predictionAndLabel_SVM)))
# Resultats: model accuracy 0.7872807017543859

# RF
print('model accuracy {}'.format(accuracy(predictionAndLabel_RF)))
# Resultats: model accuracy 0.4868421052631579
```

On remarque clairement que les modèles les plus adaptés au traitement de langage naturel sont **Naïve Bayes** et **SVM** avec une exactitude de **78%**.



# Utilité de Spark

L'utilisation de Spark  a apportée une grande différence en terme de: 

+  **Temps**: Les structures de données offertes par Spark (`RDD`) sont facilement partageables sur des threads dans l'ordinateur (ou le `cluster` si utilisation dans un réseau) ce qui accélère le travail.
+  **Re-utilisabilité**:  Le code utilisé dans un seul ordinateur (mode développement) est le même code utilisé dans le `cluster` (mode production)
+  **Facilité d'utilisation**: Spark offert une `API` riche des structures de données, des algorithmes de Machine Learning et de calcule de performance des modèles, des méthodes puissantes d'exploration des données.