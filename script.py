#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --------------- Functions ------------------ #


# ----- initialisation de Spark -------- #
import findspark
findspark.init('/opt/spark')

from pyspark import SparkContext
sc = SparkContext("local", "NLP App")


# -------- Lire les data-set ----------- #
dataset_path = '/home/mehdi/Documents/S2/2020/BDM/TPs/NLP2/scripts/'
stopwords_path = '/home/mehdi/Documents/S2/2020/BDM/TPs/NLP2/scripts/'
data = sc.textFile(dataset_path + "*.txt").map(lambda line: line.split("\t"))
stopwords = sc.textFile(stopwords_path + "english").collect()

documents = data.map(lambda line: line[0])
labels = data.map(lambda line: line[1])

# --------- Pre-traitement ----------- #
def lower_clean_str(x):
  punc='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
  lowercased_str = x.lower()
  for ch in punc:
    lowercased_str = lowercased_str.replace(ch, ' ')
  return lowercased_str.strip()

import re
documents = documents.map(lambda line: re.sub(" +", ' ', lower_clean_str(line)).split(" "))

def removeStopWords(words, stopwords):
    return [x for x in words if x not in stopwords]

documents = documents.map(lambda line: removeStopWords(line, stopwords))

documents.take(2)

# --------- Appliquer TF-IDF --------- #
from pyspark.mllib.feature import HashingTF, IDF
hashingTF = HashingTF()
tf = hashingTF.transform(documents)

tf.cache()
idf = IDF().fit(tf)
tfidf = idf.transform(tf)

idfIgnore = IDF(minDocFreq=5).fit(tf)
tfidfIgnore = idfIgnore.transform(tf)

# --------- Preparer pour le training ---------- #
from pyspark.mllib.regression import LabeledPoint

tfidfWithIndexes = tfidfIgnore.zipWithIndex().map(lambda x: (x[1], x[0]))
labelsWithIndexes = labels.zipWithIndex().map(lambda x: (x[1], x[0]))
labelsWithIndexes.take(5)
trainingData = tfidfWithIndexes.join(labelsWithIndexes).map(lambda x: LabeledPoint(x[1][1], x[1][0]))
training, test = trainingData.randomSplit([0.7, 0.3])

# -------------- La phase training --------------- #
# model 1
from pyspark.mllib.classification import NaiveBayes
model1 = NaiveBayes.train(training, 5.0)
predictionAndLabel_NB = test.map(lambda p: (model1.predict(p.features), p.label))

#model 2
from pyspark.mllib.classification import SVMWithSGD
model2 = SVMWithSGD.train(training, iterations=100)
predictionAndLabel_SVM = test.map(lambda p: (model2.predict(p.features), p.label))

#model 3
from pyspark.mllib.tree import RandomForest
model = RandomForest.trainClassifier(training, numClasses=6, numTrees=2, categoricalFeaturesInfo={}, featureSubsetStrategy="auto", maxDepth=6, maxBins=32)
predictions = model.predict(test.map(lambda x: x.features))
predictionAndLabel_RF = test.map(lambda lp: lp.label).zip(predictions)

# -------------- La phase prediction ------------#
def accuracy(predictionAndLabel):
    return 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()

print('model accuracy {}'.format(accuracy(predictionAndLabel_NB)))
print('model accuracy {}'.format(accuracy(predictionAndLabel_SVM)))
print('model accuracy {}'.format(accuracy(predictionAndLabel_RF)))