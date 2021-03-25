import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
import re

import pickle as pkl

# LOAD badworlds list
file = open('data/bad_words.txt', 'r')
file = list(file)
bad_words = []
for w in file:
    bad_words.append(re.sub(r'\n', '', w))

song = open('data/song.txt', 'r')


class CustomFeats(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feat_names = set()

    def fit(self, x, y=None):
        return self

    @staticmethod
    def features(review):
        return {
            'num_word': get_num_words(review),
            'bad_word': get_bad_words(review)
        }

    def get_feature_names(self):
        return list(self.feat_names)

    def transform(self, reviews):
        feats = []
        for review in reviews:
            f = self.features(review)
            [self.feat_names.add(k) for k in f]
            feats.append(f)
        return feats


def get_bad_words(review):
  target_word = bad_words
  count = 0
  threshold = 0
  for t in target_word:
        if review.find(t) != -1:
            count += 1
  return count > threshold


def get_num_words(review):
  threshold = 0
  words = review.split(' ')
  count = len(list(words))
  return count > threshold


# LOAD MODELS
with open('models/feats.pkl', 'rb') as file:
    feats = pkl.load(file)

with open('models/random_forest.pkl', 'rb') as file:
    random_fr = pkl.load(file)

with open('models/svm.pkl', 'rb') as file:
    svm = pkl.load(file)


# result on spotify .csv file
def get_results():
    spotify_lyrics = pd.read_csv('data/billboard-lyrics-spotify.csv')
    spotify_lyrics['explicit'] = spotify_lyrics['explicit'].apply(lambda x: 1 if x == 1.0 else 0)
    lyrics = list(spotify_lyrics['lyrics'].apply(lambda x: str(x)))
    labels = spotify_lyrics['explicit']
    test = feats.transform(lyrics)
    results_1 = random_fr.predict(test)
    results_2 = svm.predict(test)
    print(classification_report(labels, results_1))
    print(classification_report(labels, results_2))


def get_song_res(text):
    test = feats.transform(list(text))
    results_1 = random_fr.predict(test)
    results_2 = svm.predict(test)
    print(results_1)
    print(results_2)


if __name__ == '__main__':
    # get_results()
    get_song_res(song)
