# this code follows the blog post from
# https://datascience.blog.wzb.eu/2016/07/13/accurate-part-of-speech-tagging-of-german-texts-with-nltk/
# visited 24-01-2017

import nltk
import pickle
import random
from ClassifierBasedGermanTagger.ClassifierBasedGermanTagger import ClassifierBasedGermanTagger

print('load the tiger corpus...')
corp = nltk.corpus.ConllCorpusReader('.', 'tiger_release_aug07.corrected.16012013.conll09',
                                     ['ignore', 'words', 'ignore', 'ignore', 'pos'],
                                     encoding='utf-8')
tagged_sents = corp.tagged_sents()
# random.shuffle(tagged_sents)

# set a split size: use 90% for training, 10% for testing
split_perc = 0.1
split_size = int(len(tagged_sents) * split_perc)
train_sents, test_sents = tagged_sents[split_size:], tagged_sents[:split_size]

print('train the classifier...')
tagger = ClassifierBasedGermanTagger(train=train_sents)

print('evaluate the classifier...')
accuracy = tagger.evaluate(test_sents)
print('accuracy of the trained german POS tagger:', accuracy)

# save tagger to pickle
with open('nltk_german_pos_classifier_data.pickle', 'wb') as f:
    pickle.dump(tagger, f, protocol=2)
