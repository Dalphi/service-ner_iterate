#!/usr/local/bin/python3.5
# -*- coding: utf-8 -*-

# import python libs

import nltk
import pickle
import json
from nltk import Tree
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget
from pprint import pprint as pp
from nltk.chunk.api import ChunkParserI
from ClassifierBasedGermanTagger.ClassifierBasedGermanTagger import ClassifierBasedGermanTagger

# import project libs

import nltk_tree_converter
import maxent_chunker

# defining globals & constants

NLTK_CHUNKER_PATH = 'chunkers/maxent_ne_chunker/PY3/english_ace_multiclass.pickle'
POS_TAGGER_PATH = 'bin/nltk_german_pos_classifier_data.pickle'
SAVE_CHUNKER_AS_PICKLE = False
CHUNKER_PICKLE_NAME = 'implisense_ne_chunker_multiclass.pickle'

self_trained_chunker = False
german_pos_tagger = False

# methods

def ner_pipeline(raw_text):
    print('NER pipeline: splitting / tokenizing / tagging / chunking ...')

    # create a list of strings
    sentences = sentence_splitting(raw_text)

    # create a list of lists of strings
    tokenized_sentences = [word_tokenization(sentence) for sentence in sentences]

    # create a list of lists of tuples
    pos_tagged_sentences = [part_of_speech_tagging(sentence) for sentence in tokenized_sentences]

    # create a list of nltk trees containing named entity chunks
    chunk_trees = [named_entity_token_chunking(sentence) for sentence in pos_tagged_sentences]

    return chunk_trees

def sentence_splitting(raw_text):
    return nltk.sent_tokenize(raw_text)

def word_tokenization(sentence):
    return nltk.word_tokenize(sentence)

def part_of_speech_tagging(sentence):
    global german_pos_tagger

    if not german_pos_tagger:
        with open(POS_TAGGER_PATH, 'rb') as f:
            german_pos_tagger = pickle.load(f)

    return german_pos_tagger.tag(sentence)

def named_entity_token_chunking(tagged_sentence):
    global self_trained_chunker

    if self_trained_chunker:
        return self_trained_chunker.parse(tagged_sentence)
    else:
        print('using a pre trained chunker for (default NLTK NEChunker)')
        # Loads the serialized NLTK NEChunkParser object
        chunker = nltk.data.load(NLTK_CHUNKER_PATH)
        return chunker.parse(tagged_sentence)

def train_maxent_chunker(tokenized_sentences):
    training_data_tree = nltk_tree_converter.sentences_to_tree(tokenized_sentences)
    pos_tagged_tree = maxent_chunker.postag_tree(training_data_tree)
    training_data = [pos_tagged_tree]
    return build_maxent_model(training_data)

def build_maxent_model(training_data):
    global self_trained_chunker
    self_trained_chunker = maxent_chunker.NEChunkParser(training_data)

    if SAVE_CHUNKER_AS_PICKLE:
        print('Saving chunker to %s...' % CHUNKER_PICKLE_NAME)

        with open(CHUNKER_PICKLE_NAME, 'wb') as outfile:
            pickle.dump(chunker, outfile, -1)

    return self_trained_chunker

# helper

def load_annotated_raw_datum(file_name):
    file_handler = open(file_name, 'r')
    json_encoded_document = file_handler.read()
    file_handler.close()
    corpus_document = json.JSONDecoder().decode(json_encoded_document)
    return corpus_document['data']

def explain_tagging(tagged_sentence):
    # Loads the serialized NEChunkParser object
    chunker = nltk.data.load(NLTK_CHUNKER_PATH)

    # The MaxEnt classifier
    maxEnt = chunker._tagger.classifier()

    tags = []
    for i in range(0, len(tagged_sentence)):
        tag = chunker._tagger.choose_tag(tagged_sentence, i, tags)
        if tag != 'O':
            print('\nExplanation on the why the word \'' + tagged_sentence[i][0] + '\' was tagged:')
            featureset = chunker._tagger.feature_detector(tagged_sentence, i, tags)
            maxEnt.explain(featureset)
        tags.append(tag)

    return tags

def draw_to_file(tree):
    canvas = CanvasFrame()
    tree_canvas = TreeWidget(canvas.canvas(), tree)
    canvas.add_widget(tree_canvas,10,10)

    file_name = 'tree_plot.ps'

    canvas.print_to_file(file_name)
    canvas.destroy()

    # clean up using the following shell comand:
    # for i in tree_plot_*.ps; do convert $i "$i.png"; rm $i; done

# entry point as a stand alone script
if __name__ == '__main__':
    # global POS_TAGGER_PATH
    POS_TAGGER_PATH = '../' + POS_TAGGER_PATH

    text = "In Nordrhein-Westfalen betreibt die Tengelmann-Gruppe derzeit noch 105 Kaiser's-Tengelmann-Filialen mit rund 4000 Mitarbeitern."

    chunk_trees = ner_pipeline(text)
    sentences = [nltk_tree_converter.tree_to_sentence(tree) for tree in chunk_trees]
    pp(sentences)
