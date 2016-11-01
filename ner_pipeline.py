#!/usr/local/bin/python3.5
# -*- coding: utf-8 -*-

# import python libs

import nltk
from nltk import Tree
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget
from pprint import pprint as pp

# import project libs

import nltk_tree_converter

# defining globals & constants
# -

def ner_pipeline(raw_text):
    print('NER pipeline: splitting / tokenizing / tagging / chunking ...')

    # create a list of strings
    sentences = sentence_splitting(raw_text)

    # create a list of lists of strings
    tokenized_sentences = [word_tokenization(sentence) for sentence in sentences]

    # create a list of lists of tuples
    pos_tagged_sentences = [part_of_speech_tagging(sentence) for sentence in tokenized_sentences]
    pp(pos_tagged_sentences[0])
    exit()

    explain_tagging(pos_tagged_sentences[0])

    # create a list of nltk trees containing named entity chunks
    chunk_trees = [named_entity_token_chunking(sentence) for sentence in pos_tagged_sentences]

    return chunk_trees

def sentence_splitting(raw_text):
    return nltk.sent_tokenize(raw_text)

def word_tokenization(sentence):
    return nltk.word_tokenize(sentence)

def part_of_speech_tagging(sentence):
    return nltk.pos_tag(sentence)

def named_entity_token_chunking(tagged_sentence):
    # Loads the serialized NEChunkParser object
    chunker = nltk.data.load('chunkers/maxent_ne_chunker/PY3/english_ace_multiclass.pickle')
    return chunker.parse(tagged_sentence)

# helper

def explain_tagging(tagged_sentence):
    # Loads the serialized NEChunkParser object
    chunker = nltk.data.load('chunkers/maxent_ne_chunker/PY3/english_ace_multiclass.pickle')

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
    global name_index

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
    name_index = 0
    text = "The man blamed for bringing HIV to the United States just had his name cleared. New research has proved that Gaëtan Dugas, a French-Canadian flight attendant who was dubbed \"patient zero,\" did not spread HIV, the virus that causes AIDS, to the United States."

    chunk_trees = ner_pipeline(text)
    sentences = [nltk_tree_converter.convert_nltk_tree(tree) for tree in chunk_trees]
    pp(sentences)
