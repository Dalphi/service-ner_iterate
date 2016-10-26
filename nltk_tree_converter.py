#!/usr/local/bin/python3.5
# -*- coding: utf-8 -*-

# import python libs

from nltk import Tree
from pprint import pprint as pp

# import project libs
# -

# defining globals & constants

sentence = []

def convert_nltk_tree(tree):
    global sentence

    for node in tree:
        if type(node) is Tree:
            add_tokens_from(node)
        else:
            add_token_from(node)

    return sentence

def add_token_from(node):
    global sentence

    token = {
        'term': node[0]
    }
    sentence.append(token)

def add_tokens_from(tree):
    global sentence

    length = len(tree)
    token = {
        'term': tree[0][0],
        'annotation': {
            'label': annotation_label(tree),
            'length': length
        }
    }

    sentence.append(token)

    if length > 1:
        for node_index in range(1, length):
            add_token_from(tree[node_index])

def annotation_label(node):
    if node.label() == 'PERSON':
        label = 'PER'
    else:
        label = 'COM'

    return label
