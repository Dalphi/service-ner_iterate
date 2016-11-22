#!/usr/local/bin/python3.5
# -*- coding: utf-8 -*-

# import python libs

from nltk import Tree
from pprint import pprint as pp

# import project libs
# -

# defining globals & constants
# -

# methods

def corpus_to_tree(corpus):
    tree_with_entities = Tree('S', [])
    skip_count = 0
    for paragraph in corpus:
        for sentence in paragraph:
            for index, token in enumerate(sentence):
                if skip_count > 0:
                    skip_count = skip_count - 1
                    continue

                if 'annotation' in token:
                    annotation = token['annotation']
                    length = annotation['length']
                    sub_tree = Tree(annotation['label'], [token['term']])

                    if length > 1:
                        skip_count = length - 1
                        for next_index in range((index + 1), (index + length)):
                            word = sentence[next_index]['term']
                            sub_tree.append(word)

                    tree_with_entities.append(sub_tree)

                else:
                    tree_with_entities.extend([token['term']])

    return tree_with_entities

def tree_to_sentence(tree):
    sentence = []

    def add_token_from(node):
        token = {
            'term': node[0]
        }
        sentence.append(token)

    def add_tokens_from(tree):
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
        return node.label()

        # if node.label() == 'PERSON':
        #     label = 'PER'
        # else:
        #     label = 'COM'
        #
        # return label

    for node in tree:
        if type(node) is Tree:
            add_tokens_from(node)
        else:
            add_token_from(node)

    return sentence
