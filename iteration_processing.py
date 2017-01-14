#!/usr/local/bin/python3.5
# -*- coding: utf-8 -*-

# import python libs

from pprint import pprint as pp
import base64
import json
import logging
import random

# import project libs

import ner_pipeline
import nltk_tree_converter

# defining globals & constants

NEW_DOCUMENTS_LIMIT = 0
PASS_THROUGH_ONLY = True # don't do NER but use the already present labels

# methods

def process_iteration(raw_data):
    corpus = decode_post_data(raw_data)
    documents = iterate_corpus(corpus)
    statistics = iterate_statistics(documents)
    return (documents, statistics)

def decode_post_data(request_json):
    for raw_datum in request_json:
        encoded_content = raw_datum['data']
        deconded_content = base64.b64decode(encoded_content).decode('utf-8')
        deconded_content = json.JSONDecoder().decode(deconded_content)
        raw_datum['data'] = deconded_content
    return request_json

def iterate_corpus(corpus):
    training(corpus)

    annotation_documents = []
    if PASS_THROUGH_ONLY:
        logging.warning('pass through only mode; not doing a NER and just passing annotations as they are')

    for corpus_document in corpus:
        raw_datum_id = corpus_document['id']
        document_content = corpus_document['data']

        # backward compatibility with former notation
        if 'data' in document_content:
            paragraphs = document_content['data']
        else:
            paragraphs = document_content['content']

        for paragraph in paragraphs:
            human_checked = paragraph_was_human_checked(paragraph)

            if PASS_THROUGH_ONLY:
                ne_chunked_paragraph = paragraph
            else:
                plain_tokenized_sentences = deannotize(paragraph)
                ne_chunked_paragraph = ne_chunking(plain_tokenized_sentences)

            annotated_paragraph = prefere_human_annotations(paragraph, ne_chunked_paragraph)
            add_annotation_document(
                annotation_documents,
                raw_datum_id,
                annotated_paragraph,
                human_checked
            )

            if limit_criterium(annotation_documents): break
        if limit_criterium(annotation_documents): break

    return annotation_documents

# returns just some dummy statistics in order to test interoperability
def iterate_statistics(documents):
    raw_data_ids = list(set(map(lambda document: document['raw_datum_id'], documents)))
    return [
        {
            'key': 'test',
            'value': random.uniform(0, 1),
            'raw_data_ids': raw_data_ids,
            'iteration_index': random.randint(0, 123465789)
        }
    ]

# train a maxent classifier for chunking named entities with this current corpus
def training(corpus):
    decapsulated_corpus = decapsulate(corpus)
    annotated_senteces = extract_annotated_senteces(decapsulated_corpus)
    if annotated_senteces:
        ner_pipeline.train_maxent_chunker(annotated_senteces)
    else:
        logging.warning('No annotations found. Skip model training.')

# merge all paragraphs of different documents together; return a list of paragraphs
def decapsulate(corpus):
    listified_corpus = [listify(element) for element in corpus]
    list_of_paragraphs = []
    for document_index in range (0, len(listified_corpus)):
        for paragraph in listified_corpus[document_index]:
            list_of_paragraphs.append(paragraph)
    return list_of_paragraphs

# remove everything but plain text structure
def listify(obj):
    if isinstance(obj, dict):
        if 'content' in obj.keys():
            inner_object = obj['content']
        elif 'data' in obj.keys():
            inner_object = obj['data']
        return listify(inner_object)
    else:
        return obj

# return a list of annotated sentences of a document
def extract_annotated_senteces(document):
    annotated_sentences = []
    for paragraph in document:
        for sentence in paragraph:
            if sentence_is_annotated(sentence):
                annotated_sentences.append(sentence)

    return annotated_sentences

def sentence_is_annotated(sentence):
    for token in sentence:
        if 'annotation' in token:
            return True
    return False

def paragraph_was_human_checked(paragraph):
    # Checking if the pre-ML-chunked sentence already contains annotations is a good estimator
    # but paragraphs which don't contain any entities won't be recognized by this metric.
    for sentence in paragraph:
        if sentence_is_annotated(sentence): return True
    return False

# construct a paragraph containing all human made annotations, enriched by all other artificial annotations
def prefere_human_annotations(human_checked_paragraph, machine_labeled_paragraph):
    if len(human_checked_paragraph) == len(machine_labeled_paragraph):
        for sentence_index, sentence in enumerate(human_checked_paragraph):
            for token_index, token in enumerate(sentence):
                if 'annotation' in token:
                    # Overwriting the token in the ML chunked sentence with the human checked token
                    # this doesn't care about annotation lengths - might become an issue.
                    machine_labeled_paragraph[sentence_index][token_index] = token

    else:
        logging.error('prefere_human_annotations: human_checked_paragraph and machine_labeled_paragraph have different shapes!')
        machine_labeled_paragraph = human_checked_paragraph

    return machine_labeled_paragraph

def add_annotation_document(document_list, raw_id, document_content, human_checked):
    content = [document_content]
    payload = {'content': content}
    rank = len(document_list)

    if human_checked:
        rank = rank + 1000

    document_list.append({
        'rank': rank,
        'raw_datum_id': raw_id,
        'payload': payload,
        'interface_type': 'ner_complete'
    })

# remove annotations from a sentence
def deannotize(sentences):
    plain_tokenized_sentences = []
    for sentence in sentences:
        new_sentence = []
        for token in sentence:
            word = token['term']
            new_sentence.append(word)
        plain_tokenized_sentences.append(new_sentence)

    return plain_tokenized_sentences

def ne_chunking(paragraph):
    # create a list of lists of tuples
    pos_tagged_sentences = [ner_pipeline.part_of_speech_tagging(sentence) for sentence in paragraph]

    # create a list of nltk trees containing named entity chunks
    chunk_trees = [ner_pipeline.named_entity_token_chunking(sentence) for sentence in pos_tagged_sentences]

    # convert chunk trees back to sentences (list of lists of token objects)
    return [nltk_tree_converter.tree_to_sentence(tree) for tree in chunk_trees]

def limit_criterium(documents):
    return NEW_DOCUMENTS_LIMIT > 0 and len(documents) == NEW_DOCUMENTS_LIMIT
