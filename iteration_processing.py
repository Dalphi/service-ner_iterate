#!/usr/local/bin/python3.5
# -*- coding: utf-8 -*-

# import python libs

from pprint import pprint as pp
import base64
import json
import logging

# import project libs

import ner_pipeline
import nltk_tree_converter

# defining globals & constants
# -

# methods

def decode_post_data(request_json):
    post_json_data = json.dumps(request_json)
    dict_content = json.JSONDecoder().decode(post_json_data)
    for raw_datum in dict_content:
        encoded_content = raw_datum['content']
        deconded_content = base64.b64decode(encoded_content).decode('utf-8')
        deconded_content = json.JSONDecoder().decode(deconded_content)
        raw_datum['content'] = deconded_content

    return dict_content

def iterate_corpus(corpus):
    training(corpus)

    annotation_documents = []
    for corpus_document in corpus:
        raw_datum_id = corpus_document['raw_datum_id']
        document_content = corpus_document['content']

        for paragraph in document_content['content']:
            plain_tokenized_sentences = deannotize(paragraph)
            ne_chunked_paragraph = ne_chunking(plain_tokenized_sentences)
            annotated_paragraph = prefere_human_annotations(paragraph, ne_chunked_paragraph)
            add_annotation_document(annotation_documents, raw_datum_id, annotated_paragraph)

    return annotation_documents

def training(corpus):
    # train a maxent classifier for chunking named entities with this current corpus
    decapsulated_corpus = decapsulate(corpus)
    annotated_senteces = extract_annotated_senteces(decapsulated_corpus)
    if annotated_senteces:
        ner_pipeline.train_maxent_chunker(annotated_senteces)
    else:
        logging.warning('No annotations found. Skip model training.')

def decapsulate(corpus):
    listified_corpus = [listify(element) for element in corpus]
    number_of_corpus_documents = len(listified_corpus)
    if number_of_corpus_documents == 1:
        return listified_corpus[0]
    else:
        list_of_paragraphs = []
        for document_index in range (0, number_of_corpus_documents):
            for paragraph in listified_corpus[document_index]:
                list_of_paragraphs.append(paragraph)
        return list_of_paragraphs

def listify(obj):
    if isinstance(obj, dict):
        inner_object = obj['content']
        return listify(inner_object)
    else:
        return obj

def extract_annotated_senteces(corpus):
    annotated_sentences = []
    for paragraph in corpus:
        for sentence in paragraph:
            if sentence_is_annotated(sentence):
                annotated_sentences.append(sentence)

    return annotated_sentences

def sentence_is_annotated(sentence):
    for token in sentence:
        if 'annotation' in token:
            return True
    return False

def prefere_human_annotations(human_checked_paragraph, machine_labeled_paragraph):
    return machine_labeled_paragraph

def add_annotation_document(document_list, raw_id, document_content):
    content = [document_content]
    payload = {'content': content}
    encoded_payload = json.dumps(payload)

    document_list.append({
        'rank': len(document_list),
        'raw_datum_id': raw_id,
        'payload': encoded_payload,
        'interface_type': 'ner_complete'
    })

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
