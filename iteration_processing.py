#!/usr/local/bin/python3.5
# -*- coding: utf-8 -*-

# import python libs

from pprint import pprint as pp
import base64
import json

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
    annotation_documents = []

    for index, corpus_document in enumerate(corpus):
        raw_datum_id = corpus_document['raw_datum_id']
        document_content = corpus_document['content']

        for paragraph in document_content['content']:
            plain_tokenized_sentences = deannotize(paragraph)
            chunked_sentences = named_entity_chunking(plain_tokenized_sentences)
            target_content = [ chunked_sentences ]
            add_annotation_document(annotation_documents, raw_datum_id, target_content)

    return annotation_documents

def add_annotation_document(document_list, raw_id, document_content):
    payload = { 'content': document_content }
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

def named_entity_chunking(paragraph):
    # create a list of lists of tuples
    pos_tagged_sentences = [ner_pipeline.part_of_speech_tagging(sentence) for sentence in paragraph]

    # create a list of nltk trees containing named entity chunks
    chunk_trees = [ner_pipeline.named_entity_token_chunking(sentence) for sentence in pos_tagged_sentences]

    # convert chunk trees back to sentences (list of lists of token objects)
    return [nltk_tree_converter.convert_nltk_tree(tree) for tree in chunk_trees]
