#!/usr/local/bin/python3.5
# -*- coding: utf-8 -*-

# import python libs

from flask import Flask
from flask import jsonify
from flask import request
from pprint import pprint as pp
import argparse
import base64
import json
import logging
import socket

# import project libs

import nltk_nec
import nltk_tree_converter

# defining globals & constants

INTERFACE_TYPE = 'ner_complete'
app = Flask(__name__)

# Flask routes

@app.route('/', methods=['GET'])
@app.route('/iterate', methods=['GET'])
@app.route('/who_are_you', methods=['GET'])
def who_are_you():
    message = {
        'role': 'iterate',
        'title': 'RawData Converter',
        'description': 'Iterate service dummy to convert RawData to renderable AnnotationDocuments.',
        'version': 0.1,
        'problem_id': 'ner',
        'interface_types': [ INTERFACE_TYPE ]
    }
    return create_json_respons_from(message)

@app.route('/iterate', methods=['POST'])
def iterate():
    app.logger.info('post iterate')
    post_json_data = json.dumps(request.json)
    corpus = decode_post_data(post_json_data)
    documents = create_annotation_documents(corpus)

    app.logger.info('created annotation documents from posted JSON data')
    app.logger.info(documents)

    response = { 'annotation_documents': documents }
    return create_json_respons_from(response)

# helpers

def decode_post_data(post_json_data):
    dict_content = json.JSONDecoder().decode(post_json_data)
    for raw_datum in dict_content:
        encoded_content = raw_datum['content']
        deconded_content = base64.b64decode(encoded_content).decode('utf-8')
        raw_datum['content'] = deconded_content

    return dict_content

def create_annotation_documents(corpus):
    annotation_documents = []
    for index, corpus_part in enumerate(corpus):
        raw_datum_id = corpus_part['raw_datum_id']

        content = corpus_part['content']
        parsed_content = json.JSONDecoder().decode(content)
        payload = {
            'content': named_entity_chunking(parsed_content['content'])
        }

        document = {
            'rank': index,
            'raw_datum_id': raw_datum_id,
            'payload': json.dumps(payload),
            'interface_type': INTERFACE_TYPE
        }
        annotation_documents.append(document)

    return annotation_documents

def named_entity_chunking(paragraph):
    # create a list of lists of tuples
    pos_tagged_sentences = [nltk_nec.part_of_speech_tagging(sentence) for sentence in paragraph]

    # create a list of nltk trees containing named entity chunks
    chunk_trees = [nltk_nec.named_entity_token_chunking(sentence) for sentence in pos_tagged_sentences]

    # convert chunk trees back to sentences (list of lists of token objects)
    return [nltk_tree_converter.convert_nltk_tree(tree) for tree in chunk_trees]

def create_json_respons_from(hash):
    response = jsonify(hash)
    response.status_code = 200
    return response

# entry point as a stand alone script

if __name__ == '__main__':
    beVerbose = False
    usePort = 5001
    useHost = 'localhost'
    parser = argparse.ArgumentParser(
        description='Dalphi Iterate Service; 13.10.16 Robert Greinacher')

    parser.add_argument(
        '-p',
        '--port',
        type=int,
        help='set the network port number')
    parser.add_argument(
        '-l',
        '--localhost',
        action='store_true',
        dest='localhost',
        help='use "localhost" instead of current network IP')
    parser.add_argument(
        '-d',
        '--daemon',
        action='store_true',
        dest='daemon',
        help='enables daemon mode')
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        dest='verbose',
        help='enables verbose mode')
    args = parser.parse_args()

    if args.port:
        usePort = args.port
    if not args.localhost:
        hostename = socket.gethostname()
        useHost = socket.gethostbyname(hostename)
    if args.verbose:
        beVerbose = True

    file_handler = logging.FileHandler('service.log')
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('start running flask app')
    app.run(useHost, usePort, beVerbose)
