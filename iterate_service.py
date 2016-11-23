#!/usr/local/bin/python3.5
# -*- coding: utf-8 -*-

# import python libs

from flask import Flask
from flask import jsonify
from flask import request
from pprint import pprint as pp
import argparse
import json
import logging
import socket

# import project libs

import ner_pipeline
import iteration_processing
import merge_processing

# defining globals & constants

global app
app = Flask(__name__)

# Flask routes

@app.route('/', methods=['GET'])
def who_are_you():
    logging.info('who are you request; respond with JSON')
    message = {
        'services': [
            {
                'role': 'merge',
                'route': '/merge'
            },
            {
                'role': 'iterate',
                'route': '/iterate'
            }
        ]
    }

    return create_json_respons_from(message)

@app.route('/iterate', methods=['GET'])
def iterate_who_are_you():
    message = {
        'role': 'iterate',
        'title': 'MaxEnt NER Iterator',
        'description': 'Using NLTK\'s MaxEnt chunker for NER. Currently only for english language.',
        'version': 0.2,
        'problem_id': 'ner',
        'interface_types': [ 'ner_complete', 'ner_paragraph' ]
    }
    return create_json_respons_from(message)

@app.route('/iterate', methods=['POST'])
def iterate():
    logging.info('iterate request')
    corpus = iteration_processing.decode_post_data(request.json)
    documents = iteration_processing.iterate_corpus(corpus)

    logging.info('transmitted corpus contains %s documents; created %s annotation documents' % (len(corpus), len(documents)))

    response = { 'annotation_documents': documents }
    return create_json_respons_from(response)

@app.route('/merge', methods=['GET'])
def merge_who_are_you():
    message = {
        'role': 'merge',
        'title': 'RawDatum replacer',
        'description': 'Creates new raw data from annotation documents. ' \
                       'Existing Raw datum will be deleted.',
        'version': 0.1,
        'problem_id': 'ner'
    }
    return create_json_respons_from(message)

@app.route('/merge', methods=['POST'])
def merge():
    logging.info('merge request')
    (raw_datum_id, annotation_documents) = merge_processing.decode_post_data(request.json)
    logging.info('received %s documents as parts of raw datum #%s' % (len(annotation_documents), raw_datum_id))

    raw_datum = merge_processing.create_new_raw_datum(raw_datum_id, annotation_documents)
    return create_json_respons_from(raw_datum)

# helpers

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

    logging.basicConfig(filename='service.log', level=logging.INFO)
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('start running flask app')
    app.run(useHost, usePort, beVerbose)
