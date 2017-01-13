#!/usr/bin/env python3
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
import requests
import threading

# import project libs

import ner_pipeline
import iteration_processing
import merge_processing

# defining globals & constants

global app
global args
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

    return create_json_response_from(message)

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
    return create_json_response_from(message)

@app.route('/iterate', methods=['POST'])
def iterate():
    if args.async:
        logging.info('iterate request (async)')
        data = request.json
        threading.Thread(target=async_iteration_processing, args=(data,)).start()
        return create_json_response_from({ 'status': 'async' })

    else:
        logging.info('iterate request (request)')
        (documents, statistics) = iteration_processing.process_iteration(request.json['raw_data'])
        return create_json_response_from({
            'annotation_documents': documents,
            'statistics': statistics
        })

def async_iteration_processing(data):
    (documents, statistics) = iteration_processing.process_iteration(data['raw_data'])
    annotation_documents = { 'annotation_documents': documents }
    res = requests.post(
        data['callback_urls'][0],
        data=json.dumps(annotation_documents),
        headers={ 'Content-Type': 'application/json' }
    )

    statistics = { 'statistics': statistics }
    res = requests.post(
        data['callback_urls'][1],
        data=json.dumps(statistics),
        headers={ 'Content-Type': 'application/json' }
    )

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
    return create_json_response_from(message)

@app.route('/merge', methods=['POST'])
def merge():
    if args.async:
        logging.info('merge request (async)')
        data = request.json
        threading.Thread(target=async_merge_processing, args=(data,)).start()
        return create_json_response_from({ 'status': 'async' })

    else:
        logging.info('merge request (request)')
        (raw_datum_id, annotation_documents) = merge_processing.decode_post_data(request.json)
        logging.info('received %s documents as parts of raw datum #%s' % (len(annotation_documents), raw_datum_id))

        raw_datum = merge_processing.create_new_raw_datum(raw_datum_id, annotation_documents)
        return create_json_response_from(raw_datum)

def async_merge_processing(data):
    (raw_datum_id, annotation_documents) = merge_processing.decode_post_data(data)
    logging.info('received %s documents as parts of raw datum #%s' % (len(annotation_documents), raw_datum_id))

    raw_datum = merge_processing.create_new_raw_datum(raw_datum_id, annotation_documents)
    res = requests.patch(
        data['callback_url'],
        data=json.dumps(raw_datum),
        headers={ 'Content-Type': 'application/json' }
    )

# helpers

def create_json_response_from(hash):
    response = jsonify(hash)
    response.status_code = 200
    return response

# entry point as a stand alone script

if __name__ == '__main__':
    usePort = 5001
    useHost = 'localhost'
    parser = argparse.ArgumentParser(
        description='Dalphi Iterate Service; 13.10.16 Robert Greinacher')
    parser.add_argument(
        '-a',
        '--async',
        action='store_true',
        dest='async',
        help='communicates asynchronous and non-blocking with DALPHI')
    parser.add_argument(
        '-d',
        '--daemon',
        action='store_true',
        dest='daemon',
        help='enables daemon mode')
    parser.add_argument(
        '-l',
        '--localhost',
        action='store_true',
        dest='localhost',
        help='use "localhost" instead of current network IP')
    parser.add_argument(
        '-p',
        '--port',
        type=int,
        help='set the network port number')
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

    logging.basicConfig(filename='service.log', level=logging.INFO)
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('start running flask app')
    app.run(useHost, usePort, args.verbose)
