#!/usr/local/bin/python3.5
# -*- coding: utf-8 -*-

# import python libs

from pprint import pprint as pp
import base64
import json
import datetime
import os

# import project libs
# -

# defining globals & constants

SAVE_DOCUMENTS_TO_FILE = False
DOCUMENT_FOLDER = 'processed_annotation_documents'

# methods

def decode_post_data(request_json):
    post_json_data = json.dumps(request_json)
    dict_content = json.JSONDecoder().decode(post_json_data)
    annotation_documents = dict_content['annotation_documents']
    raw_datum_id = dict_content['corpus_document']['raw_datum_id']
    return (raw_datum_id, annotation_documents)

def create_new_raw_datum(raw_datum_id, annotation_documents):
    content = []
    for document in annotation_documents:
        if SAVE_DOCUMENTS_TO_FILE:
            save_document_to_file(document)

        if document['raw_datum_id'] == raw_datum_id:
            json_encoded_payload = document['payload']

            # the `content` of an annotation document holds only one paragraph
            paragraph = json_encoded_payload['content'][0]
            content.append(paragraph)

    raw_datum = {
        'content': content,
        'raw_datum_id': raw_datum_id
    }

    byte_encoded_content = json.dumps(raw_datum).encode('utf-8')
    b64_encoded_content = base64.b64encode(byte_encoded_content)
    string_content = str(b64_encoded_content, encoding='utf-8')

    return {
        'content': string_content,
        'raw_datum_id': raw_datum_id
    }

def save_document_to_file(document):
    json_encoded_document = json.dumps(document)
    file_name = generate_filename_for(document)

    file_handler = open(file_name, 'w')
    file_handler.write(json_encoded_document)
    file_handler.close()

def generate_filename_for(document):
    prefix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    raw_datum_id = document['raw_datum_id']
    rank = document['rank']

    if DOCUMENT_FOLDER:
        if not os.path.exists(DOCUMENT_FOLDER):
            os.makedirs(DOCUMENT_FOLDER)
        return "%s/%s_%s_%s.json" % (DOCUMENT_FOLDER, prefix, raw_datum_id, rank)
    else:
        return "%s_%s_%s.json" % (prefix, raw_datum_id, rank)
