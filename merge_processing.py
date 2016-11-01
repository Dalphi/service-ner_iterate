#!/usr/local/bin/python3.5
# -*- coding: utf-8 -*-

# import python libs

from pprint import pprint as pp
import base64
import json

# import project libs
# -

# defining globals & constants

SAFE_DOCUMENTS_TO_FILE = False

# methods

def decode_post_data(request_json):
    post_json_data = json.dumps(request_json)
    dict_content = json.JSONDecoder().decode(post_json_data)
    annotation_documents = dict_content['annotation_documents']
    raw_datum_id = dict_content['corpus_document']['raw_datum_id']
    return (raw_datum_id, annotation_documents)

def create_new_raw_datum(raw_datum_id, annotation_documents):
    content = []
    print('create_new_raw_datum')
    for document in annotation_documents:
        if document['raw_datum_id'] == raw_datum_id:
            json_encoded_payload = document['payload']
            payload = json.JSONDecoder().decode(json_encoded_payload)
            payload_content = payload['content']
            pp(payload_content)

            # the `content` of an annotation document holds only one paragraph
            paragraph = payload['content'][0]
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
