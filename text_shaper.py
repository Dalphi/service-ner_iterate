#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import python libs

import re
import json
import argparse
from os import listdir
from os.path import isfile, join
from pprint import pprint as pp

# import project libs

import sys
sys.path.append('lib')
import ner_pipeline
import nltk_tree_converter

# defining globals & constants
# -

def shape(raw_text):
    # simplify quotes
    raw_text = re.sub("``", ' "', raw_text)
    raw_text = re.sub("''", '" ', raw_text)

    # create a list of strings
    sentences = ner_pipeline.sentence_splitting(raw_text)

    # create a list of lists of strings
    tokenized_sentences = [ner_pipeline.word_tokenization(sentence) for sentence in sentences]

    sentences = []
    for sentence in tokenized_sentences:
        tokens = []
        for word in sentence:
            token = {
                'term': word
            }
            tokens.append(token)
        sentences.append(tokens)

    return sentences

def raw_data_json_from(shaped_sentences, input_file_name):
    raw_datum = {
        'id': input_file_name,
        'data': shaped_sentences
    }

    return json.dumps(raw_datum)

def read_input_file(file_handler):
    global input_file_name

    input_file_name = file_handler.name
    content = file_handler.read()
    file_handler.close()

    # return a list of paragraphs
    return content.split("\n\n")

def save_to_file(json, file_handler):
    file_handler.write(json)
    file_handler.close()

def iterate_plain_paragraphs(paragraphs):
    shaped_paragraphs = []

    for paragraph in paragraphs:
        sentences = shape(paragraph)
        shaped_paragraphs.append(sentences)

    return shaped_paragraphs

# entry point as a stand alone script

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Dalphi Iterate Service text shaper; converts plain text to raw data')

    parser.add_argument(
        '-i',
        '--input',
        type=argparse.FileType('r')
    )
    parser.add_argument(
        '-o',
        '--output',
        type=argparse.FileType('w')
    )
    parser.add_argument(
        "-id",
        "--input_dir",
        help="to shape all files in the input directory")
    parser.add_argument(
        "-od",
        "--output_dir",
        help="to shape all files in the input directory")
    args = parser.parse_args()

    if args.input_dir and args.output_dir:
        path = args.input_dir
        for file_name in listdir(path):
            if not (isfile(join(path, file_name)) and file_name.endswith('.txt')): continue
            file_handler = open(path + file_name, 'r', encoding='utf-8')
            paragraphs = read_input_file(file_handler)
            shaped_paragraphs = iterate_plain_paragraphs(paragraphs)
            json_object = raw_data_json_from(shaped_paragraphs, file_name)

            file_handler = open(args.output_dir + file_name + '.json', 'w', encoding='utf-8')
            save_to_file(json_object, file_handler)

    elif args.input and args.output:
        paragraphs = read_input_file(args.input)
        shaped_paragraphs = iterate_plain_paragraphs(paragraphs)
        json_object = raw_data_json_from(shaped_paragraphs, args.input.name)
        save_to_file(json_object, args.output)

    else:
        print('specify input and output (help: -h)')
