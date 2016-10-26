#!/usr/local/bin/python3.5
# -*- coding: utf-8 -*-

# import python libs

import nltk
from nltk import Tree
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget
from pprint import pprint as pp

# import project libs

import nltk_tree_converter

# defining globals & constants
# -

def ner_pipeline(raw_text):
    print('NER pipeline: splitting / tokenizing / tagging / chunking ...')

    # create a list of strings
    sentences = sentence_splitting(raw_text)

    # create a list of lists of strings
    tokenized_sentences = [word_tokenization(sentence) for sentence in sentences]

    # create a list of lists of tuples
    pos_tagged_sentences = [part_of_speech_tagging(sentence) for sentence in tokenized_sentences]

    # create a list of nltk trees containing named entity chunks
    chunk_trees = [named_entity_token_chunking(sentence) for sentence in pos_tagged_sentences]

    return chunk_trees

def sentence_splitting(raw_text):
    return nltk.sent_tokenize(raw_text)

def word_tokenization(sentence):
    return nltk.word_tokenize(sentence)

def part_of_speech_tagging(sentence):
    return nltk.pos_tag(sentence)

def named_entity_token_chunking(tagged_sentence):
    # Loads the serialized NEChunkParser object
    chunker = nltk.data.load('chunkers/maxent_ne_chunker/PY3/english_ace_multiclass.pickle')
    return chunker.parse(tagged_sentence)

# helper

def explain_tagging(tagged_sentence):
    # Loads the serialized NEChunkParser object
    chunker = nltk.data.load('chunkers/maxent_ne_chunker/PY3/english_ace_multiclass.pickle')

    # The MaxEnt classifier
    maxEnt = chunker._tagger.classifier()

    tags = []
    for i in range(0, len(tagged_sentence)):
        tag = chunker._tagger.choose_tag(tagged_sentence, i, tags)
        if tag != 'O':
            print('\nExplanation on the why the word \'' + tagged_sentence[i][0] + '\' was tagged:')
            featureset = chunker._tagger.feature_detector(tagged_sentence, i, tags)
            maxEnt.explain(featureset)
        tags.append(tag)

    return tags

def draw_to_file(tree):
    global name_index

    canvas = CanvasFrame()
    tree_canvas = TreeWidget(canvas.canvas(), tree)
    canvas.add_widget(tree_canvas,10,10)

    file_name = 'tree_plot.ps'

    canvas.print_to_file(file_name)
    canvas.destroy()

    # clean up using the following shell comand:
    # for i in tree_plot_*.ps; do convert $i "$i.png"; rm $i; done

# entry point as a stand alone script

if __name__ == '__main__':
    name_index = 0
    text = "In the third category he included those Brothers (the majority) who saw nothing in Freemasonry but the external forms and ceremonies, and prized the strict performance of these forms without troubling about their purport or significance. Such were Willarski and even the Grand Master of the principal lodge. Finally, to the fourth category also a great many Brothers belonged, particularly those who had lately joined. These according to Pierre's observations were men who had no belief in anything, nor desire for anything, but joined the Freemasons merely to associate with the wealthy young Brothers who were influential through their connections or rank, and of whom there were very many in the lodge.Pierre began to feel dissatisfied with what he was doing. Freemasonry, at any rate as he saw it here, sometimes seemed to him based merely on externals. He did not think of doubting Freemasonry itself, but suspected that Russian Masonry had taken a wrong path and deviated from its original principles. And so toward the end of the year he went abroad to be initiated into the higher secrets of the order.What is to be done in these circumstances? To favor revolutions, overthrow everything, repel force by force?No! We are very far from that. Every violent reform deserves censure, for it quite fails to remedy evil while men remain what they are, and also because wisdom needs no violence. \"But what is there in running across it like that?\" said Ilagin's groom. \"Once she had missed it and turned it away, any mongrel could take it,\" Ilagin was saying at the same time, breathless from his gallop and his excitement."
    text = 'Donald Trump warned in an interview Tuesday that Hillary Clintons policies as president to address the Syrian conflict would lead to World War III, arguing the Democratic nominee would draw the US into armed confrontation with Russia, Syria and Iran. "What we should do is focus on ISIS. We should not be focusing on Syria," Trump told Reuters on Tuesday morning at his resort in Doral, Florida. "You\'re going to end up in World War III over Syria if we listen to Hillary Clinton." The Republican nominee, who has called for a rapprochement with Russia in order to jointly combat ISIS, argued that his Democratic rival\'s calls for taking a more aggressive posture in Syria to bring the conflict there to an end and combat ISIS will only draw the US into a larger war. Trump\'s remarks come as he trails Clinton in most national and key battleground state polls just two weeks from Election Day.'
    # text = "In fünf Bundesländern haben Spezialeinheiten der Polizei am Dienstag Wohnungen wegen des Verdachts auf Terror-Planungen durchsucht. Einsatzkommandos gingen nach Angaben des Landeskriminalamtes in Erfurt in zwölf Wohnungen und einer Gemeinschaftsunterkunft in Thüringen, Hamburg, Nordrhein-Westfalen, Sachsen und Bayern vor. Anlass der Ermittlungen war nach den Angaben der Verdacht der Vorbereitung einer schweren staatsgefährdenden Gewalttat. Eine konkrete Anschlagsgefahr gab es nach den bisherigen Ermittlungen nicht. Zu Festnahmen ist es nach Auskunft der Polizei nicht gekommen. „Es gibt keinen Haftbefehl“, sagte eine Sprecherin des Thüringer Landeskriminalamtes am Dienstagnachmittag in Erfurt. Hauptverdächtig sei ein 28 Jahre alter Tschetschene mit russischem Pass, der als Asylbewerber im thüringischen Suhl leben soll."
    # text = 'I am very excited about the next generation of Apple products.'
    # text = 'I went to New York to meet John Smith'
    # text = 'Ich bin nach Berlin gefahren um Arik Grahl zu sehen.'

    chunk_trees = ner_pipeline(text)
    sentences = [nltk_tree_converter.convert_nltk_tree(tree) for tree in chunk_trees]
    pp(sentences)
