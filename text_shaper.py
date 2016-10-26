#!/usr/local/bin/python3.5
# -*- coding: utf-8 -*-

# import python libs

import json
from pprint import pprint as pp

# import project libs

import nltk_nec
import nltk_tree_converter

# defining globals & constants
# -

def shape(raw_text):
    # create a list of strings
    sentences = nltk_nec.sentence_splitting(raw_text)

    # create a list of lists of strings
    tokenized_sentences = [nltk_nec.word_tokenization(sentence) for sentence in sentences]

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

def raw_data_json_from(shaped_sentences):
    raw_datum = {
        'id': 'a_file_name.txt',
        'content': shaped_sentences
    }

    return json.dumps(raw_datum)

# entry point as a stand alone script

if __name__ == '__main__':
    name_index = 0
    text = "In the third category he included those Brothers (the majority) who saw nothing in Freemasonry but the external forms and ceremonies, and prized the strict performance of these forms without troubling about their purport or significance. Such were Willarski and even the Grand Master of the principal lodge. Finally, to the fourth category also a great many Brothers belonged, particularly those who had lately joined. These according to Pierre's observations were men who had no belief in anything, nor desire for anything, but joined the Freemasons merely to associate with the wealthy young Brothers who were influential through their connections or rank, and of whom there were very many in the lodge.Pierre began to feel dissatisfied with what he was doing. Freemasonry, at any rate as he saw it here, sometimes seemed to him based merely on externals. He did not think of doubting Freemasonry itself, but suspected that Russian Masonry had taken a wrong path and deviated from its original principles. And so toward the end of the year he went abroad to be initiated into the higher secrets of the order.What is to be done in these circumstances? To favor revolutions, overthrow everything, repel force by force?No! We are very far from that. Every violent reform deserves censure, for it quite fails to remedy evil while men remain what they are, and also because wisdom needs no violence. \"But what is there in running across it like that?\" said Ilagin's groom. \"Once she had missed it and turned it away, any mongrel could take it,\" Ilagin was saying at the same time, breathless from his gallop and his excitement."
    text = 'Donald Trump warned in an interview Tuesday that Hillary Clintons policies as president to address the Syrian conflict would lead to World War III, arguing the Democratic nominee would draw the US into armed confrontation with Russia, Syria and Iran. "What we should do is focus on ISIS. We should not be focusing on Syria," Trump told Reuters on Tuesday morning at his resort in Doral, Florida. "You\'re going to end up in World War III over Syria if we listen to Hillary Clinton." The Republican nominee, who has called for a rapprochement with Russia in order to jointly combat ISIS, argued that his Democratic rival\'s calls for taking a more aggressive posture in Syria to bring the conflict there to an end and combat ISIS will only draw the US into a larger war. Trump\'s remarks come as he trails Clinton in most national and key battleground state polls just two weeks from Election Day.'
    # text = "In fünf Bundesländern haben Spezialeinheiten der Polizei am Dienstag Wohnungen wegen des Verdachts auf Terror-Planungen durchsucht. Einsatzkommandos gingen nach Angaben des Landeskriminalamtes in Erfurt in zwölf Wohnungen und einer Gemeinschaftsunterkunft in Thüringen, Hamburg, Nordrhein-Westfalen, Sachsen und Bayern vor. Anlass der Ermittlungen war nach den Angaben der Verdacht der Vorbereitung einer schweren staatsgefährdenden Gewalttat. Eine konkrete Anschlagsgefahr gab es nach den bisherigen Ermittlungen nicht. Zu Festnahmen ist es nach Auskunft der Polizei nicht gekommen. „Es gibt keinen Haftbefehl“, sagte eine Sprecherin des Thüringer Landeskriminalamtes am Dienstagnachmittag in Erfurt. Hauptverdächtig sei ein 28 Jahre alter Tschetschene mit russischem Pass, der als Asylbewerber im thüringischen Suhl leben soll."
    # text = 'I am very excited about the next generation of Apple products.'
    # text = 'I went to New York to meet John Smith'
    # text = 'Ich bin nach Berlin gefahren um Arik Grahl zu sehen.'

    sentences = shape(text)
    json = raw_data_json_from(sentences)
    pp(json)
