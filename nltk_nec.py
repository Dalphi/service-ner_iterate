import nltk
from nltk import Tree
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget
from pprint import pprint as pp

name_index = 0
text_en = "In the third category he included those Brothers (the majority) who saw nothing in Freemasonry but the external forms and ceremonies, and prized the strict performance of these forms without troubling about their purport or significance. Such were Willarski and even the Grand Master of the principal lodge. Finally, to the fourth category also a great many Brothers belonged, particularly those who had lately joined. These according to Pierre's observations were men who had no belief in anything, nor desire for anything, but joined the Freemasons merely to associate with the wealthy young Brothers who were influential through their connections or rank, and of whom there were very many in the lodge.Pierre began to feel dissatisfied with what he was doing. Freemasonry, at any rate as he saw it here, sometimes seemed to him based merely on externals. He did not think of doubting Freemasonry itself, but suspected that Russian Masonry had taken a wrong path and deviated from its original principles. And so toward the end of the year he went abroad to be initiated into the higher secrets of the order.What is to be done in these circumstances? To favor revolutions, overthrow everything, repel force by force?No! We are very far from that. Every violent reform deserves censure, for it quite fails to remedy evil while men remain what they are, and also because wisdom needs no violence. \"But what is there in running across it like that?\" said Ilagin's groom. \"Once she had missed it and turned it away, any mongrel could take it,\" Ilagin was saying at the same time, breathless from his gallop and his excitement."
text_de = "In fünf Bundesländern haben Spezialeinheiten der Polizei am Dienstag Wohnungen wegen des Verdachts auf Terror-Planungen durchsucht. Einsatzkommandos gingen nach Angaben des Landeskriminalamtes in Erfurt in zwölf Wohnungen und einer Gemeinschaftsunterkunft in Thüringen, Hamburg, Nordrhein-Westfalen, Sachsen und Bayern vor. Anlass der Ermittlungen war nach den Angaben der Verdacht der Vorbereitung einer schweren staatsgefährdenden Gewalttat. Eine konkrete Anschlagsgefahr gab es nach den bisherigen Ermittlungen nicht. Zu Festnahmen ist es nach Auskunft der Polizei nicht gekommen. „Es gibt keinen Haftbefehl“, sagte eine Sprecherin des Thüringer Landeskriminalamtes am Dienstagnachmittag in Erfurt. Hauptverdächtig sei ein 28 Jahre alter Tschetschene mit russischem Pass, der als Asylbewerber im thüringischen Suhl leben soll."
text_en = 'I am very excited about the next generation of Apple products.'
text_en = 'I hate Apple products'

def ner_pipeline(raw_text):
    # create a list of strings
    print('sentence splitting ...')
    sentences = sentence_splitting(raw_text)

    # create a list of lists of strings
    print('word tokenizing ...')
    tokenized_sentences = [word_tokenization(sentence) for sentence in sentences]

    # create a list of lists of tuples
    print('pos tagging ...')
    pos_tagged_sentences = [part_of_speech_tagging(sentence) for sentence in tokenized_sentences]

    # create a list of nltk trees containing named entity chunks
    print('named entity token chunking ...')
    chunk_trees = [named_entity_token_chunking(sentence) for sentence in pos_tagged_sentences]

    return chunk_trees

def sentence_splitting(raw_text):
    return nltk.sent_tokenize(raw_text)

def word_tokenization(sentence):
    return nltk.word_tokenize(sentence)

def part_of_speech_tagging(sentence):
    return nltk.pos_tag(sentence)

def named_entity_token_chunking(tagged_sentence, report_ne=True):
    # return nltk.ne_chunk(tagged_sentence)

    # Loads the serialized NEChunkParser object
    chunker = nltk.data.load('chunkers/maxent_ne_chunker/PY3/english_ace_multiclass.pickle')

    # The MaxEnt classifier
    maxEnt = chunker._tagger.classifier()

    tags = []
    for i in range(0, len(tagged_sentence)):
        featureset = chunker._tagger.feature_detector(tagged_sentence, i, tags)
        tag = chunker._tagger.choose_tag(tagged_sentence, i, tags)
        # if report_ne and tag != 'O':
        if report_ne:
            print('\nExplanation on the why the word \'' + tagged_sentence[i][0] + '\' was tagged:')
            featureset = chunker._tagger.feature_detector(tagged_sentence, i, tags)
            maxEnt.explain(featureset)
        tags.append(tag)

    return tags

# helper

def draw_to_file(tree):
    global name_index

    canvas = CanvasFrame()
    tree_canvas = TreeWidget(canvas.canvas(), tree)
    canvas.add_widget(tree_canvas,10,10)

    name_index += 1
    file_name = 'tree_plot_' + str(name_index) + '.ps'

    canvas.print_to_file(file_name)
    canvas.destroy()

    # clean up using the following shell comand:
    # for i in tree_plot_*.ps; do convert $i "$i.png"; rm $i; done

# entry point as a stand alone script

if __name__ == '__main__':
    chunk_trees = ner_pipeline(text_en)
    # chunk_trees = ner_pipeline(text_de)
    pp(chunk_trees)
