from __future__ import unicode_literals, print_function, division
import queue as Q

import torch
import torch.nn as nn
import numpy as np

from models import EncoderRNN, DecoderRNN, AttnDecoderRNN
from train import evaluate, prepareData, randomPicker
from preprocess import Preprocesor

def sech(x):
    return 1./np.cosh(x)

class AutoEncoder(object):

    def __init__(self, input_lang, output_lang):
        self.input_lang = input_lang
        self.output_lang = output_lang

    def autoencoder(self, sentence):
        encoder = torch.load('trained_model/encoder_call_log_5.020941673008025_1516403418.9226089', map_location={'cuda:0': 'cpu'})
        decoder = torch.load('trained_model/decoder_call_log_5.020941673008025_1516403419.038081', map_location={'cuda:0': 'cpu'})

        output_words, _, loss = evaluate(
            encoder, decoder, sentence, self.input_lang, self.output_lang)

        return output_words, loss

def showTopAnomaly(top=5):
    lines = [line.rstrip('\n') for line in open('dataset/train.txt')]
    q = Q.PriorityQueue()
    top_anomalies = list()

    for i, line in enumerate(lines[:1000]):
        # line = replace_digits_with_words(line)
        line = p.preprocess(line)
        line = randomPicker([line, line])[0]
        decoded_text, decoded_loss = ad.autoencoder(line)
        q.put([-(decoded_loss/len(line.split())), decoded_text, i])

    for i in range(top):
        top_anomalies.append(q.get())
    return top_anomalies

if __name__ == '__main__':

    p = Preprocesor()
    text = "I hate you"
    text = p.preprocess(text)
    input_lang, output_lang, pairs, _ = prepareData('eng', 'eng', False)
    ad = AutoEncoder(input_lang, output_lang)
    decoded_text, decoded_loss = ad.autoencoder(text)
    print(decoded_text, decoded_loss)
    #top_anomaly = showTopAnomaly(10)
    #
    # lines = [line.rstrip('\n') for line in open('dataset/train.txt')]
    # for each in top_anomaly:
    #     print(-each[0], print(each[1]))
    #     print('Original Text:{}'.format(lines[each[2]]))
    #     print()

    # while True:
    #     text = input('Input:')
    #     # text = replace_digits_with_words(text)
    #     text = normalizeString(text)
    #     print('Trimmed text: ', text)
    #     text = randomPicker([text, text])[0]
    #     print('Maxlen of text by Random Index Picker: ', text)
    #     decoded_text, decoded_loss = ad.autoencoder(text)
    #     anomaly_prob = sech(decoded_loss / len(text.split()))
    #     print('Original text sequence: ', text.split())
    #     print('Decoded text sequence: ', decoded_text)
    #     print('Decoded text total loss: ', decoded_loss)
    #     print('Decoded text avg loss: ', decoded_loss / len(text.split()))
    #     print('Probability of genuine movie review: {}%'.format(anomaly_prob*100))
    #     print('================================================================================')
    #     print('Judgement of Imdb Anomaly Detector: ', 'Posiive' if anomaly_prob < 0.2 else 'Negative')