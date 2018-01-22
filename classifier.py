from __future__ import unicode_literals, print_function, division
import queue as Q

import torch
import torch.nn as nn
import numpy as np

from models import EncoderRNN, DecoderRNN, AttnDecoderRNN
from train import evaluate, prepare_data, random_picker
from preprocess import Preprocesor

def sech(x):
    return 1./np.cosh(x)

class AutoEncoder(object):

    def __init__(self, encoder, decoder, input_lang, output_lang):
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.encoder = encoder
        self.decoder = decoder

    def autoencoder(self, sentence):
        encoder = torch.load('trained_model/' + self.encoder, map_location={'cuda:0': 'cpu'})
        decoder = torch.load('trained_model/' + self.decoder, map_location={'cuda:0': 'cpu'})

        output_words, _, loss = evaluate(
            encoder, decoder, sentence, self.input_lang, self.output_lang)

        return output_words, loss

    def binary_classifier(self, sentence):
        pass


if __name__ == '__main__':

    p = Preprocesor()
    input_lang, output_lang, pairs, _ = prepare_data('eng', 'eng', False)
    ad = AutoEncoder('encoder', 'decoder', input_lang, output_lang)

    while True:
        text = input('Input:')
        if 'justin bieber' not in text.lower():
            print('Detection Result: Safe')
            print()
            continue
        text = p.preprocess(text)
        print('Trimmed text: ', text)
        text = random_picker([text, text])[0]
        print('Maxlen of text by Random Index Picker: ', text)
        decoded_text, decoded_loss = ad.autoencoder(text)
        print('Original text sequence: ', text.split())
        print('Decoded text sequence: ', decoded_text)
        print('Decoded text total loss: ', decoded_loss)
        print('Decoded text avg loss: ', decoded_loss / len(text.split()))
        print('Detection Result: {}'.format('Unsafe' if decoded_loss / len(text.split()) < 0.5 else 'Safe'))
        print()