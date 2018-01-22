# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import time
import math
import os
import sys
import random

random.seed(3458878313)

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch import optim
from torch.nn.utils import clip_grad_norm
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

from models import EncoderRNN, AttnDecoderRNN
from preprocess import Preprocesor

log = ['==========Training Log==========\n\n']
use_cuda = torch.cuda.is_available()
print('CUDA available:', use_cuda)

maximum_norm = 2.0
print_loss_avg = 0
print_val_loss_avg = 0
hidden_size = 200
teacher_forcing_ratio = 0.5
MIN_LENGTH = 3
MAX_LENGTH = 15
SOS_token = 0
EOS_token = 1

log.append('===============================\n')
log.append('CUDA available: {}\n'.format(use_cuda))
log.append('Gradient Clipping Norm: {}\n'.format(maximum_norm))
log.append('Word Embedding Dimension: {}\n'.format(hidden_size))
log.append('Teacher Forcing Ratio: {}\n'.format(teacher_forcing_ratio))
log.append('Minimum Sentence Length: {}\n'.format(MIN_LENGTH))
log.append('Maximum Sentence Length: {}\n'.format(MAX_LENGTH-1))
log.append('===============================\n\n')

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "OOV"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def read_langs(lang1, lang2, reverse=False):

    p = Preprocesor()
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('dataset/train.txt', encoding = "ISO-8859-1"). \
        read().strip().split('\n')

    # Split every line into pairs and normalize
    # pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = [[p.preprocess(l), p.preprocess(l)] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def random_picker(p):
    if len(p[0].split()) >= MAX_LENGTH:
        random_index = random.randrange(len(p[0].split()) - (MAX_LENGTH - 1) + 1)
        p[0] = ' '.join(p[0].split()[random_index:random_index+MAX_LENGTH-1])
        p[1] = p[0]
    return p

def maxlen_picker(p):
    if len(p[0].split()) >= MAX_LENGTH:
        p[0] = ' '.join(p[0].split()[:MAX_LENGTH-1])
        p[1] = p[0]
    return p



def filter_pair(p):
    return MIN_LENGTH <= len(p[0].split()) < MAX_LENGTH and \
           MIN_LENGTH <= len(p[1].split()) < MAX_LENGTH


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(random_picker(pair))]


def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    print("Read {} sentence pairs".format(len(pairs)))
    log.append('===============================\n')
    log.append("Read {} sentence pairs\n".format(len(pairs)))
    pairs = filter_pairs(pairs)
    print("Trimmed to {} sentence pairs".format(len(pairs)))
    log.append("Trimmed to {} sentence pairs\n".format(len(pairs)))
    split_divider = int(len(pairs) * 0.9)
    val_pairs = pairs[split_divider:]
    pairs = pairs[:split_divider]
    print("{} sentence pairs for training".format(len(pairs)))
    print("{} sentence pairs for validation".format(len(val_pairs)))
    log.append("{} sentence pairs for training\n".format(len(pairs)))
    log.append("{} sentence pairs for validation\n".format(len(val_pairs)))

    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    log.append("{} vocabulary tokens\n".format(input_lang.n_words))
    log.append('===============================\n\n')
    return input_lang, output_lang, pairs, val_pairs


def indexes_from_sentence(lang, sentence):
    indexes = []
    for word in sentence.split():
        if lang.word2index.get(word) is not None:
            indexes.append(lang.word2index[word])
        else:
            indexes.append(2)  # OOV
    return indexes
    # return [lang.word2index[word] for word in sentence.split(' ') if lang.word2index.get(word) is not None]


def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variables_from_pair(pair):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH, is_validation=False):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            loss += criterion(decoder_output[0], target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output[0], target_variable[di])
            if ni == EOS_token:
                break

    if not is_validation:
        loss.backward()

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
    clip_grad_norm(encoder.parameters(), maximum_norm)
    clip_grad_norm(decoder.parameters(), maximum_norm)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

# def showPlot(points, val_points, epoch):
#     plt.plot(epoch, points)
#     plt.plot(epoch, val_points)
#     plt.title('Training Graph for Anomaly Detector')
#     plt.xlabel('Sentences')
#     plt.ylabel('SGD Loss')
#     plt.savefig('log/train_result_{}.png'.format(str(time.time())))
#     # plt.show()


def train_iters(encoder, decoder, n_iters, print_every=500, plot_every=500, learning_rate=0.01):
    start = time.time()
    global print_loss_avg
    global print_val_loss_avg

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    plot_val_losses = []
    print_val_loss_total = 0  # Reset every print_every
    plot_val_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [variables_from_pair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()
    epoch = []
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            epoch.append(iter)
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            log.append('%s (%d %d%%) %.4f\n' % (time_since(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

            validating_pairs = [variables_from_pair(random.choice(val_pairs))
                                for _ in range(print_every // 9)]
            for val_iter in range(1, print_every // 9 + 1):
                validating_pair = validating_pairs[val_iter - 1]
                input_variable = validating_pair[0]
                target_variable = validating_pair[1]

                val_loss = train(input_variable, target_variable, encoder,
                                 decoder, encoder_optimizer, decoder_optimizer, criterion, is_validation=True)
                print_val_loss_total += val_loss
                plot_val_loss_total += val_loss
            print_val_loss_avg = print_val_loss_total / (print_every // 9)
            print_val_loss_total = 0
            print('==========Val_Loss: %.4f==========' % (print_val_loss_avg))
            log.append('==========Val_Loss: %.4f==========\n' % (print_val_loss_avg))

            plot_val_loss_avg = plot_val_loss_total / (print_every // 9)
            plot_val_losses.append(plot_val_loss_avg)
            plot_val_loss_total = 0

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # showPlot(plot_losses, plot_val_losses, epoch)


def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    input_variable = variable_from_sentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.init_hidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    criterion = nn.NLLLoss()
    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(input_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        loss += criterion(decoder_output[0], input_variable[di])
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1], loss.data.numpy()


def evaluate_randomly(encoder, decoder, n=20):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions, loss = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


if __name__ == '__main__':

    p = Preprocesor()

    input_lang, output_lang, pairs, val_pairs = prepare_data('eng', 'eng', False)

    # pre-trained word embedding
    embeddings_index = {}
    max_features = len(input_lang.index2word)
    f = open(os.path.join('dataset/', 'glove.6B.200d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float16')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((max_features, hidden_size))
    for word, i in input_lang.word2index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    embedding_matrix = embedding_matrix

    # load model($python3 train.py encoder decoder)
    if len(sys.argv) == 3:
        print("Resume training...")
        print("Encoder path: {}".format(sys.argv[1]))
        print("Decoder path: {}".format(sys.argv[2]))
        if use_cuda:
            call_log_encoder = torch.load(sys.argv[1])
            call_log_decoder = torch.load(sys.argv[2])
        else:
            call_log_encoder = torch.load(sys.argv[1], map_location={'cuda:0': 'cpu'})
            call_log_decoder = torch.load(sys.argv[2], map_location={'cuda:0': 'cpu'})
    else:
        call_log_encoder = EncoderRNN(input_lang.n_words, hidden_size, embedding_matrix)
        call_log_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words,
                                      1, dropout_p=0.1)

    if use_cuda:
        call_log_encoder = call_log_encoder.cuda()
        call_log_decoder = call_log_decoder.cuda()

    start_time = time.time()
    train_iters(call_log_encoder, call_log_decoder, 60000, print_every=100, plot_every=100, learning_rate=0.05)

    log.append('\nTotal Training Time: {}sec\n'.format(time.time()-start_time))
    # save model
    torch.save(call_log_encoder, 'trained_model/encoder_call_log_{}_{}'.format(print_loss_avg, time.time()))
    torch.save(call_log_decoder, 'trained_model/decoder_call_log_{}_{}'.format(print_loss_avg, time.time()))
    with open('log/log_{}.txt'.format(str(time.time())), 'wt') as fo:
        fo.write(' '.join(log))
