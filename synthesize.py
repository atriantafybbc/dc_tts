# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

import os

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_data
from data_load import load_vocab
from data_load import text_normalize
from scipy.io.wavfile import write
from tqdm import tqdm

class Synthesizer():
    def __init__(self, filename_text2mel_model, filename_ssrn_model):
        # Load data
        #self.L = load_data("synthesize")

        self._filename_text2mel_model = os.path.realpath(filename_text2mel_model)
        self._filename_ssrn_model = os.path.realpath(filename_ssrn_model)
        # Load graph
        self._graph = Graph(mode="synthesize")
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        # Restore text2mel
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(self._sess, self._filename_text2mel_model)
        # Restore ssrn
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(self._sess, self._filename_ssrn_model)
        
        self._char2idx, self._idx2char = load_vocab()        

    def encode_text(self, text):
        assert type(text) is unicode
        lines = text.splitlines()
        sents = [text_normalize(line.split(" ", 1)[-1]).strip() + "E" for line in lines] # text normalization, E: EOS
        texts = np.zeros((len(sents), hp.max_N), np.int32)
        for i, sent in enumerate(sents):
            texts[i, :len(sent)] = [char2idx[char] for char in sent]
        return texts        

    def synthesize(self, text, filename_wav):
        L = self.encode_text(text)
        Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)
        for j in tqdm(range(hp.max_T)):
            _gs, _Y, _max_attentions, _alignments = \
                self._sess.run([self._graph.global_step, self._graph.Y, self._graph.max_attentions, self._graph.alignments],
                         {self._graph.L: L,
                          self._graph.mels: Y,
                          self._graph.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]        
        # Get magnitude
        Z = self._sess.run(self._graph.Z, {self._graph.Y: Y})

        # Generate wav files
        wav_total = np.array([])
        for i, mag in enumerate(Z):
            wav_sentence = spectrogram2wav(mag)
            wav_total = np.concatenate((wav_total, wav_sentence))
        write(filename_wav, hp.sr, wav_total)

def synthesize():
    # Load data
    L = load_data("synthesize")

    # Load graph
    g = Graph(mode="synthesize"); print("Graph loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-1"))
        print("Text2Mel Restored!")

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-2"))
        print("SSRN Restored!")

        # Feed Forward
        ## mel
        Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)
        for j in tqdm(range(hp.max_T)):
            _gs, _Y, _max_attentions, _alignments = \
                sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
                         {g.L: L,
                          g.mels: Y,
                          g.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]

        # Get magnitude
        Z = sess.run(g.Z, {g.Y: Y})

        # Generate wav files
        if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
        for i, mag in enumerate(Z):
            print("Working on file", i+1)
            wav = spectrogram2wav(mag)
            write(hp.sampledir + "/{}.wav".format(i+1), hp.sr, wav)

if __name__ == '__main__':
    synthesize()
    print("Done")


