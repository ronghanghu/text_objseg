from __future__ import absolute_import, division, print_function

import tensorflow as tf

from util.rnn import lstm_layer as lstm

def lstm_net(text_seq_batch, num_vocab, embed_dim, lstm_dim):
    # embedding matrix with each row containing the embedding vector of a word
    # this has to be done on CPU currently
    with tf.variable_scope('word_embedding'), tf.device("/cpu:0"):
        embedding_mat = tf.get_variable("embedding", [num_vocab, embed_dim])
        # text_seq has shape [T, N] and embedded_seq has shape [T, N, D].
        embedded_seq = tf.nn.embedding_lookup(embedding_mat, text_seq_batch)

    lstm_top = lstm('lstm_lang', embedded_seq, None, output_dim=lstm_dim,
                    num_layers=1, forget_bias=1.0, apply_dropout=False,
                    concat_output=False)[-1]
    return lstm_top
