# -*- coding: utf-8 -*
import numpy as np
import tensorflow as tf

class BilstmCrfModel:
    def __init__(self,config,embedding_pretrained,dropout_keep=1):
      self.embedding_size = config.msraConfig.embedding_size
      self.embedding_dim = config.msraConfig.embedding_dim
      self.max_len = config.msraConfig.max_len
      self.tag_size = config.msraConfig.tag_size
      self.pretrained = config.msraConfig.pre_trained
      self.dropout_keep = dropout_keep
      self.embedding_pretrained = embedding_pretrained
      self.inputX = tf.placeholder(dtype=tf.int32, shape=[None,self.max_len], name="input_data") 
      self.inputY = tf.placeholder(dtype=tf.int32,shape=[None,self.max_len], name="labels")
      self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[None])
      self._build_net()

    def _build_net(self):
      # word_embeddings:[4027,100]
      # 词嵌入层
      with tf.name_scope("embedding"):
        # 利用预训练的词向量初始化词嵌入矩阵
        if self.pretrained:
            embedding_w = tf.Variable(tf.cast(self.embedding_pretrained, dtype=tf.float32, name="word2vec"),
                                      name="embedding_w")
        else:
            embedding_w = tf.get_variable("embedding_w", shape=[self.embedding_size, self.embedding_dim],
                                          initializer=tf.contrib.layers.xavier_initializer())
        # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
        input_embedded = tf.nn.embedding_lookup(embedding_w, self.inputX)
        input_embedded = tf.nn.dropout(input_embedded,self.dropout_keep)

      with tf.name_scope("bilstm"):
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_dim, forget_bias=1.0, state_is_tuple=True)
        (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, 
                                          lstm_bw_cell, 
                                          input_embedded,
                                          dtype=tf.float32,
                                          time_major=False,
                                          scope=None)
        bilstm_out = tf.concat([output_fw, output_bw], axis=2)

      # Fully connected layer.
      with tf.name_scope("output"):
        W = tf.get_variable(
            "output_w",
            shape=[2 * self.embedding_dim, self.tag_size],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[self.max_len, self.tag_size]), name="output_b")
        self.bilstm_out = tf.tanh(tf.matmul(bilstm_out, W) + b)
      with tf.name_scope("crf"):
        # Linear-CRF.
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.bilstm_out, self.inputY, self.seq_lens)

        self.loss = tf.reduce_mean(-log_likelihood)
        self.viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(self.bilstm_out, self.transition_params, self.seq_lens)
