import pickle
import numpy as np
import codecs

class MsraDataset:
  def __init__(self,config):
    self.config = config
    self.pickle_path = self.config.msraConfig.pickle_path
    self.charVec_path = self.config.msraConfig.charVec_path
    self.embedding_pre = []
    self.get_info()
    if config.msraConfig.pre_trained:
      self.embedding_pre = self.get_embedding()
  
  def get_info(self):
    with open(self.pickle_path, 'rb') as inp:
      self.word2idx = pickle.load(inp)
      self.idx2word = pickle.load(inp)
      self.label2idx = pickle.load(inp)
      self.idx2label = pickle.load(inp)
      self.x_train = pickle.load(inp)
      self.y_train = pickle.load(inp)
      self.z_train = pickle.load(inp)
      self.x_test = pickle.load(inp)
      self.y_test = pickle.load(inp)
      self.z_test = pickle.load(inp)
      self.x_valid = pickle.load(inp)
      self.y_valid = pickle.load(inp)
      self.z_valid = pickle.load(inp)
  
  def get_embedding(self):
    word2vec = {}
    embedding = []
    with codecs.open(self.charVec_path, 'r', encoding='utf-8') as ef:
      for line in ef.readlines():
        word2vec[line.split()[0]] = map(eval,line.split()[1:]) #这里eval用于将字符串转换为浮点型
      unknow_pre = [1 for _ in range(self.config.msraConfig.embedding_dim)]
      embedding.append(unknow_pre) #
      for word in self.get_word2idx(): #遍历由键组成的字列表
        if word in word2vec:
          embedding.append(word2vec[word])
        else:
          embedding.append(unknow_pre) #不存在字列表中的词向量就用[1,1,...,1]表示
    embedding = np.asarray(embedding)
    return embedding


  def get_word2idx(self):
    return self.word2idx
  
  def get_idx2word(self):
    return self.idx2word

  def get_label2idx(self):
    return self.label2idx
  
  def get_idx2label(self):
    return self.idx2label
  
  def get_train_data(self):
    return self.x_train, self.y_train, self.z_train
  
  def get_val_data(self):
    return self.x_valid, self.y_valid, self.z_valid
  
  def get_test_data(self):
    return self.x_test, self.y_test, self.z_test