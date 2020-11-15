from .globalConfig import PATH
import os

class TrainConfig:
  batch_size = 128
  learning_rate = 0.001
  epoch = 20

class MsraConfig:
  pickle_path = os.path.join(PATH,'process_data/msra/MSRA.pkl')
  charVec_path = os.path.join(PATH,'process_data/vec.txt')
  embedding_size = 4027 #字的总数目
  embedding_dim = 100 #字向量的维度
  max_len = 60
  tag_size = 11
  pre_trained = True

class Config:
  msraConfig = MsraConfig()
  trainConfig = TrainConfig()