from config.globalConfig import *
from config.msraConfig import Config
from dataset.msraDataset import MsraDataset
from utils.get_batch import BatchGenerator
from models.bilstm_crf import BilstmCrfModel
import tensorflow as tf
import os
import numpy as np
from utils.tmp import find_all_tag,get_labels,get_multi_metric,mean,get_binary_metric
labels_list = ['ns','nt','nr']

def train(config,model,save_path,trainBatchGen,valBatchGen):
  globalStep = tf.Variable(0, name="globalStep", trainable=False)
  save_path = os.path.join(save_path,"best_validation")
  saver = tf.train.Saver()
  with tf.Session() as sess:
    # 定义trainOp
    # 定义优化函数，传入学习速率参数
    optimizer = tf.train.AdamOptimizer(config.trainConfig.learning_rate)
    # 计算梯度,得到梯度和变量
    gradsAndVars = optimizer.compute_gradients(model.loss)
    # 将梯度应用到变量下，生成训练器
    trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)
    sess.run(tf.global_variables_initializer())
    best_f_beta_val = 0.0 #最佳验证集的f1值
    for epoch in range(1,config.trainConfig.epoch+1):
      for trainX_batch,trainY_batch,train_seqlen in trainBatchGen.next_batch(config.trainConfig.batch_size):
        feed_dict = {
          model.inputX : trainX_batch, #[batch,max_len]
          model.inputY : trainY_batch, #[batch,max_len]
          model.seq_lens : train_seqlen, #[batch]
        }
        _, loss, pre = sess.run([trainOp,model.loss,model.viterbi_sequence],feed_dict)
        currentStep = tf.train.global_step(sess, globalStep)   
        true_idx2label = [get_labels(label,idx2label,seq_len) for label,seq_len in zip(trainY_batch,train_seqlen)] 
        pre_idx2label = [get_labels(label,idx2label,seq_len) for label,seq_len in zip(pre,train_seqlen)] 
        precision,recall,f1 = get_multi_metric(true_idx2label,pre_idx2label,train_seqlen,labels_list)
        if currentStep % 100 == 0:
          print("[train] step:{} loss:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(currentStep,loss,precision,recall,f1))
        if currentStep % 100 == 0:
          #要计算所有验证样本的
          losses = []
          f_betas = []
          precisions = []
          recalls = []
          for valX_batch,valY_batch,val_seqlen in valBatchGen.next_batch(config.trainConfig.batch_size):
            feed_dict = {
              model.inputX : valX_batch, #[batch,max_len]
              model.inputY : valY_batch, #[batch,max_len]
              model.seq_lens : val_seqlen, #[batch]
            }
            val_loss, val_pre = sess.run([model.loss,model.viterbi_sequence],feed_dict)
            val_true_idx2label = [get_labels(label,idx2label,seq_len) for label,seq_len in zip(valY_batch,val_seqlen)] 
            val_pre_idx2label = [get_labels(label,idx2label,seq_len) for label,seq_len in zip(val_pre,val_seqlen)] 
            val_precision,val_recall,val_f1 = get_multi_metric(val_true_idx2label,val_pre_idx2label,val_seqlen,labels_list)
            losses.append(val_loss)
            f_betas.append(val_f1)
            precisions.append(val_precision)
            recalls.append(val_recall)
          if mean(f_betas) > best_f_beta_val:
            # 保存最好结果
            best_f_beta_val = mean(f_betas)
            last_improved = currentStep
            saver.save(sess=sess, save_path=save_path)
            improved_str = '*'
          else:
            improved_str = ''
          print("[val] loss:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f} {}".format(
            mean(losses),mean(precisions),mean(recalls),mean(f_betas),improved_str
          ))
def test(config,model,save_path,testBatchGen):
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()  
    ckpt = tf.train.get_checkpoint_state('checkpoint/msra/')
    path = ckpt.model_checkpoint_path
    saver.restore(sess, path)  # 读取保存的模型
    precisions = []
    recalls = []
    f1s = []
    for testX_batch,testY_batch,test_seqlen in testBatchGen.next_batch(config.trainConfig.batch_size):
      feed_dict = {
        model.inputX : testX_batch, #[batch,max_len]
        model.inputY : testY_batch, #[batch,max_len]
        model.seq_lens : test_seqlen, #[batch]
      }
      test_pre = sess.run([model.viterbi_sequence],feed_dict) #这里有点奇怪，和train、val出来的数据相比多了一个[]
      test_pre = test_pre[0] 
      test_true_idx2label = [get_labels(label,idx2label,seq_len) for label,seq_len in zip(testY_batch,test_seqlen)] 
      test_pre_idx2label = [get_labels(label,idx2label,seq_len) for label,seq_len in zip(test_pre,test_seqlen)] 
      precision,recall,f1 = get_multi_metric(test_true_idx2label,test_pre_idx2label,test_seqlen,labels_list)
      precisions.append(precision)
      recalls.append(recall)
      f1s.append(f1)
    print("[test] precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(
            mean(precisions),mean(recalls),mean(f1s)))

def predict(word2idx,idx2word,idx2label):
  max_len = 60
  input_list = []
  input_len = []
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()  
    ckpt = tf.train.get_checkpoint_state('checkpoint/msra/')
    path = ckpt.model_checkpoint_path
    saver.restore(sess, path)  # 读取保存的模型
    while True:
      print("请输入一句话：")
      line = input()
      if line == 'q':
        break
      line_len = len(line)
      input_len.append(line_len)
      word_list = [word2idx[word] if word in word2idx else word2idx['unknow'] for word in line]
      if line_len < max_len:
        word_list =word_list + [0]*(max_len-line_len)
      else:
        word_list = word_list[:max_len]
      input_list.append(word_list) #需要增加一个维度
      input_list = np.array(input_list)
      input_label = np.zeros((input_list.shape[0],input_list.shape[1])) #标签占位
      input_len = np.array(input_len) 
      feed_dict = {
        model.inputX : input_list, #[batch,max_len]
        model.inputY : input_label, #[batch,max_len]
        model.seq_lens : input_len, #[batch]
      }
      pred_label = sess.run([model.viterbi_sequence],feed_dict)
      pred_label = pred_label[0]
      # 将预测标签id还原为真实标签
      pred_idx2label = [get_labels(label,idx2label,seq_len) for label,seq_len in zip(pred_label,input_len)]
      for line,pre,s_len in zip(input_list,pred_idx2label,input_len):
        res = find_all_tag(pre,s_len)
        for k in res:
          for v in res[k]:
            if v:
              print(k,"".join([idx2word[word] for word in line[v[0]:v[0]+v[1]]]))
      input_list = []
      input_len = []



          

if __name__ == "__main__":
  config = Config()
  msraDataset = MsraDataset(config)
  word2idx = msraDataset.get_word2idx()
  idx2word = msraDataset.get_idx2word()
  label2idx = msraDataset.get_label2idx()
  idx2label = msraDataset.get_idx2label()
  embedding_pre = msraDataset.get_embedding()
  x_train,y_train,z_train = msraDataset.get_train_data()
  x_val,y_val,z_val = msraDataset.get_val_data()
  x_test,y_test,z_test = msraDataset.get_test_data()
  print("====验证是否得到相关数据===")
  print("word2idx:",len(word2idx))
  print("idx2word:",len(idx2word))
  print("label2idx:",len(label2idx))
  print("idx2label:",len(idx2label))
  print("embedding_pre:",embedding_pre.shape)
  print(x_train.shape,y_train.shape,z_train.shape)
  print(x_val.shape,y_val.shape,z_val.shape)
  print(x_test.shape,y_test.shape,z_test.shape)
  print("======打印相关参数======")
  print("batch_size:",config.trainConfig.batch_size)
  print("learning_rate:",config.trainConfig.learning_rate)
  print("embedding_dim:",config.msraConfig.embedding_dim)
  is_train,is_val,is_test = True,True,True
  model = BilstmCrfModel(config,embedding_pre)
  if is_train:
    trainBatchGen = BatchGenerator(x_train,y_train,z_train,shuffle=True)
  if is_val:
    valBatchGen = BatchGenerator(x_val,y_val,z_val,shuffle=False)
  if is_test:
    testBatchGen = BatchGenerator(x_test,y_test,z_test,shuffle=False)
  dataset = "msra"
  if dataset == "msra":
    save_path = os.path.join(PATH,'checkpoint/msra/')
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  #train(config,model,save_path,trainBatchGen,valBatchGen)
  #test(config,model,save_path,testBatchGen)
  predict(word2idx,idx2word,idx2label)

