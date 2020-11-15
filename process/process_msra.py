#coding:utf-8
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #当前程序上上一级目录，这里为ner
import sys
sys.path.append(BASE_DIR)
print(BASE_DIR)
import codecs
import re
import pandas as pd
import numpy as np
from config.globalConfig import *

#============================第一步：给每一个字打上标签===================================
def wordtag():
    #用utf-8-sig编码的原因是文本保存时包含了BOM（Byte Order Mark，字节顺序标记，\ufeff出现在文本文件头部，为了去掉这个
    input_data = codecs.open(os.path.join(PATH,'data/msra/train.txt'),'r','utf-8-sig') #一般使用codes打开文件，不会出现编码问题
    output_data = codecs.open(os.path.join(PATH,'data/msra/wordtag.txt'),'w','utf-8') 
    for line in input_data.readlines():
        #line=re.split('[，。；！：？、‘’“”]/[o]'.decode('utf-8'),line.strip())
        line = line.strip().split()
        
        if len(line)==0: #过滤掉''
            continue
        for word in line: #遍历列表中的每一个词
            word = word.split('/') #['希望工程', 'o']，每个词是这样的了
            if word[1]!='o': #如果不是o
                if len(word[0])==1: #如果是一个字，那么就直接给标签
                    output_data.write(word[0]+"/B_"+word[1]+" ")
                elif len(word[0])==2: #如果是两个字则拆分给标签
                    output_data.write(word[0][0]+"/B_"+word[1]+" ")
                    output_data.write(word[0][1]+"/E_"+word[1]+" ")
                else: #如果两个字以上，也是拆开给标签
                    output_data.write(word[0][0]+"/B_"+word[1]+" ")
                    for j in word[0][1:len(word[0])-1]:
                        output_data.write(j+"/M_"+word[1]+" ")
                    output_data.write(word[0][-1]+"/E_"+word[1]+" ")
            else: #如果表示前是o的话，将拆开为字并分别给标签/o
                for j in word[0]:
                    output_data.write(j+"/o"+" ")
        output_data.write('\n')               
    input_data.close()
    output_data.close()

#============================第二步：构建二维字列表以及其对应的二维标签列表===================================
wordtag()
datas = list()
labels = list()
linedata=list()
linelabel=list()

# 0表示补全的id
tag2id = {'' :0,
'B_ns' :1,
'B_nr' :2,
'B_nt' :3,
'M_nt' :4,
'M_nr' :5,
'M_ns' :6,
'E_nt' :7,
'E_nr' :8,
'E_ns' :9,
'o': 10}

id2tag = {0:'' ,
1:'B_ns' ,
2:'B_nr' ,
3:'B_nt' ,
4:'M_nt' ,
5:'M_nr' ,
6:'M_ns' ,
7:'E_nt' ,
8:'E_nr' ,
9:'E_ns' ,
10: 'o'}


input_data = codecs.open(os.path.join(PATH,'data/msra/wordtag.txt'),'r','utf-8')
for line in input_data.readlines(): #每一个line实际上是这样子的：当/o 希/o 望/o 工/o 程/o 救/o 助/o  注意最后多了个''
  line=re.split('[，。；！：？、‘’“”]/[o]'.encode("utf-8").decode('utf-8'),line.strip()) #a按指定字符划分字符串
  for sen in line: #
      sen = sen.strip().split() #每一个字符串列表再按照弄空格划分，然后每个字是：当/o
      if len(sen)==0: #过滤掉为空的
          continue
      linedata=[]
      linelabel=[]
      num_not_o=0
      for word in sen: #遍历每一个字
          word = word.split('/') #第一位是字，第二位是标签
          linedata.append(word[0]) #加入到字列表
          linelabel.append(tag2id[word[1]]) #加入到标签列表，要转换成对应的id映射

          if word[1]!='o':
              num_not_o+=1 #记录标签不是o的字的个数
      if num_not_o!=0: #如果num_not_o不为0，则表明当前linedata和linelabel有要素
          datas.append(linedata) 
          labels.append(linelabel)
            
input_data.close()    
print(len(datas))
print(len(labels))

#============================第三步：构建word2id以及id2word=================================== 
#from compiler.ast import flatten (在python3中不推荐使用)，我们自己定义一个
def flat2gen(alist):
  for item in alist:
    if isinstance(item, list):
      for subitem in item: yield subitem
    else:
      yield item
all_words = list(flat2gen(datas)) #获得包含所有字的列表
sr_allwords = pd.Series(all_words) #转换为pandas中的Series
sr_allwords = sr_allwords.value_counts() #统计每一个字出现的次数，相当于去重
set_words = sr_allwords.index #每一个字就是一个index，这里的字按照频数从高到低排序了
set_ids = range(1, len(set_words)+1) #给每一个字一个id映射，注意这里是从1开始，因为我们填充序列时使用0填充的，也就是id为0的已经被占用了 
word2id = pd.Series(set_ids, index=set_words) #字 id
id2word = pd.Series(set_words, index=set_ids) #id 字
 
word2id["unknow"] = len(word2id)+1 #加入一个unknow，如果没出现的字就用unknow的id代替

#============================第四步：定义序列最大长度，对序列进行处理================================== 
max_len = MAX_LEN #句子的最大长度
def X_padding(words):
  """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
  ids = list(word2id[words])
  if len(ids) >= max_len:  # 长则弃掉
      return ids[:max_len]
  ids.extend([0]*(max_len-len(ids))) # 短则补全
  return ids

def y_padding(ids):
  """把 tags 转为 id 形式， 并自动补全位 max_len 长度。"""
  if len(ids) >= max_len:  # 长则弃掉
      return ids[:max_len]
  ids.extend([0]*(max_len-len(ids))) # 短则补全
  return ids

def get_true_len(ids):
  return len(ids)

df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas))) #DataFrame，索引是序列的个数，列是字序列以及对应的标签序列
df_data['length'] = df_data["tags"].apply(get_true_len) #获得每个序列真实的长度
df_data['length'][df_data['length'] > MAX_LEN] = MAX_LEN #这里需要注意，如果序列长度大于最大长度，则其真实长度必须设定为最大长度，否则后面会报错
df_data['x'] = df_data['words'].apply(X_padding) #超截短补，新定义一列
df_data['y'] = df_data['tags'].apply(y_padding) #超截短补，新定义一列
x = np.asarray(list(df_data['x'].values)) #转为list
y = np.asarray(list(df_data['y'].values)) #转为list
length = np.asarray(list(df_data['length'].values)) #转为list

#============================第四步：划分训练集、测试集、验证集================================== 
#from sklearn.model_selection import train_test_split
#x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=43) #random_state：避免每一个划分得不同
#x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,  test_size=0.2, random_state=43)
#我们要加入每个序列的长度，因此sklearn自带的划分就没有用了，自己写一个
def split_data(data,label,seq_length,ratio):
  len_data = data.shape[0]
  #设置随机数种子，保证每次生成的结果都是一样的
  np.random.seed(43)
  #permutation随机生成0-len(data)随机序列
  shuffled_indices = np.random.permutation(len_data)
  #test_ratio为测试集所占的百分比
  test_set_size = int(len_data * ratio)  
  test_indices = shuffled_indices[:test_set_size]
  train_indices = shuffled_indices[test_set_size:]
  train_data = data[train_indices,:]
  train_label = label[train_indices]
  train_seq_length = seq_length[train_indices]
  test_data = data[test_indices,:]
  test_label = label[test_indices]
  test_seq_length = seq_length[test_indices]
  return train_data,test_data,train_label,test_label,train_seq_length,test_seq_length
x_train,x_test, y_train, y_test, z_train, z_test = split_data(x, y, seq_length=length, ratio=0.1) #random_state：避免每一个划分得不同
x_train, x_valid, y_train, y_valid, z_train, z_valid = split_data(x_train, y_train, seq_length=z_train, ratio=0.2)

#============================第五步：将所有需要的存为pickle文件备用================================== 
print('Finished creating the data generator.')
import pickle
import os
with open(os.path.join(PATH,'process_data/msra/MSRA.pkl'), 'wb') as outp:
  pickle.dump(word2id, outp)
  pickle.dump(id2word, outp)
  pickle.dump(tag2id, outp)
  pickle.dump(id2tag, outp)
  pickle.dump(x_train, outp)
  pickle.dump(y_train, outp)
  pickle.dump(z_train, outp)
  pickle.dump(x_test, outp)
  pickle.dump(y_test, outp)
  pickle.dump(z_test, outp)
  pickle.dump(x_valid, outp)
  pickle.dump(y_valid, outp)
  pickle.dump(z_valid, outp)
print('** Finished saving the data.')
