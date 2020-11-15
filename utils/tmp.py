pre = "0 0 B_SONG M_SONG M_SONG E_SONG 0 B_SONG M_SONG M_SONG E_SONG 0 0 B_SINGER M_SINGER M_SINGER E_SINGER 0 O O O B_ALBUM M_ALBUM M_ALBUM E_ALBUM O O B_TAG M_TAG M_TAG E_TAG O"
true = "0 0 B_SONG M_SONG M_SONG E_SONG 0 0 0 0 0 0 0 B_SINGER M_SINGER M_SINGER E_SINGER 0 O O O B_ALBUM M_ALBUM M_ALBUM E_ALBUM O O B_TAG M_TAG M_TAG E_TAG O"
#tags = [("B_SONG","I_SONG"),("B_SINGER","I_SINGER"),("B_ALBUM","I_ALBUM"),("B_TAG","I_TAG")]

"""
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
"""

#tags = [('B_SONG','M_SONG','E_SONG'),('B_SINGER','M_SINGER','E_SINGER'),('B_ALBUM','M_ALBUM','E_ALBUM'),('B_TAG','M_TAG','E_TAG')]
tags = [('B_ns','M_ns','E_ns'),('B_nt','M_nt','E_nt'),('B_nr','M_nr','E_nr')]
labels_list = ['ns','nt','nr']

#将结果转换为标签
def get_labels(labels,idx2label,seq_len):
  char_labels = []
  for i in range(seq_len):
    char_labels.append(idx2label[labels[i]])
  return char_labels


def find_tag(labels,seq_len,B_label="B_ns",M_label="M_ns",E_label='E_ns'):
    result = []
    if isinstance(labels,str): # 如果labels是字符串
        labels = labels.strip().split() # 将labels进行拆分
        labels = ["O" if label =="0" else label for label in labels] # 如果标签是O就就是O，否则就是label
        # print(labels)
    for num in range(seq_len): 
        if labels[num] == B_label: 
            song_pos0 = num # 记录B_SONG的位置
        if labels[num] == M_label and labels[num-1] == B_label: # 如果当前lable是I_SONG且前一个是B_SONG
            lenth = 2 # 当前长度为2 
            for num2 in range(num,seq_len): # 从该位置开始继续遍历
                if labels[num2] == M_label and labels[num2-1] == M_label: # 如果当前位置和前一个位置是I_SONG
                    lenth += 1 # 长度+1
                if labels[num2] == E_label: # 如果当前标签是结尾
                    lenth += 1
                    result.append((song_pos0,lenth)) #z则取得B的位置和长度
                    break # 退出第二个循环
    return result

def find_all_tag(labels,seq_len):

    result = {}
    for tag in tags:
        res = find_tag(labels,seq_len,B_label=tag[0],M_label=tag[1],E_label=tag[2])
        result[tag[0].split("_")[1]] = res # 将result赋值给就标签
    return result

#精确率是指预测为正样本的个数有多少个是正确的
def binary_precision(pre_labels,true_labels,seq_len,positive='ns'):
    '''
    :param pre_tags: list
    :param true_tags: list
    :return:
    '''
    if isinstance(pre_labels,str):
        pre_labels = pre_labels.strip().split() # 字符串转换为列表
        pre_labels = ["O" if label =="0" else label for label in pre_labels]
    if isinstance(true_labels,str):
        true_labels = true_labels.strip().split()
        true_labels = ["O" if label =="0" else label for label in true_labels]
    corr = 0
    pred_corr = 0
    for pre_label,true_label,s_len in zip(pre_labels,true_labels,seq_len):
      pre_result = find_all_tag(pre_label,s_len) # pre_result是一个字典，键是标签，值是一个元组，第一位是B的位置，第二位是长度
      true_result = find_all_tag(true_label,s_len)
      for k in pre_result:
        for x in pre_result[k]:
          if x:
            if k == positive:
              pred_corr += 1
              if pre_label[x[0]:x[0]+x[1]] == true_label[x[0]:x[0]+x[1]]: # 判断对应位置的每个标签是否一致
                corr += 1
    return (corr / pred_corr) if pred_corr > 0 else 0 #为1的个数/总个数



#召回率就是在正样本中有多少个预测出来了
def binary_recall(pre_labels,true_labels,seq_len,positive='ns'):
    '''
    :param pre_tags: list
    :param true_tags: list
    :return:
    '''
    recall = []
    if isinstance(pre_labels,str):
        pre_labels = pre_labels.strip().split()
        pre_labels = ["O" if label =="0" else label for label in pre_labels]
    if isinstance(true_labels,str):
        true_labels = true_labels.strip().split()
        true_labels = ["O" if label =="0" else label for label in true_labels]

    corr = 0
    true_corr = 0
    for pre_label,true_label,s_len in zip(pre_labels,true_labels,seq_len):
      pre_result = find_all_tag(pre_label,s_len) # pre_result是一个字典，键是标签，值是一个元组，第一位是B的位置，第二位是长度
      true_result = find_all_tag(true_label,s_len)
      for k in true_result:
        for x in true_result[k]:
          if x:
            if k == positive:
              true_corr += 1
              if pre_label[x[0]:x[0]+x[1]] == true_label[x[0]:x[0]+x[1]]: # 判断对应位置的每个标签是否一致
                corr += 1
    return (corr / true_corr) if true_corr > 0 else 0 #为1的个数/总个数

def mean(item: list) -> float:
    """
    计算列表中元素的平均值
    :param item: 列表对象
    :return:
    """
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res

def binary_f1_score(pre_labels,true_labels,seq_len,positive='ns'):
  precision = binary_precision(pre_labels,true_labels,seq_len,positive='ns')
  recall = binary_recall(pre_labels,true_labels,seq_len,positive='ns')
  try:
    f1 = (2*precision*recall)/(precision+recall)
  except:
    f1 = 0
  return f1 # 有了precision和recall，计算F1就简单了

def multi_precision(pre_labels,true_labels,seq_len,labels_list):
  precision = mean([binary_precision(pre_labels,true_labels,seq_len,positive=label) for label in labels_list])
  return precision

def multi_recall(pre_labels,true_labels,seq_len,labels_list):
  recall = mean([binary_recall(pre_labels,true_labels,seq_len,positive=label) for label in labels_list])
  return recall

def multi_f1_score(pre_labels,true_labels,seq_len,labels_list):
  f1 = mean([binary_f1_score(pre_labels,true_labels,seq_len,positive=label) for label in labels_list])
  return f1

def get_binary_metric(pre_labels,true_labels,seq_len,positive='ns'):
  precision = binary_precision(pre_labels,true_labels,seq_len,positive='ns')
  recall = binary_recall(pre_labels,true_labels,seq_len,positive='ns')
  f1 = binary_f1_score(pre_labels,true_labels,seq_len,positive='ns')
  return precision,recall,f1

def get_multi_metric(pre_labels,true_labels,seq_len,labels_list):
  precision = multi_precision(pre_labels,true_labels,seq_len,labels_list)
  recall = multi_recall(pre_labels,true_labels,seq_len,labels_list)
  f1 = multi_f1_score(pre_labels,true_labels,seq_len,labels_list)
  return precision,recall,f1


