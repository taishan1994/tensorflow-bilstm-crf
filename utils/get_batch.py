import numpy as np
class BatchGenerator(object):
    """ Construct a Data generator. The input X, y should be ndarray or list like type.
    
    Example:
        Data_train = BatchGenerator(X=X_train_all, y=y_train_all, shuffle=False)
        Data_test = BatchGenerator(X=X_test_all, y=y_test_all, shuffle=False)
        X = Data_train.X
        y = Data_train.y
        or:
        X_batch, y_batch = Data_train.next_batch(batch_size)
     """ 
    
    def __init__(self, X, y, seq_len, shuffle=False):
        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        if type(seq_len) != np.ndarray:
            seq_len = np.asarray(seq_len)
        self._X = X
        self._y = y
        self._seq_len = seq_len
        self._number_examples = self._X.shape[0] #样本的数目
        self._shuffle = shuffle #是否进行shuffle
        if self._shuffle: #如果进行shuffle,则进行以下操作
            new_index = np.random.permutation(self._number_examples)
            self._X = self._X[new_index]
            self._y = self._y[new_index]
            self._seq_len = self._seq_len[new_index]
                
    @property
    def x(self):
        return self._X
    
    @property
    def y(self):
        return self._y
    
    @property
    def seq_len(self):
        return self._seq_len

    @property
    def num_examples(self):
        return self._number_examples
    
    def next_batch(self, batch_size): #这里才是核心
      num_batch = int((self._number_examples - 1) / batch_size) + 1 #计算要迭代多少次
      for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, self._number_examples)
        yield self._X[start_id:end_id,:], self._y[start_id:end_id,:], self._seq_len[start_id:end_id]
