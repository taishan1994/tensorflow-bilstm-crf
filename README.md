# tensorflow-bilstm-crf
基于tensorflow的bilstm+crf的命名实体识别

## Introduction
该项目是基于tensorflow的命名实体识别，使用的基础模型是bilstm-crf<br>
目录结构：<br>
|--checkpint：保存模型目录<br>
|--|--msra：模型保存位置；<br>
|--config：配置文件；<br>
|--|--msraConfig.py：包含训练配置、模型配置、数据集配置；<br>
|--|--globaConfig.py：全局配置文件，主要是全局路径、全局参数等；<br>
|--data：数据保存位置；<br>
|--|--msra：数据；<br>
|--process：对数据进行预处理；<br>
|--models：模型保存文件；<br>
|--process_data：对原始数据进行处理后的数据；<br>
|--utils：辅助函数保存位置；<br>
|--main.py：主运行文件，选择模型、训练、测试和预测；<br>

需要的环境
tensorflow == 1.15.0 <br>

## 运行
main.py中：<br>
```python
train(config,model,save_path,trainBatchGen,valBatchGen)
test(config,model,save_path,testBatchGen)
predict(word2idx,idx2word,idx2label)
```
分别对应：训练和验证、测试、预测<br>
需要注意的是，在进行预测的时候，是在终端输入一句话，会返回命名实体识别的结果，当输入q的时候退出。<br>

## 相关结果
```python
====验证是否得到相关数据===
word2idx: 4026
idx2word: 4025
label2idx: 11
idx2label: 11
embedding_pre: (4027, 100)
(36066, 60) (36066, 60) (36066,)
(9016, 60) (9016, 60) (9016,)
(5009, 60) (5009, 60) (5009,)
======打印相关参数======
batch_size: 128
learning_rate: 0.001
embedding_dim: 100
。。。。。。
[train] step:10500 loss:0.8870 precision:0.9699 recall:0.9734 f1:1.0000
[val] loss:1.5881 precision:0.8432 recall:0.8569 f1:0.7964 
[train] step:10600 loss:1.3130 precision:0.9445 recall:0.9314 f1:0.8696
[val] loss:1.6127 precision:0.8302 recall:0.8720 f1:0.7913 
[train] step:10700 loss:0.9924 precision:0.9762 recall:0.9730 f1:0.9836
[val] loss:1.6147 precision:0.8490 recall:0.8488 f1:0.7923 
[train] step:10800 loss:0.9863 precision:0.9481 recall:0.9495 f1:0.9697
[val] loss:1.6455 precision:0.8375 recall:0.8672 f1:0.8015 
[train] step:10900 loss:1.0629 precision:0.9580 recall:0.9148 f1:0.9355
[val] loss:1.7692 precision:0.8174 recall:0.8064 f1:0.7735 
[train] step:11000 loss:0.9846 precision:0.9800 recall:0.9689 f1:0.9643
[val] loss:1.5785 precision:0.8562 recall:0.8544 f1:0.7996 
[train] step:11100 loss:0.9315 precision:0.9609 recall:0.9651 f1:0.9455
[val] loss:1.5665 precision:0.8568 recall:0.8646 f1:0.8100 *
[train] step:11200 loss:0.9989 precision:0.9804 recall:0.9629 f1:0.9697
[val] loss:1.5807 precision:0.8519 recall:0.8569 f1:0.7941
```
```python
[test] precision:0.8610 recall:0.8780 f1:0.8341
```
```python
请输入一句话：
我要感谢洛杉矶市民议政论坛、亚洲协会南加中心、美中关系全国委员会、美中友协美西分会等友好团体的盛情款待。
2020-11-15 08:02:48.049927: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
nt 洛杉矶市民议政论坛、亚洲协会南加中心、美中关系全国委员会、美中友协美西分会
请输入一句话：
今天的演讲会是由哈佛大学费正清东亚研究中心主任傅高义主持的。
nt 哈佛大学
nt 清东亚研究中心
nr 傅高义
请输入一句话：
美方有哈佛大学典礼官亨特、美国驻华大使尚慕杰等。
nr 亨特、
nr 尚慕杰
请输入一句话：
q
q
```



