# 机器学习期末项目

本项目为同济大学2024机器学习期末项目，基于以下两个github开源库

* [Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)
* [scikit-learn官方库](https://github.com/scikit-learn/scikit-learn)

## 一、传统机器学习方法

```
python run_tradition.py
```

* 会提示输入dataset和method，输入即可

* dataset需要自己进行文件夹的构造，格式如下

  ```
  ├─data
  │      class.txt // 分类的类别标签，用于输出
  │      dev.txt	 // 验证集
  │      <embedding_SougouNews.npz> // 如需要，预训练字向量的numpy文件
  │      <sgns.sogou.word>          // 如需要，预训练字向量文件
  │      test.txt  // 测试集
  │      train.txt // 训练集
  │
  ├─log  			// 日志文件
  └─saved_dict	// 保存模型
  ```

  + 对于`class.txt`，每一行为对应“行数-1”类别的标签（如第一行为类别0的标签）
  + 对于`train/dev/test.txt`，每一行为一条数据，为`<sentencce> <classIndex>`的格式

* method可在logistic，knn，bayes,tree和mlp中选择
  * 输入后会自动开始训练，训练完成后模型保存到相应dataset文件夹下的saved_dict里面



## 二、现代深度学习方法

### 训练

```python
python run.py --Model <ModelName> --word <True/False> --embedding <True/False>
```

* 运行run文件，可以选择模型，是否分词输入，以及嵌入层是否随机初始（不适用预训练向量）

* 可以选择的模型有TextCNN、TextRNN、TextRCNN

* 选择分词后，代码会将文本按空格分割后，输入进模型，因此对于中文请预先在文本数据中进行分词

* 嵌入层可以选择随机初始化，则跳过预训练词/字向量的初始化

* 如果需要切换数据集，需要同样自己构造对应数据集文件夹，结构同上述文件树，然后在`run.py`文件中修改变量名为文件夹名即可

  ```python
  # run.py
  if __name__ == '__main__':
      # dataset = 'THUCNews'  # 清华新闻数据集
      dataset = 'THUCNewsWord'  # 清华新闻分词数据集
      # dataset = 'weibo_4modes' # 三十六万微博四分类数据集
      # dataset = 'weibo_senti_100k'  # 十万条微博两分类数据集
      # dataset = 'yelp_review_full'  # 七十万条Yelp五分类数据集
      # dataset = 'sst_2'  # SST二分类数据集
  
  
  ```

  

### 预训练词/字向量

* 如果需要使用预训练的字/词向量，需要将文件如上述文件树格式放入data文件中，并执行`utils.py`进行对应词典和索引向量的预购造，方便后续模型初始化。

  ```python
  # util.py
  if __name__ == "__main__":
      '''提取预训练词向量'''
      # 下面的目录、文件名按需更改。
      train_dir = "./THUCNews/data/train.txt"
      vocab_dir = "./THUCNews/data/vocab.pkl"
      pretrain_dir = "./THUCNews/data/sgns.sogou.char"
      emb_dim = 300
      filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
  ```

  

* 然后在`run.py`文件中修改`embedding`变量为对应预训练向量的numpy数组文件即可

  ```python
  # run.py    
      # 搜狗新闻:embedding_SougouNews.npz, 随机初始化:random
      embedding = 'embedding_SougouNews.npz'
  ```

  

### 测试

* 对于训练完的模型（即save_dict下有权重文件），可以使用`test`命令行参数进行测试，进行人为的输入并令模型进行对应类别的输出

  ```cmd
  Vocab size: 4762
  Enter a sentence: 丰台丽泽商圈一品公馆推5套特价房全款享9折
  Predicted label: realty
  Enter a sentence: 曼联签新赞助年入8100万镑 博比-查尔顿曼苏尔买错队
  Predicted label: sports
  ```

  







