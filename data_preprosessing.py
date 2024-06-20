# 以下代码为小组新增代码
import pandas as pd
from sklearn.model_selection import train_test_split


# 读取parquet数据文件
all = pd.read_parquet('0000.parquet')

# 按照一定比例将数据划分为训练集、验证集和测试集
train, val_test = train_test_split(all, test_size=0.3, random_state=42)
val, test = train_test_split(val_test, test_size=0.33, random_state=42)

# 将数据保存为txt文件
train.to_csv('train.txt', header=False, index=False, columns=['text', 'label'], sep='\t')
val.to_csv('dev.txt', header=False, index=False, columns=['text', 'label'], sep='\t')
test.to_csv('test.txt', header=False, index=False, columns=['text', 'label'], sep='\t')