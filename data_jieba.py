# 以下代码用于对数据集中的文本进行分词处理，为小组新增代码
import jieba
import os

def process_file(file_path):
    processed_lines = []  # 用于存储处理后的行
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentence, label = line.strip().split('\t')  # 分割句子和标签
            words = jieba.cut(sentence)  # 分词
            processed_sentence = ' '.join(words)  # 用空格连接分词结果
            processed_line = f'{processed_sentence}\t{label}'  # 重新组合成一行
            processed_lines.append(processed_line)
    
    # 将处理后的内容写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in processed_lines:
            file.write(line + '\n')

# 文件列表
files = ['dev.txt', 'train.txt', 'test.txt']
for file_name in files:
    process_file(file_name)  # 对每个文件应用处理函数