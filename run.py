# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import interact

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--test', default=False, type=bool, help='True for test, False for train')
args = parser.parse_args()


if __name__ == '__main__':
    # dataset = 'THUCNews'  # 清华新闻数据集
    # dataset = 'weibo_4modes' # 三十六万微博四分类数据集
    # dataset = 'weibo_senti_100k'  # 十万条微博两分类数据集
    # dataset = 'yelp_review_full'  # 七十万条Yelp五分类数据集
    dataset = 'sst_2'  # SST二分类数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    
    if args.test == False:
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样

        start_time = time.time()
        print("Loading data...")
        # 必须构造出vacab才进行模型的初始化，否则会导致嵌入层和词典大小不一样而爆内存
        vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
        model = x.Model(config).to(config.device)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

        # train
        config.n_vocab = len(vocab)
        if model_name != 'Transformer':
            init_network(model)
        print(model.parameters)
        train(config, model, train_iter, dev_iter, test_iter)

    else:
        model = x.Model(config).to(config.device)
        interact(model, config, dataset, model_name, word=args.word)