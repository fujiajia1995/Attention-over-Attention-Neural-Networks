# -*- utf-8 -*-

import argparse

config = argparse.Namespace()

# train parameter
config.data_size = 2370
config.batch_size = 40
config.shuffle = True
config.drop_last = True
config.lr = 0.0005
config.epoch = 5000
config.save_per = 5
config.grad_clip = 0.01
config.drop = 0.5
config.loss_file = "./loss.log"


# checkpoint parameter
config.use_checkpoint = False
config.checkpoint_address = "./checkpoint/"


# dataset parameter
config.train_percent = 0.9
config.test_percent = 0.1
config.train_data_address = "./data/laptop_train2.txt"
config.test_data_address = ""
config.split_token = " "
config.unknown_token = "<uknown>"
config.positive = 0
config.negative = 1
config.conflict = 2
config.neutral = 3
config.padding_symbol = 0
config.data_split_pattern = "<text>(.*)<aspect>(.*)<polarity>(.*)<dataType>(.*)"

# embedding parameter
config.use_embedding = True
config.embedding_size = 50
config.embedding_file_adderss = "./word2vec_pretrained_model/glove.6b.50d.txt"

# model parameter
config.lstm_hidden_size = 100
config.model_output_size = 4
config.device = "cpu"




