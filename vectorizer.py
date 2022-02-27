# -*- utf-8 -*-

from config import config
from vocabulary import Vocabulary
import torch
from utils import file_to_pandas
import numpy as np


class ReviewVectorizer(object):
    def __init__(self, vocabulary, sentence_len):
        self.vocabulary = vocabulary
        self.sentence_len = sentence_len

    @classmethod
    def from_dataframe(cls, load_dataframe):
        vocabulary = Vocabulary()
        sentence_len = 0
        dataframe = load_dataframe
        for text in dataframe["text"]:
            token_list = text.split(config.split_token)
            if len(token_list) > sentence_len:
                sentence_len = len(token_list)
            for token in text.split(config.split_token):
                vocabulary.add_token(token)
        return cls(vocabulary, sentence_len)

    def vectorize(self, text):
        result = np.zeros(self.sentence_len,dtype=np.int32)
        for index, token in enumerate(text.split(config.split_token)):
            result[index] = self.vocabulary.lookup_token(token)
        return torch.tensor(result).to(config.device)


if __name__ == "__main__":
    dataframe = file_to_pandas.load_file(config.train_data_address)
    temp_vectorize = ReviewVectorizer.from_dataframe(dataframe)
    temp_indices = temp_vectorize.vectorize("When I called Sony the Customer Service was Great .")
    print(temp_indices)
    for i in temp_indices.data:
        print(temp_vectorize.vocabulary.lookup_index(int(i)), end=" ")
