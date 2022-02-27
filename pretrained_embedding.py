# -*- utf-8 -*-
from vocabulary import Vocabulary
from config import config
import collections
import tqdm
import torch


class PreTrainEmbedding(object):
    def __init__(self):
        pass

    @classmethod
    def load_from_file(cls, vocabulary, file_address):
        tensor_list = []
        embedding_dict = collections.defaultdict(list)
        with open(file_address, "r") as f:
            for text in tqdm.tqdm(f.readlines()):
                data = text.split(" ")
                name = data[0]
                vector = data[1:]
                for i in vector:
                    embedding_dict[name].append(float(i))
        for i in range(len(vocabulary)):
            if vocabulary.lookup_index(i) not in embedding_dict.keys():
                temp = torch.tensor([float(0) for s in range(config.embedding_size)])
            else:
                temp = torch.tensor(embedding_dict[vocabulary.lookup_index(i)])
            tensor_list.append(temp)
        return torch.stack(tensor_list)


if __name__ == "__main__":
    vocabulary = Vocabulary()
    vocabulary.add_token("the")
    vocabulary.add_token("of")
    vocabulary.add_token("a")
    vocabulary.add_token("that")
    vocabulary.add_token("for")
    temp = PreTrainEmbedding.load_from_file(vocabulary, "./word2vec_pretrained_model/glove.6b.50d.txt")
    print(temp.size())