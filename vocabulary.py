# -*- utf-8 -*-
from config import config


class Vocabulary(object):
    def __init__(self, token_to_idx=None, add_unknown=True):
        if token_to_idx is None:
            token_to_idx = {}
        self._add_unknown = add_unknown
        self.unkonwn_token = config.unknown_token
        self._token_to_idx = token_to_idx
        self._id_to_token = {index: token for token, index in self._token_to_idx.items()}
        self.padding_index = self.add_token(config.padding_symbol)
        if add_unknown:
            self.unknown_index = self.add_token(config.unknown_token)

    def add_token(self, token):
        if token in self._token_to_idx.keys():
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._id_to_token[index] = token
        return index

    def lookup_token(self, token):
        if self._add_unknown:
            return self._token_to_idx.get(token, self.unknown_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        if index not in self._id_to_token.keys():
            raise KeyError("cant find index")
        return self._id_to_token[index]

    def __str__(self):
        return "<vocabulary database> len:" + str(len(self))

    def __len__(self):
        return len(self._token_to_idx.keys())
