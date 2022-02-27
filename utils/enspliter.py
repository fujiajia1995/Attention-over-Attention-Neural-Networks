# -*- utf-8 -*-
import spacy
import time
from config import config


class EnSpliter(object):
    def __init__(self, model="en_core_web_lg"):
        self.nlp = spacy.load(model)

    def __call__(self, sentence=None):
        tokens = self.nlp(sentence)
        result = []
        for token in tokens:
            result.append(token.text)
        return config.split_token.join(result)


en_spliter =EnSpliter()

if __name__ == "__main__":
    sentence = "I charge it at night and skip taking the cord with me because of the good battery life."
    result = en_spliter(sentence)
    print(result)