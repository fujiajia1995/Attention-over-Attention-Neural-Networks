# -*- utf-8 -*-
from utils import file_to_pandas
from config import config
from vectorizer import ReviewVectorizer
import torch


class SaDataset(object):
    def __init__(self, dataframe,vectorize):
        self.dataframe = dataframe
        self._vectorize = vectorize

        self.train_df = self.dataframe[self.dataframe.dataType == "train"]
        self.train_len = len(self.train_df)
        self.test_df = self.dataframe[self.dataframe.dataType == "test"]
        self.test_len = len(self.test_df)

        self._lookup_dict = {
            "train": (self.train_df, self.train_len),
            "test": (self.test_df, self.test_len)
        }
        self.set_dataType("train")

    def get_vectorizer(self):
        return self._vectorize

    @classmethod
    def load_from_txt(cls, file):
        Nowdataframe = file_to_pandas.load_file(file)
        return cls(Nowdataframe, ReviewVectorizer.from_dataframe(Nowdataframe))

    def set_dataType(self, text="train"):
        self.target = text
        self._target_df, self._target_len = self._lookup_dict[text]

    def __len__(self):
        return self._target_len

    def __getitem__(self, index):
        data = self._target_df.iloc[index]
        text_indices = self._vectorize.vectorize(data.text)
        aspect_indices = self._vectorize.vectorize(data.aspect)
        polarity = data["polarity"]
        return {
            "text": text_indices,
            "aspect": aspect_indices,
            "polarity": torch.tensor(int(polarity)).to(config.device)
        }


if __name__ == "__main__":
    tempdataset = SaDataset.load_from_txt(config.train_data_address)
    tempdataset.set_dataType("test")
    print(tempdataset[0])
    vectorizer = tempdataset.get_vectorizer()
    for i in tempdataset[0]["aspect"]:
        print(vectorizer.vocabulary.lookup_index(int(i)), end=" ")
    for i in tempdataset[0]["text"]:
        print(vectorizer.vocabulary.lookup_index(int(i)), end=" ")
