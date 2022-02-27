from xml.etree import ElementTree as ET
from config import config
from enspliter import en_spliter
import numpy as np

data_address = "../rawdata/Laptops_Train.xml"
result_address = "../data/laptop_train2.txt"

random_list = np.random.permutation(config.data_size)
train_length = int(config.data_size*config.train_percent)
train_indexes = random_list[:train_length]
test_indexes = random_list[train_length:]

tree = ET.parse(data_address)
root = tree.getroot()
"""
config.positive = 0
config.negative = 1
config.conflict = 2
config.neutral = 3
"""
with open(result_address, "w") as f:
    index = 0
    for data in root:
        text = data.find("text").text
        aspectterms = data.find("aspectTerms")
        if aspectterms:
            for aspect in aspectterms:
                if aspect.attrib["polarity"] == "positive": polarity = config.positive
                elif aspect.attrib["polarity"] == "negative": polarity = config.negative
                elif aspect.attrib["polarity"] == "conflict": polarity = config.conflict
                elif aspect.attrib["polarity"] == "neutral": polarity = config.neutral
                text = en_spliter(text)
                aspect = en_spliter(aspect.attrib["term"])
                if index in train_indexes:
                    dataType = "train"
                elif index in test_indexes:
                    dataType = "test"
                else:
                    print("data size:" + str(index))
                    exit(0)
                print("<text>"+text+"<aspect>"+aspect+"<polarity>"+str(polarity)+"<dataType>"+dataType, file=f)
                index += 1

