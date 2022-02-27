import pandas as pd
import re
from config import config


def load_file(file_address):
    pattern = re.compile(config.data_split_pattern)
    text = []
    aspects = []
    polarity = []
    dataType = []
    with open(file_address, "r") as f:
        for i in f.readlines():
            groups = pattern.match(i)
            text.append(groups[1])
            aspects.append(groups[2])
            polarity.append(groups[3])
            dataType.append(groups[4])
    laptop_review = pd.DataFrame(
        data={
            "text": text,
            "aspect": aspects,
            "polarity": polarity,
            "dataType": dataType
        },
        columns=["text", "aspect", "polarity", "dataType"])
    return laptop_review


if __name__ == "__main__":
    temp = load_file("../data/laptop_train2.txt")
    print(temp)
    print(temp.iloc[2300].text)
    print(temp.iloc[2300].aspect)
    print(temp.iloc[2300].polarity)
    print(temp.iloc[2300].dataType)