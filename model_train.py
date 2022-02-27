# -*- utf-8 -*-

from torch.utils.data import DataLoader
import torch
from config import config
from dataset import SaDataset
from model import Encoder
from pretrained_embedding import PreTrainEmbedding
import datetime
from tqdm import tqdm
import numpy as np


#def save_model(model, dataset, path):
##    torch.save(
#        {"model_stat": model.state_dict(),
#         "dataset": dataset
#        }, path
#    )


def generate_batch(dataset, batch_size, shuffle=True, drop_last=True):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=shuffle, drop_last=drop_last)

    for data_dict in data_loader:
        output_dict = {}
        for name, vector in data_dict.items():
            output_dict[name] = data_dict[name].to(config.device)
        yield output_dict


# dataset
dataset = SaDataset.load_from_txt(config.train_data_address)
vectorizer = dataset.get_vectorizer()

# model
if config.use_embedding:
    embedding_weight = PreTrainEmbedding.load_from_file(vectorizer.vocabulary, config.embedding_file_adderss)
    encoder = Encoder(len(vectorizer.vocabulary), config.embedding_size, embedding_weight=embedding_weight).to(config.device)
else:
    encoder = Encoder(len(vectorizer.vocabulary), config.embedding_size).to(config.device)

# loss optimizer
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(encoder.parameters(), lr=config.lr)

# check point
if config.use_checkpoint:
    checkpoint = torch.load(config.checkpoint_address)
    encoder.load_state_dict(checkpoint["model_stat"])
    optimizer.load_state_dict(checkpoint["optimizer_stat"])
    start = checkpoint["start_stat"]
    loss1 = checkpoint["loss_stat"]
else:
    start = 0
# train
old_prec = 0
for epoch_index in tqdm(range(start, config.epoch)):
    dataset.set_dataType("train")
    train_batch_generator = generate_batch(dataset, config.batch_size,
                                     shuffle=config.shuffle, drop_last=config.drop_last)

    train_loss = 0.0
    train_acc = 0.0

    encoder.train()

    for batch_index, batch_dict in enumerate(train_batch_generator):
        optimizer.zero_grad()
        y_pred = encoder(sentence_source=batch_dict["text"], target_source=batch_dict["aspect"])
        loss1 = loss_func(y_pred, batch_dict["polarity"])
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.grad_clip)
        optimizer.step()
        train_loss += loss1
    tqdm.write("epoch:"+str(epoch_index), end=" ")
    tqdm.write("train_loss:", end=" ")
    tqdm.write(str(float(train_loss.data)), end=" ")

    dataset.set_dataType("test")
    test_batch_generator = generate_batch(dataset,
                                          batch_size=config.batch_size,
                                          )
    encoder.eval()

    test_loss = 0.0
    test_acc = 0.0

    prec = 0
    for batch_index, batch_dict in enumerate(test_batch_generator):
        count = 0
        y_pred = encoder(sentence_source=batch_dict["text"], target_source=batch_dict["aspect"])
        y_pred = y_pred.tolist()
        target = batch_dict["polarity"].tolist()
        y_pred = np.argmax(y_pred, axis=1)
        for i in range(len(y_pred)):
            if int(y_pred[i]) == int(target[i]):
                count += 1
        prec += count/len(y_pred)
    log_loss = str(prec/(batch_index+1))
    tqdm.write(str(prec/(batch_index+1)))
        #print(y_pred.tolist(), batch_dict["polarity"])

    file_name = str(datetime.datetime.now())+".pt"
    #if epoch_index % config.save_per == 0:
    if old_prec < prec:
        old_prec = prec
        torch.save({
            "dataset": dataset,
            "model_stat": encoder.state_dict(),
            "optimizer_stat": optimizer.state_dict(),
            "loss_stat": loss1,
            "start_stat": epoch_index
        }, "./checkpoint/"+"bestmodel.ckpt")
        with open(config.loss_file, "a") as f:
            print(file_name+":"+log_loss, file=f)

