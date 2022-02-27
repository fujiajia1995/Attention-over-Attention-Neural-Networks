# -*- utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional
from torch.nn import LSTMCell
from torch.nn import Linear
from torch.nn import Embedding
from config import config


class Encoder(nn.Module):
    def __init__(self, num_embedding, embedding_size, embedding_weight=None):
        super(Encoder, self).__init__()
        self.word_embedding = Embedding(num_embeddings=num_embedding,
                                        embedding_dim=embedding_size,
                                        padding_idx=0,
                                        _weight=embedding_weight,
                                        )
        self.f_sentence_lstm_cell = LSTMCell(input_size=embedding_size,
                                             hidden_size=config.lstm_hidden_size,)
        self.b_sentence_lstm_cell = LSTMCell(input_size=embedding_size,
                                             hidden_size=config.lstm_hidden_size, )
        self.f_target_lstm_cell = LSTMCell(input_size=embedding_size,
                                           hidden_size=config.lstm_hidden_size)
        self.b_target_lstm_cell = LSTMCell(input_size=embedding_size,
                                           hidden_size=config.lstm_hidden_size, )
        self.output = Linear(in_features=config.lstm_hidden_size*2, out_features=config.model_output_size)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=config.drop)

    @staticmethod
    def _init_state(batch_size):
        return torch.zeros(batch_size, config.lstm_hidden_size).to(config.device)

    def _lstm_cell(self, embedding_source, model, reverse=False):
        source_sequence_len, source_batch_size, source_emmbedding_size = embedding_source.shape
        hidden_list = []
        hidden_state = self._init_state(source_batch_size)
        cell_state = self._init_state(source_batch_size)
        for i in range(source_sequence_len):
            if reverse:
                index = source_sequence_len-1-i
            else:
                index = i
            hidden_state, cell_state = model(embedding_source[index], (hidden_state, cell_state))
            hidden_list.append(hidden_state)
        return hidden_list

    def _bidirection_lstm_cell(self, embedding_source, f_model, b_model, batch_first=True):
        if batch_first:
            embedding_source = embedding_source.permute(1, 0, 2)
        forward_hidden_list = self._lstm_cell(embedding_source, f_model)
        backward_hidden_list = self._lstm_cell(embedding_source, b_model, reverse=True)
        backward_hidden_list.reverse()
        backward_hidden_stat = torch.stack(backward_hidden_list).permute(1, 0, 2).to(config.device)
        forward_hidden_stat = torch.stack(forward_hidden_list).permute(1, 0, 2).to(config.device)
        return torch.cat((forward_hidden_stat, backward_hidden_stat), 2)

    def forward(self, sentence_source, target_source):
        sentence_vector = self.dropout(self.word_embedding(sentence_source).to(config.device))
        target_vector = self.word_embedding(target_source).to(config.device)
        sentence_hidden_stat = self._bidirection_lstm_cell(sentence_vector, self.f_sentence_lstm_cell,
                                                           self.b_sentence_lstm_cell)
        target_hidden_stat = self._bidirection_lstm_cell(target_vector, self.f_target_lstm_cell,
                                                         self.b_target_lstm_cell)
        attention_matrix = torch.matmul(sentence_hidden_stat, torch.transpose(target_hidden_stat, 1, 2)).to(config.device)
        col_attention_matrix = torch.softmax(attention_matrix, dim=1).to(config.device)
        row_attention_matrix = torch.softmax(attention_matrix, dim=2).to(config.device)
        row_attention_vector = torch.mean(row_attention_matrix, keepdim=True, dim=1).to(config.device)
        attention_result = torch.matmul(col_attention_matrix, torch.transpose(row_attention_vector, 1, 2)).to(config.device)
        score_result = torch.matmul(torch.transpose(attention_result, 1, 2), sentence_hidden_stat).to(config.device)
        classification_score = self.softmax(self.output(score_result).to(config.device))
        #result = torch.softmax(classification_score, dim=-1).squeeze()
        #print(result.size(), result)
        return classification_score.squeeze()


if __name__ == "__main__":
    encoder = Encoder(50, 100)
    sentence1 = torch.tensor([2, 4, 5, 1, 3])
    sentence2 = torch.tensor([2, 3, 5, 6, 1])
    target1 = torch.tensor([2, 3])
    target2 = torch.tensor([2, 4])
    sentence_batch = torch.stack([sentence1, sentence2])
    target_batch = torch.stack([target1, target2])
    result = encoder(sentence_batch, target_batch)
    print(result, result.size())
    for k, v in encoder.named_parameters():
        print(k, v.size())
