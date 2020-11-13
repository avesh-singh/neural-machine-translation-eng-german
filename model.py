import torch
import torch.nn as nn
import random


class EncoderRNN(nn.Module):
    def __init__(self, embedding_size, input_size, hidden_size, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size)

    def forward(self, input_word):
        embedding = self.dropout(self.embedding(input_word))
        output, hidden = self.gru(embedding)
        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, input_size, hidden_size, output_size, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size + hidden_size, hidden_size)
        self.output = nn.Linear(2 * hidden_size + embedding_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_word, hidden, context):
        embedding = self.dropout(self.embedding(input_word))
        rnn_input = torch.cat((embedding, context), dim=2)
        output, hidden = self.gru(rnn_input, hidden)
        linear_input = torch.cat((hidden, context, embedding), dim=2)
        output = self.output(linear_input)
        output = self.softmax(output)
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hidden_size == decoder.hidden_size, "encoder and decoder must have equal units in hidden layer"

    def forward(self, input, target, teacher_forcing_ratio=0.5):
        batch_size = target.shape[1]
        seq_len = target.shape[0]
        output_size = self.decoder.output_size

        outputs = torch.zeros(seq_len, batch_size, output_size, device=self.device)
        hidden = self.encoder(input)
        context = hidden
        dec_input = target[0, :].unsqueeze(0)
        for i in range(seq_len):
            dec_output, hidden = self.decoder(dec_input, hidden, context)
            outputs[i] = dec_output
            if random.random() < teacher_forcing_ratio:
                dec_input = target[i].unsqueeze(0)
            else:
                dec_input = dec_output.argmax(2)
        return outputs


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
