import torch
import torch.nn as nn
import random


class EncoderRNN(nn.Module):
    def __init__(self, embedding_size, input_size, hidden_size, dropout=0.1, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=bidirectional)
        self.directions = 2 if bidirectional else 1
        self.output = nn.Linear(self.directions * hidden_size, hidden_size)

    def forward(self, input_word):
        embedding = self.dropout(self.embedding(input_word))
        output, hidden = self.gru(embedding)
        context = torch.tanh(self.output(torch.cat((hidden[0], hidden[1]), 1)))
        return output, context.unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size, bidirectional_encoder=False):
        super(Attention, self).__init__()
        self.enc_directions = 2 if bidirectional_encoder else 1
        self.attn = nn.Linear(self.enc_directions * enc_hidden_size + dec_hidden_size, dec_hidden_size)
        self.v = nn.Linear(dec_hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden: torch.Tensor, encoder_states: torch.Tensor):
        # encoder_states: [src_len, batch_size, 2 * enc_hidden_size]
        # hidden: [1, batch_size, dec_hidden_size]
        batch_size = encoder_states.shape[1]
        src_len = encoder_states.shape[0]
        hidden = hidden.permute(1, 0, 2).repeat(1, src_len, 1)
        # hidden -> [batch_size, 1, dec_hidden_size] -> [batch_size * 1, 1 * src_len, dec_hidden_size * 1]

        encoder_states = encoder_states.permute(1, 0, 2)
        # encoder_states -> [batch_size, src_len, 2 * enc_hidden_state]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_states), 2)))
        # energy -> [batch_size, src_len, dec_hidden_size]
        weights = self.v(energy)
        # weights: [batch_size, src_len]
        return self.softmax(weights).view(batch_size, 1, src_len)


class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, input_size, enc_hidden_size, dec_hidden_size, output_size, attention,
                 dropout=0.1, bidirectional_encoder=False):
        super(DecoderRNN, self).__init__()
        self.hidden_size = dec_hidden_size
        self.output_size = output_size
        self.attention = attention
        self.directions = 2 if bidirectional_encoder else 1
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size + self.directions * enc_hidden_size, dec_hidden_size)
        self.output = nn.Linear((self.directions * enc_hidden_size) + dec_hidden_size + embedding_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_word, hidden, encoder_states):
        # embedding: [batch_size, embedding_size]
        embedding = self.dropout(self.embedding(input_word))
        attention = self.attention(hidden, encoder_states)
        # attention: [batch_size, 1, src_len]
        encoder_states = encoder_states.permute(1, 0, 2)
        # encoder_states: [batch_size, src_len, directions * enc_hidden_size]
        context = torch.bmm(attention, encoder_states).permute(1, 0, 2)
        # context: [1, batch_size, directions * enc_hidden_state]
        rnn_input = torch.cat((embedding, context), dim=2)
        # rnn_inputs: [1, batch_size, directions * enc_hidden_size + embedding_size]
        output, hidden = self.gru(rnn_input, hidden)
        # hidden: [1, batch_size, dec_hidden_size]
        linear_input = torch.cat((output, context, embedding), dim=2)
        output = self.output(linear_input.squeeze(0))
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
        encoder_states, hidden = self.encoder(input)
        dec_input = target[0, :]
        for i in range(seq_len):
            dec_input = dec_input.unsqueeze(0)
            dec_output, hidden = self.decoder(dec_input, hidden, encoder_states)
            outputs[i] = dec_output
            if random.random() < teacher_forcing_ratio:
                dec_input = target[i]
            else:
                dec_input = dec_output.argmax(-1)
        return outputs


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
