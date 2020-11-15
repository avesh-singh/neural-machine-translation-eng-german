# Neural Machine Translation - English to German
Pytorch implementation of English to German Neural machine translation model.

It is a basic Encoder-Decoder RNN with LSTMs implemented on pytorch. Heavily inspired from [this tutorial.](https://github.com/bentrevett/pytorch-seq2seq)

Please check other branches of this repo as well. I have implemented several improved versions of this model.

Basic Encoder-Decoder RNN (LSTM) results - Training time: 5m 44.22s (15 epochs) | Testing loss: 2.921 | Testing ppl:  18.554

Encoder-Decoder GRU with context reuse results - Training time: 3m 33.66s (10 epochs) | Testing loss: 2.654 | Testing ppl:  14.209

Encoder-Decoder GRU with attention results - Testing loss: 2.635 | Testing ppl:  13.948