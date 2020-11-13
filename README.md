# Neural Machine Translation - English to German
Pytorch implementation of English to German Neural machine translation model.

It is a basic Encoder-Decoder RNN with LSTMs implemented on pytorch. Heavily inspired from [this tutorial.](https://github.com/bentrevett/pytorch-seq2seq)

Please check other branches of this repo as well. I have implemented several improved versions of this model.
Feel free to fork this repo and extend this solution. 

Basic Encoder-Decoder RNN (LSTM) results - Testing loss: 2.855 | Testing perplexity:  17.378

Encoder-Decoder RNN with context reuse results - Testing loss: 2.656 | Testing ppl:  14.235

Encoder-Decoder GRU with attention results - Testing loss: 2.635 | Testing ppl:  13.948