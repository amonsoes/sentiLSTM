# Sentiment Analysis with LSTM

### Positional Args

```
python3 ./data/ sentiment.train.tsv sentiment.test.dev.tsv sentiment.test.tsv
```

- path to data folder
- training file
- development file
- test file

### Optional Args

- --max_vocab : set max size of vocab. Default: 7000
- --min_freq :  set the min count of words to be included in vocab. Default: 2
- --emd_dim : set dimension of embedding. Default: 300
- --lstm_dim : set dimension of lstm. Default: 400
- --do_factor : set dropout rate. Default: 0.5
- --lr : set lr for Adam optimizer.  Default: 0.0005
- --batch_size : set batch_size. Default: 8
- --epochs : set nr of epochs. Default: 3
- --visual_train : set if you want the losses over time to be plotted. Value needs to be True if wanted. Default:False