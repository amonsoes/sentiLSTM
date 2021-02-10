import torch
import torchtext
import argparse

from torch import nn
from matplotlib import pyplot as plt

class BatchWrapper:

      def __init__(self, dl, x_var, y_vars):
            self.dl, self.x_var, self.y_vars = dl, x_var, y_vars

      def __iter__(self):
            for batch in self.dl:
                x = getattr(batch, self.x_var)
                y = batch.label
                yield (x, y)

      def __len__(self):
            return len(self.dl)

class TSVProcessor:

    def __init__(self, path, train, dev, test, batch_size, max_vocab, min_freq):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.LABELS = torchtext.data.LabelField(sequential=False, use_vocab=False)
        self.TEXT = torchtext.data.Field(sequential=True, tokenize=lambda x: x.split(), lower=True, batch_first=True)
        fields = [('label', self.LABELS),('text', self.TEXT)]
        self.train, self.dev, self.test = torchtext.data.TabularDataset.splits(path=path,
                                                                               train=train,
                                                                               validation=dev,
                                                                               test=test,
                                                                               fields=fields,
                                                                               format='tsv')
        self.TEXT.build_vocab(self.train, max_size=max_vocab, min_freq=min_freq, vectors="glove.6B.300d")
        self.vocab = self.TEXT.vocab
        train_iter = torchtext.data.BucketIterator(self.train,
                                                   batch_size,
                                                   train=True,
                                                   shuffle=True,
                                                   sort_key=lambda x: len(x.text),
                                                   sort_within_batch=True,
                                                   device=device)
        dev_iter = torchtext.data.BucketIterator(self.dev, batch_size, sort_key=lambda x: len(x.text), device=device)
        self.train_iter = BatchWrapper(train_iter, "text", "label")
        self.dev_iter = BatchWrapper(dev_iter, "text", "label")

class SentimentLSTM(nn.Module):

    def __init__(self, data, vocab_size, embedding_dim, lstm_dim, out_dim, do_factor):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(data.vocab.vectors)
        self.dropout = nn.Dropout(do_factor)
        self.biLstm = nn.LSTM(embedding_dim, lstm_dim, 2, batch_first=True, bidirectional=True) # till: hier fehlt noch der dropout
        self.fc = nn.Linear(2*lstm_dim, out_dim)
        self.to(self.device)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        output, _ = self.biLstm(x)
        x = torch.max(output, dim=1).values.to(self.device)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train_net(model, data, epochs, learning_rate, visual):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = torch.nn.CrossEntropyLoss()
    loss_list, best_acc, best_ep = [], 0.0, 0 # till: das ist mmn nicht so gut lesbar bzw. zu viel des guten :)
    print(f"\n\ntraining using {model.device}...\n\n")
    model.train()
    for ep in range(epochs):
        running_loss = .0
        for x,y in data.train_iter:
            optimizer.zero_grad()
            output = model(x)
            loss_output = loss(output, y)
            print(loss_output)
            loss_output.backward()
            optimizer.step()
            running_loss += loss_output.item()
        epoch_loss = running_loss / len(data.train)
        print(epoch_loss)
        loss_list.append(epoch_loss)
        ep_accuracy = dev_evaluate(model, data, ep)
        if ep_accuracy > best_acc:
            best_acc = ep_accuracy
            best_ep = ep
    print(f"\n\nbest ovserved accuracy: {best_acc} at epoch: {best_ep}")
    if visual:
        plt.plot(loss_list)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title(' avg loss per epoch')
        plt.show()

def dev_evaluate(model, data, epoch):
    with torch.no_grad():
        model.eval()
        correct, all = 0, 0 # till: '= 0' reicht
        for x, y in data.dev_iter:
            output = model(x)
            results = [torch.argmax(vec).item() for vec in output]
            for y_out, y_hat in zip(results, y):
                if y_out == y_hat.item():
                    correct += 1
                all += 1
        print(f"\n\nEVALUATED ON DEVELOPMENT DATA: {correct/all} ON EPOCH: {epoch}\n\n")
    model.train()
    return correct / all

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='set path to data')
    parser.add_argument('train_name', type=str, help='specify name of training file in data dir')
    parser.add_argument('dev_name', type=str, help='specify name of dev file in data dir')
    parser.add_argument('test_name', type=str, help='specify name of test file in data dir')
    parser.add_argument('--max_vocab', type=int, default=7000, help='set max size of vocab')
    parser.add_argument('--min_freq', type=int, default=2, help='set the min count of words to be included in vocab')
    parser.add_argument('--emb_dim', type=int, default=300, help='set dimension of embedding')
    parser.add_argument('--lstm_dim', type=int, default=400, help='set dimension of lstm')
    parser.add_argument('--do_factor', type=float, default=0.5, help='set dropout rate')
    parser.add_argument('--lr', type=float, default=0.0005, help='set lr for Adam optimizer')
    parser.add_argument('--batch_size', type=int, default=8, help='set the size of the batch sent to the network')
    parser.add_argument('--epochs', type=int, default=5, help='iterations over dataset')
    parser.add_argument('--visual_train', type=lambda x: x=='True', default='False', help='iterations over dataset')
    args = parser.parse_args()

    print("\n\n###### LSTM FOR SENTIMENT_ANALYSIS ######\n\n")

    data = TSVProcessor(args.path,
                        args.train_name,
                        args.dev_name,
                        args.test_name,
                        batch_size=args.batch_size,
                        max_vocab=args.max_vocab,
                        min_freq=args.min_freq)

    net = SentimentLSTM(data, vocab_size=len(data.vocab),
                        embedding_dim=args.emb_dim,
                        lstm_dim=args.lstm_dim,
                        out_dim=5,
                        do_factor=args.do_factor)

    train_net(net, data, args.epochs, args.lr, args.visual_train)
