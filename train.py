from model import *
from data import *
import time
import math
import matplotlib.pyplot as plt


# Setup the model and train
device = "cuda:0" if torch.cuda.is_available() else "cpu"
ENC_HIDDEN_SIZE = 512
DEC_HIDDEN_SIZE = 512
EMBEDDING_SIZE = 256
BIDIRECTIONAL = True
EPOCHS = 10
NUM_LAYERS = 2
DROPOUT = 0.3
teacher_forcing_ratio = 0.4

(train_data, valid_data, test_data), src, tgt = prepare_data()
SRC_PAD_INDEX = src.vocab.stoi[src.pad_token]

encoder = EncoderRNN(EMBEDDING_SIZE, len(src.vocab), ENC_HIDDEN_SIZE, DROPOUT, BIDIRECTIONAL)
attention = Attention(ENC_HIDDEN_SIZE, DEC_HIDDEN_SIZE, BIDIRECTIONAL)
decoder = DecoderRNN(EMBEDDING_SIZE, len(tgt.vocab), ENC_HIDDEN_SIZE, DEC_HIDDEN_SIZE, len(tgt.vocab), attention,
                     DROPOUT,
                     BIDIRECTIONAL)
model = Seq2Seq(encoder, decoder, device, SRC_PAD_INDEX).to(device)
model.apply(init_weights)
model_optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt.vocab.stoi[tgt.pad_token])


def train(batch, clip):
    model_optimizer.zero_grad()
    src, src_len = batch.src
    tgt = batch.trg

    output, attn_weights = model(src, src_len, tgt, teacher_forcing_ratio)

    output_dim = output.shape[-1]
    output = output[1:].view(-1, output_dim)
    tgt = tgt[1:].view(-1)
    loss = criterion(output, tgt)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    model_optimizer.step()
    return loss.item() / output_dim


def train_iters():
    start = time.time()
    plot_loss = []
    print_loss_total = 0
    plot_loss_total = 0
    print_every = 500
    plot_every = 100
    clip = 1
    best_valid_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        train_loss = 0
        model.train()
        for i, batch in enumerate(train_data):
            loss = train(batch, clip)
            train_loss += loss
            print_loss_total += loss
            plot_loss_total += loss

            if i % plot_every == 0 and i != 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_loss_total = 0
                plot_loss.append(plot_loss_avg)
        valid_loss = evaluate(model, valid_data)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model.pt')
        print(f"Epoch: {epoch} | Time: {time_since(start, epoch / EPOCHS)}")
        print(f"\tTrain loss: {train_loss:.3f}      | Train ppl: {math.exp(train_loss):7.3f}")
        print(f"\tValidation loss: {valid_loss:.3f} | Validation ppl: {math.exp(valid_loss):7.3f}")
    return plot_loss


def evaluate(model, data):
    model.eval()
    loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data):
            src, src_len = batch.src
            tgt = batch.trg
            output, attn_weights = model(src, src_len, tgt)
            output_dim = output.size(-1)
            output = output[1:].view(-1, output_dim)
            tgt = tgt[1:].view(-1)
            loss += criterion(output, tgt).item()
    return loss / len(data)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s:.4f}s"


def time_since(start_time, fraction):
    now = time.time()
    s = now - start_time
    rs = s / fraction - s
    return f"{as_minutes(s)} (- {as_minutes(rs)})"


if __name__ == "__main__":
    losses = train_iters()
    plt.plot(losses)
    plt.show()
    print("\ntesting...")
    model.load_state_dict(torch.load('model.pt'))
    test_loss = evaluate(model, test_data)
    print(f"\nTesting loss: {test_loss:.3f} | Testing ppl: {math.exp(test_loss):7.3f}")
