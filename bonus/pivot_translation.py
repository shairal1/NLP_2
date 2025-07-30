import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random
import pathlib


def tokenize(sent):
    return str(sent).split()

def build_vocab(token_lists, min_freq=2):
    counter = Counter(token for sent in token_lists for token in sent)
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def encode(tokens, vocab):
    return [vocab.get(tok, vocab['<UNK>']) for tok in tokens]

def pad(seq, max_len, pad_idx):
    return seq + [pad_idx] * (max_len - len(seq))

# translation

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=30):
        self.src_data = []
        self.tgt_data = []
        for src, tgt in zip(src_sentences, tgt_sentences):
            src_tok = ['<SOS>'] + tokenize(src)[:max_len - 2] + ['<EOS>']
            tgt_tok = ['<SOS>'] + tokenize(tgt)[:max_len - 2] + ['<EOS>']
            src_enc = encode(src_tok, src_vocab)
            tgt_enc = encode(tgt_tok, tgt_vocab)
            self.src_data.append(pad(src_enc, max_len, src_vocab['<PAD>']))
            self.tgt_data.append(pad(tgt_enc, max_len, tgt_vocab['<PAD>']))

    def __len__(self):
        return len(self.src_data)



    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx]), torch.tensor(self.tgt_data[idx])


# Seq2Seq model

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        _, hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input.unsqueeze(1))
        output, hidden = self.rnn(embedded, hidden)
        return self.fc(output.squeeze(1)), hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, sos_idx, eos_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)

        hidden = self.encoder(src)
        input = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            input = tgt[:, t] if teacher_force else output.argmax(1)

        return outputs

# training

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt)
        output_dim = output.shape[-1]
        loss = criterion(output[:, 1:].reshape(-1, output_dim), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# pivot

def train_model_pair():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    MAX_LEN = 30
    EPOCHS = 10
    ROOT = pathlib.Path(__file__).parent.parent / "data" 

    # 1st training NL → EN
    with open(ROOT / "nl-en" / "europarl-v7.nl-en.nl", encoding="utf-8") as f:
        nl_sentences = f.read().splitlines()
    with open(ROOT / "nl-en" / "europarl-v7.nl-en.en", encoding="utf-8") as f:
        en_sentences = f.read().splitlines()

    frac = 0.05
    indices = random.sample(range(len(nl_sentences)), int(frac * len(nl_sentences)))
    nl_sample = [nl_sentences[i] for i in indices]
    en_sample = [en_sentences[i] for i in indices]

    nl_vocab = build_vocab([['<SOS>'] + tokenize(s) + ['<EOS>'] for s in nl_sample])
    en_vocab = build_vocab([['<SOS>'] + tokenize(s) + ['<EOS>'] for s in en_sample])

    train_ds = TranslationDataset(nl_sample, en_sample, nl_vocab, en_vocab, MAX_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model1 = Seq2Seq(
        Encoder(len(nl_vocab), 128, 256),
        Decoder(len(en_vocab), 128, 256),
        en_vocab['<SOS>'], en_vocab['<EOS>'], device
    ).to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
    criterion1 = nn.CrossEntropyLoss(ignore_index=en_vocab['<PAD>'])

    print("\n Training: NL → EN")
    for epoch in range(EPOCHS):
        loss = train(model1, train_dl, optimizer1, criterion1, device)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
    torch.save(model1.state_dict(), "nl2en_model.pt")
    print("succesful: nl2en_model.pt")

    # 2nd training EN → SV
    with open(ROOT / "sv-en" / "europarl-v7.sv-en.en", encoding="utf-8") as f:
        en_sv = f.read().splitlines()
    with open(ROOT / "sv-en" / "europarl-v7.sv-en.sv", encoding="utf-8") as f:
        sv_sentences = f.read().splitlines()

    indices2 = random.sample(range(len(en_sv)), int(frac * len(en_sv)))
    en_sample2 = [en_sv[i] for i in indices2]
    sv_sample = [sv_sentences[i] for i in indices2]

    en_sv_vocab = build_vocab([['<SOS>'] + tokenize(s) + ['<EOS>'] for s in en_sample2])
    sv_vocab = build_vocab([['<SOS>'] + tokenize(s) + ['<EOS>'] for s in sv_sample])

    train_ds2 = TranslationDataset(en_sample2, sv_sample, en_sv_vocab, sv_vocab, MAX_LEN)
    train_dl2 = DataLoader(train_ds2, batch_size=BATCH_SIZE, shuffle=True)

    model2 = Seq2Seq(
        Encoder(len(en_sv_vocab), 128, 256),
        Decoder(len(sv_vocab), 128, 256),
        sv_vocab['<SOS>'], sv_vocab['<EOS>'], device
    ).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
    criterion2 = nn.CrossEntropyLoss(ignore_index=sv_vocab['<PAD>'])

    print("\n Training: EN → SV")
    for epoch in range(EPOCHS):
        loss = train(model2, train_dl2, optimizer2, criterion2, device)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
    torch.save(model2.state_dict(), "en2sv_model.pt")
    print("succesful: en2sv_model.pt")

if __name__ == "__main__":
    train_model_pair()
