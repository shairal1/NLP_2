
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

# load csv data
eng_df = pd.read_csv("eng_preprocessed.csv", header=None, names=["en"])
nl_df = pd.read_csv("nl_preprocessed.csv", header=None, names=["nl"])
df = pd.concat([eng_df, nl_df], axis=1)

# Train-Test-Split (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

#  Tokenizer und Vokabular 

def build_vocab(sentences, min_freq=2):
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence.split())
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def tokenize(sentence, vocab):
    return [vocab.get(word, vocab["<unk>"]) for word in sentence.split()]

# Dataset Klasse 

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, max_len=40):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src = tokenize(self.src_texts[idx], self.src_vocab)
        tgt = tokenize(self.tgt_texts[idx], self.tgt_vocab)
        src = src[:self.max_len]
        tgt = tgt[:self.max_len]
        src += [self.src_vocab["<pad>"]] * (self.max_len - len(src))
        tgt = [self.tgt_vocab["<sos>"]] + tgt + [self.tgt_vocab["<eos>"]]
        tgt += [self.tgt_vocab["<pad>"]] * (self.max_len + 2 - len(tgt))
        return torch.tensor(src), torch.tensor(tgt)

#  Modelldefinition 

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Parameter(torch.rand(hid_dim))

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy @ self.v
        return torch.softmax(energy, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(hid_dim + emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim * 2 + emb_dim, output_dim)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        a = self.attention(hidden.squeeze(0), encoder_outputs)
        a = a.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = self.fc(torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1))
        return output, hidden, a.squeeze(1)

#  main 

    def main():
        # Load and prepare data
        eng_df = pd.read_csv("eng_preprocessed.csv", header=None, names=["en"])
        nl_df = pd.read_csv("nl_preprocessed.csv", header=None, names=["nl"])
        df = pd.concat([eng_df, nl_df], axis=1)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # Build vocabularies
        src_vocab = build_vocab(train_df["en"])
        tgt_vocab = build_vocab(train_df["nl"])
        inv_tgt_vocab = {i: w for w, i in tgt_vocab.items()}

        # Create datasets and loaders
        train_dataset = TranslationDataset(train_df["en"].tolist(), train_df["nl"].tolist(), src_vocab, tgt_vocab)
        test_dataset = TranslationDataset(test_df["en"].tolist(), test_df["nl"].tolist(), src_vocab, tgt_vocab)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)

        # Initialize models
        input_dim = len(src_vocab)
        output_dim = len(tgt_vocab)
        emb_dim = 128
        hid_dim = 256

        encoder = Encoder(input_dim, emb_dim, hid_dim)
        attention = Attention(hid_dim)
        decoder = Decoder(output_dim, emb_dim, hid_dim, attention)

        # Forward pass for encoder to test dimensions
        encoder_outputs, hidden = encoder(next(iter(train_loader))[0])
        print("Encoder output shape:", encoder_outputs.shape)

        # Choose a sample from the test set
        sample_src_text = test_df["en"].iloc[0]
        sample_tgt_text = test_df["nl"].iloc[0]

        sample_src_tensor = torch.tensor([tokenize(sample_src_text, src_vocab)])
        encoder.eval()
        decoder.eval()

        # Perform greedy decoding and collect attention weights
        with torch.no_grad():
            encoder_outputs, hidden = encoder(sample_src_tensor)
            input_token = torch.tensor([tgt_vocab["<sos>"]])
            predicted_tokens = []
            attentions = []

            for _ in range(20):  # limit max output length
                output, hidden, attention = decoder(input_token, hidden, encoder_outputs)
                pred_token = output.argmax(1).item()
                predicted_tokens.append(pred_token)
                attentions.append(attention.cpu().numpy())
                input_token = torch.tensor([pred_token])
                if pred_token == tgt_vocab["<eos>"]:
                    break

        # Convert token IDs back to words
        predicted_words = [list(tgt_vocab.keys())[list(tgt_vocab.values()).index(t)]
                        for t in predicted_tokens if t not in [tgt_vocab["<eos>"], tgt_vocab["<pad>"]]]
        attention_matrix = np.stack(attentions)

        # visualation
        import pathlib, os
        plot_dir = pathlib.Path("plots")
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / "attention_sample.png"

        visualize_attention(sample_src_text, ' '.join(predicted_words), attention_matrix, plot_path)

if __name__ == "__main__":
    main()
