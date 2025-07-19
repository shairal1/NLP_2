#%%
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu
import random
import time
import platform
import logging

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Print device info
logger.info(f'PyTorch version: {torch.__version__}')
logger.info(f'Platform: {platform.platform()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
logger.info(f'Using device: {device}')
if device.type == 'mps':
    logger.info('Apple Silicon (M1/M2/M3/M4) Metal backend detected.')

# 1. Load preprocessed data
eng_sample = pd.read_csv('../sampled_data/eng_sampled.csv')
nl_sample = pd.read_csv('../sampled_data/nl_sampled.csv')

# 2. Tokenization (word-level)
def tokenize(sentences):
    return [str(s).split() for s in sentences]

eng_tokens = tokenize(eng_sample['sentence'])
nl_tokens = [['<SOS>'] + str(s).split() + ['<EOS>'] for s in nl_sample['sentence']]

# 3. Build vocabularies
def build_vocab(token_lists, min_freq=2):
    counter = Counter(token for sent in token_lists for token in sent)
    vocab = {'<PAD>':0, '<SOS>':1, '<EOS>':2, '<UNK>':3}
    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab

eng_vocab = build_vocab(eng_tokens)
nl_vocab = build_vocab(nl_tokens)

def encode(tokens, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

eng_indices = [encode(sent, eng_vocab) for sent in eng_tokens]
nl_indices = [encode(sent, nl_vocab) for sent in nl_tokens]

# 4. Padding
def pad_sequences(sequences, max_len, pad_value=0):
    return [seq + [pad_value]*(max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in sequences]

max_len_eng = min(50, max(len(seq) for seq in eng_indices))
max_len_nl = min(50, max(len(seq) for seq in nl_indices))

eng_padded = pad_sequences(eng_indices, max_len_eng)
nl_padded = pad_sequences(nl_indices, max_len_nl)

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(eng_padded, nl_padded, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# 6. PyTorch Dataset and DataLoader
class TranslationDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = torch.tensor(src, dtype=torch.long)
        self.tgt = torch.tensor(tgt, dtype=torch.long)
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

train_ds = TranslationDataset(X_train, y_train)
val_ds = TranslationDataset(X_val, y_val)
test_ds = TranslationDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

# 7. Encoder-Decoder Model
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, batch_first=True)
    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, input, hidden):
        embedded = self.embedding(input.unsqueeze(1))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)
        hidden = self.encoder(src)
        input = tgt[:,0]
        for t in range(1, tgt_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:,t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:,t] if teacher_force else top1
        return outputs

# 8. Training and Evaluation
encoder = Encoder(len(eng_vocab), emb_dim=256, hidden_dim=512).to(device)
decoder = Decoder(len(nl_vocab), emb_dim=256, hidden_dim=512).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=0)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

def train(model, loader):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for batch_idx, (src, tgt) in enumerate(loader):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt)
        output_dim = output.shape[-1]
        output_flat = output[:,1:].reshape(-1, output_dim)
        tgt_flat = tgt[:,1:].reshape(-1)
        loss = criterion(output_flat, tgt_flat)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # Accuracy: compare predicted tokens to target tokens (excluding PAD)
        preds = output_flat.argmax(1)
        mask = tgt_flat != 0
        correct += (preds[mask] == tgt_flat[mask]).sum().item()
        total += mask.sum().item()
        # Log batch loss every 100 batches
        if (batch_idx + 1) % 100 == 0:
            logger.info(f'Batch {batch_idx+1}/{len(loader)}: Loss={loss.item():.4f}')
    accuracy = correct / total if total > 0 else 0
    return epoch_loss / len(loader), accuracy

def evaluate(model, loader):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            output_flat = output[:,1:].reshape(-1, output_dim)
            tgt_flat = tgt[:,1:].reshape(-1)
            loss = criterion(output_flat, tgt_flat)
            epoch_loss += loss.item()
            preds = output_flat.argmax(1)
            mask = tgt_flat != 0
            correct += (preds[mask] == tgt_flat[mask]).sum().item()
            total += mask.sum().item()
    accuracy = correct / total if total > 0 else 0
    return epoch_loss / len(loader), accuracy

# Training loop (with timing and accuracy)
for epoch in range(10):
    start_time = time.time()
    train_loss, train_acc = train(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)
    elapsed = time.time() - start_time
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    logger.info(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Time={elapsed:.1f}s')

# Plot training and validation loss/accuracy
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('loss_curve.png')
plt.close()

plt.figure(figsize=(10,5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('accuracy_curve.png')
plt.close()

# 9. Evaluation Metrics (BLEU)
def compute_bleu(model, loader, tgt_vocab):
    model.eval()
    references = []
    hypotheses = []
    inv_tgt_vocab = {v:k for k,v in tgt_vocab.items()}
    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            outputs = model(src, tgt, teacher_forcing_ratio=0)
            preds = outputs.argmax(-1).cpu().numpy()
            for i in range(preds.shape[0]):
                ref = [[inv_tgt_vocab[idx] for idx in tgt[i].cpu().numpy() if idx not in [0,1,2,3]]]
                hyp = [inv_tgt_vocab[idx] for idx in preds[i] if idx not in [0,1,2,3]]
                references.append(ref)
                hypotheses.append(hyp)
    bleu = corpus_bleu(references, hypotheses)
    return bleu

bleu_score = compute_bleu(model, test_loader, nl_vocab)
logger.info(f'Test BLEU score: {bleu_score:.4f}')

# Test accuracy
_, test_acc = evaluate(model, test_loader)
logger.info(f'Test Accuracy: {test_acc:.4f}')
