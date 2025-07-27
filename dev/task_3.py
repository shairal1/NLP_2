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
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
import os
import mlflow

# === Device Selection Option ===
# Set this to 'auto', 'cuda', 'mps', or 'cpu' to force device selection
DEVICE_OPTION = 'auto'  # 'auto', 'cuda', 'mps', or 'cpu'

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Device selection logic
if DEVICE_OPTION == 'cuda' and torch.cuda.is_available():
    device = torch.device('cuda')
    logger.info('Using CUDA (GPU)')
elif DEVICE_OPTION == 'mps' and torch.backends.mps.is_available():
    device = torch.device('mps')
    logger.info('Using Apple Silicon MPS (M1/M2/M3/M4)')
elif DEVICE_OPTION == 'cpu':
    device = torch.device('cpu')
    logger.info('Using CPU')
else:
    # Auto mode: prefer CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info('Auto-selected CUDA (GPU)')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info('Auto-selected Apple Silicon MPS (M1/M2/M3/M4)')
    else:
        device = torch.device('cpu')
        logger.info('Auto-selected CPU')
logger.info(f'Final device: {device}')

# Checkpoint directory
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 1. Load preprocessed data
eng_sample = pd.read_csv('/Users/shairal/Documents/Private/NLP_2/preprocessed_data/eng_preprocessed.csv')
nl_sample = pd.read_csv('/Users/shairal/Documents/Private/NLP_2/preprocessed_data/nl_preprocessed.csv')

sample_indices = eng_sample.sample(frac=0.3 , random_state=42).index

eng_sample = eng_sample.loc[sample_indices].reset_index(drop=True)
nl_sample = nl_sample.loc[sample_indices].reset_index(drop=True)
# 2. Tokenization
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

eng_vocab = build_vocab(eng_tokens, min_freq=2)
nl_vocab = build_vocab(nl_tokens, min_freq=2)

def encode(tokens, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

eng_indices = [encode(sent, eng_vocab) for sent in eng_tokens]
nl_indices = [encode(sent, nl_vocab) for sent in nl_tokens]

def pad_sequences(sequences, max_len, pad_value=0):
    return [seq + [pad_value]*(max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in sequences]

max_len_eng = min(50, max(len(seq) for seq in eng_indices))
max_len_nl = min(50, max(len(seq) for seq in nl_indices))

eng_padded = pad_sequences(eng_indices, max_len_eng)
nl_padded = pad_sequences(nl_indices, max_len_nl)

# Compute lengths for packing
eng_lengths = [min(len(seq), max_len_eng) for seq in eng_indices]

# 5. Split data
X_train, X_test, y_train, y_test, len_train, len_test = train_test_split(
    eng_padded, nl_padded, eng_lengths, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val, len_train, len_val = train_test_split(
    X_train, y_train, len_train, test_size=0.1, random_state=42)
#%%
#%%
# 6. PyTorch Dataset and DataLoader
class TranslationDataset(Dataset):
    def __init__(self, src, tgt, src_lengths):
        self.src = torch.tensor(src, dtype=torch.long)
        self.tgt = torch.tensor(tgt, dtype=torch.long)
        self.src_lengths = torch.tensor(src_lengths, dtype=torch.long)
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.src_lengths[idx]

train_ds = TranslationDataset(X_train, y_train, len_train)
val_ds = TranslationDataset(X_val, y_val, len_val)
test_ds = TranslationDataset(X_test, y_test, len_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

# 7. Encoder, Decoder, Seq2Seq with packed sequences
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            emb_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        packed_embedded = pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            emb_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, input, hidden):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)
        hidden = self.encoder(src, src_lengths)
        input = tgt[:, 0]
        for t in range(1, tgt_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1
        return outputs

# 8. Training and Evaluation


encoder = Encoder(len(eng_vocab), emb_dim=128, hidden_dim=256).to(device)
decoder = Decoder(len(nl_vocab), emb_dim=128, hidden_dim=256).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=0)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_val_loss = float('inf')

mlflow.set_experiment("nmt_seq2seq")
with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("embedding_dim", 128)
    mlflow.log_param("hidden_dim", 256)
    mlflow.log_param("batch_size", 64)
    mlflow.log_param("num_epochs", 10)
    mlflow.log_param("dropout", 0.2)
    mlflow.log_param("vocab_size_en", len(eng_vocab))
    mlflow.log_param("vocab_size_nl", len(nl_vocab))

    def train(model, loader):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for batch_idx, (src, tgt, src_lengths) in enumerate(tqdm(loader, desc="Training")):
            src, tgt, src_lengths = src.to(device), tgt.to(device), src_lengths.to(device)
            optimizer.zero_grad()
            output = model(src, src_lengths, tgt)
            output_dim = output.shape[-1]
            output_flat = output[:,1:].reshape(-1, output_dim)
            tgt_flat = tgt[:,1:].reshape(-1)
            loss = criterion(output_flat, tgt_flat)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds = output_flat.argmax(1)
            mask = tgt_flat != 0
            correct += (preds[mask] == tgt_flat[mask]).sum().item()
            total += mask.sum().item()
        accuracy = correct / total if total > 0 else 0
        return epoch_loss / len(loader), accuracy

    def evaluate(model, loader):
        model.eval()
        epoch_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for src, tgt, src_lengths in tqdm(loader, desc="Evaluating"):
                src, tgt, src_lengths = src.to(device), tgt.to(device), src_lengths.to(device)
                output = model(src, src_lengths, tgt, teacher_forcing_ratio=0)
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

    # Training loop
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
        # MLflow logging
        mlflow.log_metric("train_loss", train_loss, step=epoch+1)
        mlflow.log_metric("train_acc", train_acc, step=epoch+1)
        mlflow.log_metric("val_loss", val_loss, step=epoch+1)
        mlflow.log_metric("val_acc", val_acc, step=epoch+1)
        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }, checkpoint_path)
        mlflow.log_artifact(checkpoint_path)
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            mlflow.log_artifact(best_path)
#%%
    # Plot curves
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()
    mlflow.log_artifact('loss_curve.png')

    plt.figure(figsize=(10,5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('accuracy_curve.png')
    plt.close()
    mlflow.log_artifact('accuracy_curve.png')

    # BLEU score
    def compute_bleu(model, loader, tgt_vocab):
        model.eval()
        references = []
        hypotheses = []
        inv_tgt_vocab = {v:k for k,v in tgt_vocab.items()}
        with torch.no_grad():
            for src, tgt, src_lengths in tqdm(loader, desc="Computing BLEU"):
                src, tgt, src_lengths = src.to(device), tgt.to(device), src_lengths.to(device)
                outputs = model(src, src_lengths, tgt, teacher_forcing_ratio=0)
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
    mlflow.log_metric("test_bleu", bleu_score)

    _, test_acc = evaluate(model, test_loader)
    logger.info(f'Test Accuracy: {test_acc:.4f}')
    mlflow.log_metric("test_acc", test_acc)