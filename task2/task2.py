#%%
import pandas as pd
import pathlib
import pandas as pd
from collections import Counter
import torch
#%%
# Paths to the sampled data files (update if needed)
ENG_SAMPLED_PATH = '../sampled_data/eng_sampled.csv'
NL_SAMPLED_PATH = '../sampled_data/nl_sampled.csv'

# Load the sampled English and Dutch sentences
eng_sample = pd.read_csv(ENG_SAMPLED_PATH)
nl_sample = pd.read_csv(NL_SAMPLED_PATH)

#%%
# Ensure alignment
assert len(eng_sample) == len(nl_sample), 'Sampled files are not aligned!'

# Ensure no NaN values (replace with empty string)
eng_sample['sentence'] = eng_sample['sentence'].fillna('')
nl_sample['sentence'] = nl_sample['sentence'].fillna('')



# --- Preprocessing Steps ---

# 1. Remove lines with XML tags (starting with '<') in either language
mask_no_xml = (~eng_sample['sentence'].str.startswith('<')) & (~nl_sample['sentence'].str.startswith('<'))
eng_sample = eng_sample[mask_no_xml].reset_index(drop=True)
nl_sample = nl_sample[mask_no_xml].reset_index(drop=True)

# 2. Lowercase the text
eng_sample['sentence'] = eng_sample['sentence'].str.lower()
nl_sample['sentence'] = nl_sample['sentence'].str.lower()



# 3. Strip empty lines and their correspondences (remove pairs where either is empty or only whitespace)
mask_nonempty = eng_sample['sentence'].str.strip().astype(bool) & nl_sample['sentence'].str.strip().astype(bool)
eng_sample = eng_sample[mask_nonempty].reset_index(drop=True)
nl_sample = nl_sample[mask_nonempty].reset_index(drop=True)

# 4. Keep only pairs with word length less than 90 in both languages
eng_word_len = eng_sample['sentence'].apply(lambda x: len(str(x).split()))
nl_word_len = nl_sample['sentence'].apply(lambda x: len(str(x).split()))
mask_len = (eng_word_len < 40) & (nl_word_len < 40)
eng_sample = eng_sample[mask_len].reset_index(drop=True)
nl_sample = nl_sample[mask_len].reset_index(drop=True)

# Preview the cleaned data
print('Number of sentence pairs after preprocessing and length filtering:', len(eng_sample))
print('Sample English sentences:')
print(eng_sample['sentence'].head())
print('Sample Dutch sentences:')
print(nl_sample['sentence'].head())

# Save preprocessed data
import os
PREPROCESSED_DIR = '../preprocessed_data'
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
eng_sample.to_csv(f'{PREPROCESSED_DIR}/eng_preprocessed.csv', index=False)
nl_sample.to_csv(f'{PREPROCESSED_DIR}/nl_preprocessed.csv', index=False)
print(f"Saved preprocessed English data to {PREPROCESSED_DIR}/eng_preprocessed.csv")
print(f"Saved preprocessed Dutch data to {PREPROCESSED_DIR}/nl_preprocessed.csv")



#%%
import pandas as pd 
import pathlib
import os
import matplotlib.pyplot as plt
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

file_path = pathlib.Path(__file__).parent.parent / 'preprocessed_data' 

eng = pd.read_csv(f'{file_path}/eng_preprocessed.csv')

nl = pd.read_csv(f'{file_path}/nl_preprocessed.csv')



# Boxplot for sentence length in words
plt.figure(figsize=(8,6))
plt.boxplot([eng['word_count'], nl['word_count']], patch_artist=True)
plt.title('Sentence Length Variance (Words)')
plt.ylabel('Number of Words')
plt.xticks([1, 2], ['English', 'Dutch'])
plt.savefig(f"{PLOTS_DIR}/boxplot_word_count.png")
plt.close()

# Boxplot for sentence length in characters
plt.figure(figsize=(8,6))
plt.boxplot([eng['char_count'], nl['char_count']], patch_artist=True)
plt.title('Sentence Length Variance (Characters)')
plt.ylabel('Number of Characters')
plt.xticks([1, 2], ['English', 'Dutch'])
plt.savefig(f"{PLOTS_DIR}/boxplot_char_count.png")
plt.close()

# Histogram for word count
plt.figure(figsize=(12,5))
plt.hist(eng['word_count'], bins=50, alpha=0.5, label='English')
plt.hist(nl['word_count'], bins=50, alpha=0.5, label='Dutch')
plt.legend()
plt.title('Sentence Length Distribution (Words)')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.savefig(f"{PLOTS_DIR}/hist_word_count.png")
plt.close()

# Histogram for character count
plt.figure(figsize=(12,5))
plt.hist(eng['char_count'], bins=50, alpha=0.5, label='English')
plt.hist(nl['char_count'], bins=50, alpha=0.5, label='Dutch')
plt.legend()
plt.title('Sentence Length Distribution (Characters)')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')

plt.savefig(f"{PLOTS_DIR}/hist_char_count.png")
plt.close()

# Scatter plot for word count comparison
plt.figure(figsize=(8,8))
plt.scatter(eng['word_count'], nl['word_count'], alpha=0.1, s=1)
plt.xlabel('English Sentence Length (words)')
plt.ylabel('Dutch Sentence Length (words)')
plt.title('English vs Dutch Sentence Lengths')

plt.savefig(f"{PLOTS_DIR}/scatter_word_count.png")
plt.close()



#%%

# Now eng_sample['sentence'] and nl_sample['sentence'] are preprocessed and ready for further steps
'''
# Task specific pre -processing

# 2. Tokenization (simple whitespace)
eng_tokens = [s.split() for s in eng_sample['sentence']]
nl_tokens = [s.split() for s in nl_sample['sentence']]

# 3. Add <SOS> and <EOS> to target (Dutch) sentences
nl_tokens = [['<SOS>'] + s + ['<EOS>'] for s in nl_tokens]

# 4. Build vocabularies
def build_vocab(token_lists, min_freq=2):
    counter = Counter(token for sent in token_lists for token in sent)
    vocab = {'<PAD>':0, '<SOS>':1, '<EOS>':2, '<UNK>':3}
    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab

eng_vocab = build_vocab(eng_tokens)
nl_vocab = build_vocab(nl_tokens)

# 5. Integer encoding
def encode(tokens, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

eng_indices = [encode(sent, eng_vocab) for sent in eng_tokens]
nl_indices = [encode(sent, nl_vocab) for sent in nl_tokens]

# 6. Padding
def pad_sequences(sequences, max_len, pad_value=0):
    return [seq + [pad_value]*(max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in sequences]

max_len_eng = max(len(seq) for seq in eng_indices)
max_len_nl = max(len(seq) for seq in nl_indices)
#%%
min_len_eng = min(len(seq) for seq in eng_indices)
min_len_nl = min(len(seq) for seq in nl_indices)
print(min_len_eng)
#%%
# Find the index of the shortest English sentence
min_len_eng = min(len(seq) for seq in eng_indices)
min_idx_eng = [i for i, seq in enumerate(eng_indices) if len(seq) == min_len_eng][0]
print("Shortest English sentence (length={}):".format(min_len_eng))
print(eng_sample['sentence'].iloc[min_idx_eng])

# Do the same for Dutch
min_len_nl = min(len(seq) for seq in nl_indices)
min_idx_nl = [i for i, seq in enumerate(nl_indices) if len(seq) == min_len_nl][0]
print("Shortest Dutch sentence (length={}):".format(min_len_nl))
print(nl_sample['sentence'].iloc[min_idx_nl])'''