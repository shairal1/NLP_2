#%%
import pandas as pd 
import pathlib
import os
import matplotlib.pyplot as plt

PLOTS_DIR = "task1/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

file_path = pathlib.Path(__file__).parent.parent / 'data' / 'nl-en' 
with open(file_path/'europarl-v7.nl-en.en', encoding='utf-8') as f:
    english_sentences = f.read().splitlines()
eng = pd.DataFrame({'sentence': english_sentences})
with open(file_path/'europarl-v7.nl-en.nl', encoding='utf-8') as f:
    dutch_sentences= f.read().splitlines()
nl = pd.DataFrame({'sentence': dutch_sentences})

# Statistics
print('Number of sentences (English):', len(eng))
print('Number of sentences (Dutch):', len(nl))

eng['word_count'] = eng['sentence'].apply(lambda x: len(str(x).split()))
nl['word_count'] = nl['sentence'].apply(lambda x: len(str(x).split()))

eng['char_count'] = eng['sentence'].apply(lambda x: len(str(x)))
nl['char_count'] = nl['sentence'].apply(lambda x: len(str(x)))
#%%
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
# Data Sampling (10%)
sample_frac = 0.1
sample_indices = eng.sample(frac=sample_frac, random_state=42).index
eng_sample = eng.loc[sample_indices].reset_index(drop=True)
nl_sample = nl.loc[sample_indices].reset_index(drop=True)

print(f'Randomly sampled {len(eng_sample)} sentence pairs for training.')
SAMPLED_DATA = "../sampled_data"
os.makedirs(SAMPLED_DATA, exist_ok=True)
eng_sample.to_csv(f'{SAMPLED_DATA}/eng_sampled.csv', index=False)
nl_sample.to_csv(f'{SAMPLED_DATA}/nl_sampled.csv', index=False)
#%%
print("Saved:", os.path.abspath(f'{SAMPLED_DATA}/eng_sampled.csv'))
print("Saved:", os.path.abspath(f'{SAMPLED_DATA}/nl_sampled.csv'))