##############################
### Valon Haslem
### 12 - 29 - 2025
### build df of len instance count, len dist words, and share of slang words, then plot
#############################
import pandas as pd
from tqdm import tqdm
import spacy
from collections import Counter

# PARAMETERS (tweak to taste)
CSV_PATH = 'data/training.1600000.processed.noemoticon.csv'
TEXT_COL = 'text'
CHUNKSIZE = 10000   # rows per pandas chunk read
SPACY_MODEL = 'en_core_web_sm'
SPACY_BATCH = 2000  # tokens processed per nlp.pipe batch
NLP_N_PROCESS = 1   # set >1 if your spaCy supports multiprocessing on your machine
MIN_TOKEN_LEN = 2   # ignore 1-char tokens
TOP_VOCAB = 50000   # keep this many most frequent tokens for cooccurrence pass ## NOTE: normally 50000, reduced to 500

nlp = spacy.load(SPACY_MODEL, disable=['parser', 'ner', 'textcat'])
nlp.max_length = 10_000_000  # accommodate long lines if any

def tokenize_and_calculate_share(csv_path, frac, chunksize = CHUNKSIZE):
    df = pd.read_csv(csv_path,
    names = ['target', 'ids', 'date', 'flag', 'user', 'text'],
    usecols = ['text'], 
    chunksize = chunksize,
    iterator = True,
    encoding = 'latin-1')
    for chunk in df:
        texts = chunk['text'].sample(frac = frac, random_state = 42).astype(str).tolist()
        # spaCy stream
        for doc in nlp.pipe(texts, batch_size = SPACY_BATCH, n_process=NLP_N_PROCESS):
            tokens = [tok.lemma_.lower() for tok in doc
                      if tok.is_alpha and len(tok.text) >= MIN_TOKEN_LEN]
            yield tokens # so yield by set of tokens

# set of tokens is an instance -- per instance, calculate against regular eng words
slang_words = pd.read_csv('C:/Users/valon/OneDrive/Desktop/HONS Senior Project/output1/slang_weights_top_vocab_large.csv')['SlangWord'].tolist() # NOTE

share_df = pd.DataFrame(columns= ['instance length', 'distinct tokens', 'slang share', 'iterate']) # per instance, calculate length, num of distinct words, and slang_share
# iterate: need to count off tokens_len, len(set(tokens)) as pair, then test in the counter
id_counts = Counter()
for tokens in tqdm(tokenize_and_calculate_share(CSV_PATH, frac = 1), desc='Pass through instance calc'):
    tokens_len = len(tokens) # len of instance tweet
    tokens_len_distinct = len(set(tokens)) # len of unique sets of tokens
    id_counts.update([(tokens_len, tokens_len_distinct)]) 
    share_df.loc[len(share_df)] = [tokens_len, tokens_len_distinct, 1 - len([token for token in tokens if token not in slang_words])/tokens_len if tokens_len != 0 else 0, id_counts[(tokens_len, tokens_len_distinct)]]


# plot
import matplotlib.pyplot as plt

import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# avg_slang_share = round(np.mean(share_df['slang share'].tolist()),2)

sc = ax.scatter(share_df['instance length'], 
                 share_df['distinct tokens'], 
                 share_df['iterate'],
                 c = share_df['slang share'],
                 cmap='viridis' , s=2, alpha = 0.8, edgecolor='none', linewidths=0.3)# update point size later?

# divider = make_axes_locatable(ax)
# cax = divider.append_axes(size="5%", pad=0.1)
fig.colorbar(sc, ax=ax, shrink=0.5, aspect=8, label = 'Share of Slang Words', orientation = 'horizontal')




ax.set_xlabel('Length')
ax.set_ylabel('Distinct Tokens')
ax.set_zlabel('Frequency')
# plt.xlabel('Tweet Length')
# plt.ylabel('Distinct Tokenized Words in Tweet')
ax.set_title(f'Heap\'s Law Dictionary Model') # (Average Slang Share: {avg_slang_share})

plt.savefig('output1/heaps_plot_3d_large.png', dpi = 300)
