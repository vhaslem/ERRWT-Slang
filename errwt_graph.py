###############################3
# Valon 
# 11 - 16 - 2025
# This code uses the errwt model to determine slang, by taking a share of 
# the dataset (start with 20-80) and then tests the remainder and checks for new nodes
####################


## first, select 20% of data to classify as english
import os
from collections import Counter
import itertools
import math
import csv

import pandas as pd
from tqdm import tqdm

import spacy
from scipy.sparse import dok_matrix, csr_matrix
import numpy as np
import networkx as nx


# PARAMETERS (tweak to taste)
CSV_PATH = 'data/training.1600000.processed.noemoticon.csv'
TEXT_COL = 'text'
CHUNKSIZE = 10000   # rows per pandas chunk read
SPACY_MODEL = 'en_core_web_sm'
SPACY_BATCH = 2000  # tokens processed per nlp.pipe batch
NLP_N_PROCESS = 1   # set >1 if your spaCy supports multiprocessing on your machine
MIN_TOKEN_LEN = 2   # ignore 1-char tokens
TOP_VOCAB = 25000   # keep this many most frequent tokens for cooccurrence pass ## NOTE: normally 50000, reduced to 500

nlp = spacy.load(SPACY_MODEL, disable=['parser', 'ner', 'textcat'])
nlp.max_length = 10_000_000  # accommodate long lines if any

def tokenize_texts_iter(csv_path, frac, chunksize=CHUNKSIZE):
    reader = pd.read_csv(csv_path,
                         names=['target','ids','date','flag','user','text'],
                         usecols=['text'],
                         chunksize=chunksize,
                         iterator=True,
                         encoding='latin-1')
    for chunk in reader:
        texts = chunk['text'].sample(frac = frac, random_state = 42).astype(str).tolist()
        # use spaCy streaming
        for doc in nlp.pipe(texts, batch_size=SPACY_BATCH, n_process=NLP_N_PROCESS):
            tokens = [tok.lemma_.lower() for tok in doc
                      if tok.is_alpha and len(tok.text) >= MIN_TOKEN_LEN]
            yield tokens
###### NOTE: first pull takes 20%


# First pass: count token frequencies across full dataset (streaming)
token_counter = Counter()
total_docs = 0
for tokens in tqdm(tokenize_texts_iter(CSV_PATH, frac = 1), desc='Pass1 token counting'):
    token_counter.update(tokens)
    total_docs += 1

print(f"Seen {total_docs} documents; {len(token_counter)} unique tokens")
# Keep top-K as candidate vocabulary
top_tokens = [w for w, _ in token_counter.most_common(TOP_VOCAB)]
vocab_set = set(top_tokens)

# Build index map for top tokens
vocab = top_tokens  # ordered by frequency
vocab_index = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
print(f"Using vocab size V={V}")

# PARAMETERS for pass 2
WINDOW = 10   # consider co-occurrence within this window (tokens to either side)
USE_COMBINATIONS = False  # if True, counts all pairs in the doc (O(L^2)). If False, use sliding window O(L * WINDOW)

# we'll build a dok_matrix first (efficient incremental updates), then convert to csr
cooc = dok_matrix((V, V), dtype=np.int32)

def process_and_update_cooc(tokens, vocab_index, cooc_mat, window=WINDOW):
    # convert tokens to indices ignoring tokens outside vocab
    indices = [vocab_index[t] for t in tokens if t in vocab_index]
    L = len(indices)
    if L <= 1:
        return
    if USE_COMBINATIONS:
        # all unordered pairs in document
        for i, j in itertools.combinations(indices, 2):
            cooc_mat[i, j] = cooc_mat.get((i, j), 0) + 1
            cooc_mat[j, i] = cooc_mat.get((j, i), 0) + 1
    else:
        # sliding window approach: for each token, look ahead up to window tokens
        for pos, i in enumerate(indices):
            end = min(pos + window, L - 1)
            for pos2 in range(pos + 1, end + 1):
                j = indices[pos2]
                cooc_mat[i, j] = cooc_mat.get((i, j), 0) + 1
                cooc_mat[j, i] = cooc_mat.get((j, i), 0) + 1

# Second pass streaming and update co-occurrence (this is the heavy step)
for tokens in tqdm(tokenize_texts_iter(CSV_PATH, frac = 1), desc='Pass2 building cooc'):
    process_and_update_cooc(tokens, vocab_index, cooc, window=WINDOW)

# Convert to csr for efficient math and storage
cooc_csr = cooc.tocsr()
print("Co-occurrence matrix built:", cooc_csr.shape)

### START HERE
###
all_tokens = list(tokenize_texts_iter(CSV_PATH, frac=1.0))

N = len(all_tokens)
eng_tokens = all_tokens[: int(0.2*N)]
slang_tokens = all_tokens[int(0.2*N):]

eng_words = set(tok for tokens in eng_tokens for tok in tokens)
slang_candidates = {tok for tokens in slang_tokens for tok in tokens if tok not in eng_words}
# eng_words = {tok for tokens in tokenize_texts_iter(CSV_PATH, frac=.2) for tok in tokens} # generates a small graph -- so only identify new nodes
# slang_candidates = {tok for tokens in tokenize_texts_iter(CSV_PATH, frac=0.8) for tok in tokens if tok not in eng_words} # eng words is set, so should be fast
# NOTE: these are not disjoint sets, could be problem -- is there way to make disjoint?


print(f"English words in vocab: {len(eng_words)}; slang candidates: {len(slang_candidates)}")
# Total weight = sum of cooccurrence counts for each word index
row_sums = np.array(cooc_csr.sum(axis=1)).flatten()  # shape (V,)

# Write English word weights
with open('word_weights_top_vocab_errwt.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Word', 'TotalWeight', 'IsEnglish'])
    for i, w in enumerate(vocab):
        writer.writerow([w, int(row_sums[i]), w in eng_words])

# Write slang weights (top by weight)
# slang_indices = [vocab_index.get(w, 0) for w in slang_candidates] #[vocab_index[w] for w in slang_candidates] # NOTE: set default to zero just to put at back
UNK_INDEX = len(vocab) - 1
slang_indices = [vocab_index.get(w, UNK_INDEX) for w in slang_candidates]
slang_sorted = sorted(slang_indices, key=lambda i: row_sums[i], reverse=True)
with open('slang_weights_top_vocab_errwt.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['SlangWord', 'TotalWeight'])
    for idx in slang_sorted[:5000]:  # cap output to top 5k slang candidates
        writer.writerow([vocab[idx], int(row_sums[idx])])

TOP_GRAPH = 200  # number of slang nodes to visualize / save
top_slang_by_weight = [vocab[i] for i in slang_sorted[:TOP_GRAPH]]
top_indices = [vocab_index[w] for w in top_slang_by_weight]
G = nx.Graph()

# Add nodes
for w in top_slang_by_weight:
    G.add_node(w)

# Add edges based on co-occurrence weights (sparse access)
for i in top_indices:
    row = cooc_csr.getrow(i)
    # get nonzeros
    cols = row.indices
    data = row.data
    for col_idx, wt in zip(cols, data):
        if col_idx in top_indices and col_idx != i:
            w1 = vocab[i]
            w2 = vocab[col_idx]
            if not G.has_edge(w1, w2):
                G.add_edge(w1, w2, weight=int(wt))

# Save graph
nx.write_gml(G, 'slang_network_top_errwt.gml')
print(f"Saved graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# small visualization (only for small graphs)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, seed=42)  # deterministic layout
weights = [G[u][v]['weight'] for u, v in G.edges()]
nx.draw(G, pos, with_labels=True, node_size=300, width=np.log1p(weights))
plt.savefig('slang_graph_top.png', dpi=200)
