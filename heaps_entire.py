#####################3
## Valon Haslem
## 1 - 23 - 2026
## This code treats the entire dataset as one corpus
## tokenizes/lemmatizes data
## then filters through words and counts # of words against # of distinct words
## then graphs in matplotlib and fits exp
###################


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
TOP_VOCAB = 50000   # keep this many most frequent tokens for cooccurrence pass 


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
import numpy as np
heaps_df = pd.DataFrame(columns=['Word Count', 'Distinct Word Count'])
# iterate: need to count off tokens_len, len(set(tokens)) as pair, then test in the counter
id_counts = Counter()
total_tokens_amt = 0
for tokens in tqdm(tokenize_and_calculate_share(CSV_PATH, frac = 1), desc='Pass through tokenizer/lemmatizer'):
    id_counts.update(tuple(tokens))
    tokens_distinct = len(id_counts)
    total_tokens_amt = total_tokens_amt + len(tokens)
    heaps_df.loc[len(heaps_df)] =[total_tokens_amt, tokens_distinct]

from sklearn.linear_model import LinearRegression

## take exp regression

# log transform
N = np.array(heaps_df['Word Count'])
V = np.array(heaps_df['Distinct Word Count'])
log_N = np.log10(N).reshape(-1,1)
log_V = np.log10(V).reshape(-1,1)

model = LinearRegression()
model.fit(X = log_N, y = log_V)

# Extract parameters
beta = model.coef_[0]
log_K = model.intercept_
K = 10**log_K
# V = KN^beta
x_vals = np.linspace(N[0], N[-1], 100000)
V_est = K * (x_vals ** beta)
Vt_low = 10 * (x_vals ** .4)
# Vt_high = 100 * (x_vals ** .6)
## plot
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(N, V, s = 1, c = 'darkviolet', label = 'Empirical Data')
plt.plot(x_vals, V_est, linewidth = 2, c = 'green', alpha = .5, label = rf'$V = {K[0]:.2f}N^{{{beta[0]:.2f}}}$')
# plt.plot(x_vals, Vt_high, linewidth = 1, c = 'black', label = rf'Theoretical Upper Bound')
plt.plot(x_vals, Vt_low, linewidth = 1, c = 'black', label = 'English Typical Lower Bound $V = 10N^{0.4}$')
plt.xlabel('Tokens Count')
plt.ylabel('Distinct Tokens Count')
plt.title("Empirical Heaps\'s Law")
plt.legend()
plt.savefig('output2/entire_heaps_empirical_test.png', dpi = 300)