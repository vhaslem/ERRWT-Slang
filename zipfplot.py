####
## VALON
### 11 - 12 -- 2025
## OBJ: Using matplotlib, using adjacency values for all words in dataset and slang words in dataset, graph to evaluate

### ZIPFS law -- rank frequency graph, combine both figs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# load
slang_df = pd.read_csv('slang_weights_top_vocab.csv')
words_df = pd.read_csv('word_weights_top_vocab.csv')

# need to assign rankings
slang_ranking = pd.Series(np.arange(1, len(slang_df) + 1))
words_ranking = pd.Series(np.arange(1, len(words_df) + 1))


# gen figure
plt.figure(figsize=(8,6))
plt.loglog(slang_ranking, slang_df['TotalWeight'], 'o-', label = 'Slang Words', alpha = .7) # log log 
plt.loglog(words_ranking, words_df['TotalWeight'], 's-', label = 'All Words', alpha = .7)

# Labels and legend
plt.xlabel('Rank (log scale)')
plt.ylabel('Frequency (log scale)')
plt.title('Rank-Frequency Diagram (Zipf Plot)')
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
# save, dont show
plt.savefig('figs/zipf_both.png')
