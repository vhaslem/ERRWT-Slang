####
## VALON
### 11 - 12 -- 2025
## OBJ: Using matplotlib, using adjacency values for all words in dataset and slang words in dataset, graph to evaluate

### 12 - 3 -2025
## using updated diagrams, outputting to new output
### ZIPFS law -- rank frequency graph, combine both figs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# load
slang_base_df = pd.read_csv('C:/Users/valon/OneDrive/Desktop/HONS Senior Project/output1/slang_weights_top_vocab_large.csv')
words_base_df = pd.read_csv('C:/Users/valon/OneDrive/Desktop/HONS Senior Project/output1/word_weights_top_vocab_large.csv')
slang_errwt_df = pd.read_csv('C:/Users/valon/OneDrive/Desktop/HONS Senior Project/output2/slang_weights_top_vocab_errwt_large.csv')
words_errwt_df = pd.read_csv('C:/Users/valon/OneDrive/Desktop/HONS Senior Project/output2/word_weights_top_vocab_errwt_large.csv')

# modify words_base_df currently overlaps
words_base_filtered_df = words_base_df[words_base_df['IsEnglish'] == True]
# need to assign rankings
slang_base_ranking = pd.Series(np.arange(1, len(slang_base_df) + 1))
words_base_ranking = pd.Series(np.arange(1, len(words_base_filtered_df) + 1))
slang_errwt_ranking = pd.Series(np.arange(1, len(slang_errwt_df) + 1))
words_errwt_ranking = pd.Series(np.arange(1, len(words_errwt_df) + 1))

######################
# Show theoretical powerlaw
import powerlaw
def calculate_p_value(data, simulations=2500):
    """
    Calculates the p-value for a power law fit using bootstrapping.
    p > 0.1 means the power law is a plausible model.
    """
    # 1. Fit the original data
    original_fit = powerlaw.Fit(data, verbose=False, discrete=True)
    D_obs = original_fit.power_law.D
    alpha_obs = original_fit.power_law.alpha
    xmin_obs = original_fit.xmin
    
    print(f"Observed: alpha={alpha_obs:.4f}, xmin={xmin_obs:.4f}, D={D_obs:.4f}")
    
    # 2. Start Simulations
    count_D_greater = 0
    print(f"Running {simulations} simulations...")
    
    for i in range(simulations):
        # Generate synthetic data from the fitted distribution
        # Note: we must generate data with the same size and xmin as original
        synthetic_data = original_fit.power_law.generate_random(len(data))
        
        # Fit the synthetic data (estimating parameters again!)
        sim_fit = powerlaw.Fit(synthetic_data, xmin=xmin_obs, verbose=False)
        
        if sim_fit.power_law.D >= D_obs:
            count_D_greater += 1
            
        if (i + 1) % 20 == 0:
            print(f"Iteration {i+1} complete...")

    p_value = count_D_greater / simulations
    return p_value


# gen figure
plt.figure(figsize=(8,6))
plt.loglog(slang_base_ranking, slang_base_df['TotalWeight'], 'o-', label = f"Slang Words Dictionary Model p = {calculate_p_value(slang_base_df['TotalWeight']):.4f}", c = 'purple', alpha = .9) # log log 
plt.loglog(words_base_ranking, words_base_filtered_df['TotalWeight'], 's-', label = f"English Words Dictionary Model p = {calculate_p_value(words_base_filtered_df['TotalWeight']):.4f}", c = 'red', alpha = .4)
plt.loglog(slang_errwt_ranking, slang_errwt_df['TotalWeight'], '^-', label = f"Slang Words ERRWT Model p = {calculate_p_value(slang_errwt_df['TotalWeight']):.4f}", c= 'green', alpha = .9) # log log 
plt.loglog(words_errwt_ranking, words_errwt_df['TotalWeight'], 'x-', label = f"English Words ERRWT Model p = {calculate_p_value(words_errwt_df['TotalWeight']):.4f}", c = 'darkblue', alpha = .4)
# Labels and legend
plt.xlabel('Rank (log scale)')
plt.ylabel('Frequency (log scale)')
plt.title('Rank-Frequency Diagram (Zipf Plot)')
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
# save, dont show
plt.savefig('output2/zipf_grouped_p_vals.png', dpi = 300)
