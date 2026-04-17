import matplotlib.pyplot as plt
import powerlaw
import numpy as np
import pandas as pd

# 1. Generate sample data (Lognormal data often mimics Power Laws)
data = pd.read_csv('C:/Users/valon/OneDrive/Desktop/HONS Senior Project/output1/slang_weights_top_vocab_large.csv')['TotalWeight']


# 2. Fit the data
fit = powerlaw.Fit(data, verbose=False, discrete=True)

# 3. Statistical Comparison
# R is the log-likelihood ratio. 
# R > 0: Power Law is a better fit.
# R < 0: Lognormal is a better fit.
R, p = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)

print(f"Log-Likelihood Ratio (R): {R:.4f}")
print(f"Significance (p): {p:.4f}")

# 4. Visualization
plt.figure(figsize=(10, 6))

# Plot the real data (using the CCDF for a smoother look)
fig = fit.plot_ccdf(color='black', linewidth=2, label='Empirical Data (CCDF)')

# Plot the two theoretical models
fit.power_law.plot_ccdf(color='r', linestyle='--', ax=fig, label='Power Law Fit')
fit.lognormal.plot_ccdf(color='g', linestyle='--', ax=fig, label='Lognormal Fit')

# Formatting
plt.title('Power Law vs. Lognormal Fit For Dictionary Slang Data', fontsize=14)
plt.xlabel('Value (x)', fontsize=12)
plt.ylabel('P(X ≥ x)', fontsize=12)
plt.legend()
plt.savefig('output2/plaw_dict_slang.png', dpi = 300)