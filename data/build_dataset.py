# data/build_dataset.py
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)   # makes the random values reproducible every run
N = 800

# Anchor the output next to this script so it works from any working directory.
OUTPUT_PATH = Path(__file__).resolve().parent / 'code_quality.csv'

# readable = 1, hard_to_read = 0
# p=[0.4, 0.6] means roughly 40% hard-to-read, 60% readable
readable_mask = np.random.choice([0, 1], N, p=[0.4, 0.6])

# Class ranges deliberately overlap: real code is not cleanly separable,
# and a model trained on gapless data behaves arbitrarily for values that
# fall between the classes (e.g. nesting depth 3, complexity 6).
df = pd.DataFrame({
    'cyclomatic_complexity': np.where(
        readable_mask,
        np.random.uniform(1, 8, N),         # readable: mostly low
        np.random.uniform(5, 25, N)         # hard: mostly high, overlaps 5-8
    ),
    'max_nesting_depth': np.where(
        readable_mask,
        np.random.randint(1, 4, N),         # readable: 1-3
        np.random.randint(3, 9, N)          # hard: 3-8, overlaps at 3
    ),
    'naming_entropy': np.where(
        readable_mask,
        np.random.uniform(0.5, 1.0, N),     # readable: descriptive names
        np.random.uniform(0.0, 0.65, N)     # hard: short names, overlaps 0.5-0.65
    ),
    'avg_function_length': np.where(
        readable_mask,
        np.random.uniform(5, 35, N),        # readable: short functions
        np.random.uniform(20, 100, N)       # hard: long functions, overlaps 20-35
    ),
    'has_docstrings': np.where(
        readable_mask,
        np.random.choice([0, 1], N, p=[0.2, 0.8]),   # readable: usually documented
        np.random.choice([0, 1], N, p=[0.8, 0.2])    # hard: usually not
    ),
    'num_magic_numbers': np.where(
        readable_mask,
        np.random.randint(0, 4, N),         # readable: 0-3
        np.random.randint(2, 15, N)         # hard: 2-14, overlaps 2-3
    ),
    'num_try_except': np.random.randint(0, 3, N),    # control: same in both classes
    'readable': readable_mask
})

# Real-world labels are noisy: human reviewers disagree on readability a few
# percent of the time. Flipping 5% of labels caps AUC below a suspicious 1.0
# and keeps the model from overfitting to a perfectly separable toy problem.
noise_idx = np.random.choice(N, size=int(0.05 * N), replace=False)
df.loc[noise_idx, 'readable'] = 1 - df.loc[noise_idx, 'readable']

df.to_csv(OUTPUT_PATH, index=False)

print(f"Dataset saved: {len(df)} rows -> {OUTPUT_PATH}")
print(df['readable'].value_counts())
print(df.describe())
