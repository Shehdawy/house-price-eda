"""
Complete EDA for Housing dataset.
Loads data, cleans/encodes categoricals, handles outliers, generates comprehensive plots.
Plots saved to eda_plots/. Modifies df in-place (encoded + outlier-capped).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots dir if not exists
os.makedirs('eda_plots', exist_ok=True)

 

df = pd.read_csv("Housing.csv")
print("Dataset Shape:", df.shape)
print(df.info())
print(df.head())

# Data cleaning and encoding
print("\nMissing values:\n", df.isnull().sum())
print("\nDescribe:\n", df.describe())
print("\nPrice skewness:", df['price'].skew())
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[binary_cols] = df[binary_cols].map({'yes': 1, 'no': 0})

# Furnishing status ordinal encoding
furnish_map = {'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0}
df['furnishingstatus'] = df['furnishingstatus'].map(furnish_map)

print("\nEncoded data describe:\n", df.describe())

# Outlier detection (cap price at 99th percentile)
price_99 = df['price'].quantile(0.99)
df['price'] = df['price'].clip(upper=price_99)
print("\nPrice after outlier capping (99th %ile):", price_99)

# Univariate
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True)
plt.title("Price Distribution (after outlier capping)")
plt.savefig('eda_plots/price_dist.png')
plt.show()

# Bivariate (expanded)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=df['area'], y=df['price'])
plt.title("Area vs Price")
plt.subplot(1, 2, 2)
sns.boxplot(x=df['bedrooms'], y=df['price'])
plt.title("Bedrooms vs Price")
plt.tight_layout()
plt.savefig('eda_plots/bivariate.png')
plt.show()
# Full correlation matrix (post-encoding)
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Full Correlation Matrix (Encoded Features)")
plt.savefig('eda_plots/corr_heatmap.png')
plt.show()
# More categorical plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
categorical_cols = ['stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning']
for i, col in enumerate(categorical_cols):
    sns.boxplot(x=col, y='price', data=df, ax=axes[i//3, i%3])
    axes[i//3, i%3].set_title(f"Price vs {col}")
plt.tight_layout()
plt.savefig('eda_plots/categorical.png')
plt.show()

print("EDA complete. Plots saved to eda_plots/")




from scipy.stats import ttest_ind

price_pref = df[df['prefarea']==1]['price']
price_nonpref = df[df['prefarea']==0]['price']

t_stat, p_val = ttest_ind(price_pref, price_nonpref, equal_var=False)
print("t-statistic:", t_stat)
print("p-value:", p_val)