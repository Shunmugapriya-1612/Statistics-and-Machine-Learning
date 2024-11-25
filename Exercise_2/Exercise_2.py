import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Data Loading and Overview
# Loading the Iris dataset
iris = sns.load_dataset('iris')

# Display the first few rows of the dataset
print(iris.head())

# Step 2: Simple Random Sampling
# Randomly sample 30 observations from the dataset
random_sample = iris.sample(n=30, random_state=42)
print("Random Sample (30 observations):")
print(random_sample)

# Step 3: Sample Mean Distribution Analysis
# Initialize a list to hold the sample means
sample_means = []

# Repeat the random sampling 100 times
for _ in range(100):
    sample = iris.sample(n=30, random_state=None)  # No seed for randomness
    sample_mean = sample['sepal_length'].mean()  # Calculate the mean of sepal_length
    sample_means.append(sample_mean)

# Plot the distribution of these sample means using a histogram
plt.figure(figsize=(10, 6))
plt.hist(sample_means, bins=20, edgecolor='black', alpha=0.7)
plt.title("Distribution of 100 Sample Means")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Step 4: Systematic Sampling
def systematic_sampling(data, sample_fraction):
    # Determine the step size based on the sample fraction
    step = int(1 / sample_fraction)
    # Select every 'step'-th row
    sample_data = data.iloc[::step]
    return sample_data

# Take 20% of the dataset using systematic sampling
systematic_sample = systematic_sampling(iris, sample_fraction=0.2)

# Plot the original dataset and the sample dataset
plt.figure(figsize=(12, 6))

# Plot the original dataset
plt.subplot(1, 2, 1)
sns.scatterplot(x='sepal_length', y='sepal_width', data=iris, hue='species')
plt.title("Original Iris Dataset")

# Plot the systematic sample dataset
plt.subplot(1, 2, 2)
sns.scatterplot(x='sepal_length', y='sepal_width', data=systematic_sample, hue='species')
plt.title("Systematic Sample (20%)")

plt.tight_layout()
plt.show()
