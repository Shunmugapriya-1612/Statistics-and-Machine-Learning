import seaborn as sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load the Iris dataset
iris = sns.load_dataset('iris')

# 1. T-Test: Compare mean petal lengths of Setosa vs Versicolor
setosa_petal_length = iris[iris['species'] == 'setosa']['petal_length']
versicolor_petal_length = iris[iris['species'] == 'versicolor']['petal_length']

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(setosa_petal_length, versicolor_petal_length)
print(f"T-test between Setosa and Versicolor petal lengths:")
print(f"T-statistic: {t_stat}, p-value: {p_value}")

# Null hypothesis: Means are equal; if p-value < 0.05, reject the null hypothesis.

# 2. Z-Test: Test if the mean sepal length of Setosa equals 5.0
setosa_sepal_length = iris[iris['species'] == 'setosa']['sepal_length']
population_mean = 5.0
population_std = setosa_sepal_length.std()  # Standard deviation of the sample (approximating population)
n = len(setosa_sepal_length)

# Z-test statistic calculation
z_stat = (setosa_sepal_length.mean() - population_mean) / (population_std / np.sqrt(n))
p_value_z = stats.norm.sf(abs(z_stat)) * 2  # Two-tailed test
print(f"\nZ-test for Setosa sepal length against population mean 5.0:")
print(f"Z-statistic: {z_stat}, p-value: {p_value_z}")

# Null hypothesis: Mean equals 5.0; if p-value < 0.05, reject the null hypothesis.

# 3. ANOVA: Compare mean petal widths across all three species
anova_data = iris[['species', 'petal_width']]
model = ols('petal_width ~ species', data=anova_data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("\nANOVA: Compare mean petal widths across all species")
print(anova_table)

# Null hypothesis: All means are equal; if p-value < 0.05, reject the null hypothesis.

# 4. Correlation/Regression: Explore the relationship between sepal length and petal length
sepal_length = iris['sepal_length']
petal_length = iris['petal_length']

# Calculate the correlation coefficient
correlation = np.corrcoef(sepal_length, petal_length)[0, 1]
print(f"\nCorrelation between Sepal Length and Petal Length: {correlation}")

# Scatter plot and regression line
plt.figure(figsize=(8, 6))
sns.regplot(x='sepal_length', y='petal_length', data=iris, line_kws={'color': 'red'})
plt.title("Scatter Plot and Regression Line: Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()
