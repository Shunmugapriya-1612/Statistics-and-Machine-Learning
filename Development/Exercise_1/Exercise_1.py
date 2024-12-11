from scipy import stats as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("Student_performance.csv")

# Mean of Total Score
Mean = np.mean(data['total_score'])
print("Mean: ", Mean)

# Mean of Total Score
Median = np.median(data['total_score'])
print("Median: ", Median)

# Mode of Total Score
Mode = st.mode(data['total_score'])
print("Mode: ", Mode)

# Variance of Total Score
variance = np.var(data['total_score'])
print("Variance: ", variance)

# StdDev of Total Score
StdDev = np.std(data['total_score'])
print("Standard Deviation: ", StdDev)

# Plotting Total score in Histogram
minus1StdDev = Mean - StdDev
plt.axvline(minus1StdDev,color = "r", linestyle = "dashed", linewidth =1)

Plus1StdDev = Mean + StdDev
plt.axvline(Plus1StdDev,color = "r", linestyle = "dashed", linewidth = 1)

plt.hist(data['total_score'])
plt.show()

# Plotting Total score in BoxPlot
plt.boxplot(data['total_score'])
plt.show()

