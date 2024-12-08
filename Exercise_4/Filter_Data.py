import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score

df = pd.read_csv("Students_Performance.csv")
#print(df)
df.head()

#Handling Missing Values

#Mean
df_imputed_mean = df['total_score'].fillna(df['total_score'].mean())
print(df_imputed_mean)

#Median
df_imputed_median = df['total_score'].fillna(df['total_score'].median())
print(df_imputed_median)

#Mode
df_imputed_mode = df['total_score'].fillna(df['total_score'].mode().iloc[0])
print(df_imputed_mode)

#Scaling

#Standardization
mean = df['math_score'].mean()
std = df['math_score'].std()
standardized_series = (df['math_score'] - mean) / std
print(standardized_series)

#Normalization
df_normalized = (df['reading_score'] - df['reading_score'].min()) / (df['reading_score'].max() - df['reading_score'].min())
print("Normalized DataFrame:")
print(df_normalized)

sorted_df = df.sort_values(by='math_score', ascending=True)
X = df['reading_score'].sort_values(ascending=True)
y = df['total_score'].sort_values(ascending=True)

#manual noise injected
y_noise = np.sin(X)

#sort the normalized data set
y_clean= np.sin(df_normalized).sort_values(ascending=True)

#smooth the noisy dataset
y_smooth=np.convolve(y_noise,np.ones(5)/5,mode='valid')
x_smooth = X[:len(y_smooth)]


#x_smooth = X.rolling(window=3).mean()


plt.figure(figsize=(8, 6))

plt.scatter(X, y_noise, label='Noise data', color='green')
plt.plot(X, y_clean, label='Clean Sine Wave', color='blue')
plt.plot(x_smooth, y_smooth, label='Smooth Sine Wave', color='yellow')
plt.title('plt with Noise or Outliers)')
plt.xlabel('Reading Score')
plt.ylabel('Total Score')
plt.legend()
plt.grid(True)
plt.show()


#outlier dataset
z_scores = [(X - mean) / std for X in df['reading_score']]
print(z_scores)

threshold = 3
outliers = [X for X, z in zip(df['reading_score'], z_scores) if abs(z) > threshold]

print("Outliers", outliers)


#Transform Outliers
lower_bound, upper_bound = np.percentile(df['reading_score'], [5, 95])
Transformed_data = [max(lower_bound, min(x, upper_bound)) for x in df['reading_score']]

print("Transformed Data:", Transformed_data)

#-----------------------------------------------------------------------------------------

#Filter methods

#Pearson correlation
correlations = df.corr()['total_score'].drop('total_score')
print("Feature Correlations:\n", correlations)

#Mutual Information
X_Input = df[['math_score', 'reading_score', 'writing_score','average_score']]
Y_Input = df['total_score']

# Calculate mutual information
mi = mutual_info_regression(X_Input, Y_Input)

# Convert to a pandas Series for easy interpretation
mi = pd.Series(mi, index=X_Input.columns).sort_values(ascending=False)

print("Mutual Information Scores:\n", mi)

#---------------------------------------------------------------------------------------

#Recursive Feature Elimination
X_Inputs = df[['math_score', 'reading_score', 'writing_score','average_score','total_score']]
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=2)
rfe.fit(X_Inputs.drop(columns='total_score'), df['total_score'])
selected_features = X_Inputs.drop(columns='total_score').columns[rfe.support_].tolist()
print("Selected Features with RFE:", selected_features)

#--------------------------------------------------------------------------------------------------

def backward_feature_selection(X_Inputs, Y_Input, model, scoring='neg_mean_squared_error'):
    features = list(X_Inputs.columns)
    while len(features) >= 1:
        # Evaluate model performance with current features
        scores = cross_val_score(model, X_Inputs[features], Y_Input, cv=3, scoring=scoring)
        print(f"Features: {features}, CV Score: {np.mean(scores):.4f}")

        # Calculate feature importances (coefficients for linear regression)
        model.fit(X_Inputs[features], Y_Input)
        importances = np.abs(model.coef_)

        # Identify least important feature
        least_important = features[np.argmin(importances)]

        # Remove the least important feature
        features.remove(least_important)

    return features

# Run backward feature selection
selected_features = backward_feature_selection(X_Inputs, Y_Input, model)
print("Selected Features:", selected_features)




