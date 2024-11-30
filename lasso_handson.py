from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


df = pd.read_csv("Students_Performance.csv")
#replace empty values with Mean
dfM = df.fillna(df.mean())


X = dfM[['math_score', 'reading_score', 'writing_score','average_score','total_score']]
y = dfM['total_score']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso regression model
lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)

# Display the feature coefficients
lasso_coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lasso.coef_
})
print(lasso_coefficients)

# Visualize the coefficients
plt.figure(figsize=(8, 4))
sns.barplot(x='Feature', y='Coefficient', data=lasso_coefficients)
plt.title('Lasso Regression Coefficients')
plt.show()