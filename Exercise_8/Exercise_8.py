from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier


heart_disease = fetch_ucirepo(id=45) 
  
X = pd.DataFrame(heart_disease.data.features, columns=heart_disease.data.headers)

y = pd.Series(heart_disease.data.targets.squeeze(), name="num")

df = pd.concat([X, y], axis=1)
df = df.dropna(axis=1, how='all')
missing_values_before = df.isnull().sum()
print("*************Data Cleaning********************")
print("\nMissing values per column before imputation:")
print(missing_values_before[missing_values_before > 0])
print("\n")

imputer = KNNImputer(n_neighbors=5)

# Performing KNN imputation
imputed_data = imputer.fit_transform(df)
df_imputed = pd.DataFrame(imputed_data, columns=df.columns)


# Replacing the empty values with the imputed values
df['num'] = df_imputed['num']
df['ca'] = df_imputed['ca']
df['thal'] = df_imputed['thal']

missing_values_after = df_imputed.isnull().sum()
print("\nMissing values per column after imputation:")
print(missing_values_after[missing_values_after > 0])
print("\n")

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = "num"), df['num'], test_size=0.2, random_state=42)
#---------------------------------------------------------------------------------------------
#training dataset using Decision tree Algorithm
print("****************************Decision Tree***********************************")
# Decision tree                  
dt_classifier = DecisionTreeClassifier(criterion='entropy',random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred = dt_classifier.predict (X_test)

#decision tree rules
print(export_text(dt_classifier, feature_names=list(df.drop(columns = "num"))))

#accuracy measure on predicted dataset based on test data
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred,zero_division=0))


#--------------------------------------------------------------------------------------------------
#confusion matrix for decision tree algorithm(ID3)

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):

    cm = confusion_matrix(y_true, y_pred)

    # heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])

    #labels and title
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)

    # Display the plot
    plt.show()

plot_confusion_matrix(y_test, y_pred, title="ID3 (Decision Tree) Confusion Matrix")

#tree shaped visuzalition
plt.figure(figsize=(15, 10))
tree.plot_tree(dt_classifier, filled=True, feature_names=df.drop(columns = "num").columns, class_names=heart_disease.data.target_names, rounded=True, fontsize=12)
plt.show()
#----------------------------------------------------------------------------------------------
#training dataset using random forest Algorithm
print("****************************Random Forest***********************************")
rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions,zero_division=0))

#heat map visualization on random forest confusion matrix
plot_confusion_matrix(y_test, rf_predictions, title="Random forest Confusion Matrix")
#--------------------------------------------------------------------------------------------------
#training dataset using Gradient Boosted Trees Algorithm
print("****************************Gradient Boosted Trees***********************************")
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_clf.fit(X_train, y_train)


gb_clf_predictions = gb_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

#heat map visualization on Gradient Boosted Trees confusion matrix
plot_confusion_matrix(y_test, gb_clf_predictions, title="Gradient Boosted Trees Confusion Matrix")
#--------------------------------------------------------------------------------------------------
#highest accuracy is shown when 11 features are selected to train the model using knn
print("****************************KNN***********************************")
knn_model_11 = KNeighborsClassifier(n_neighbors=11)
knn_model_11.fit(X_train, y_train)

# Generate predictions
knn_predictions_11  = knn_model_11.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, knn_predictions_11))
print(classification_report(y_test, knn_predictions_11,zero_division=0))

#heat map visualization on random forest confusion matrix
plot_confusion_matrix(y_test, knn_predictions_11, title="KNN Confusion Matrix (k=11)")
#--------------------------------------------------------------------------------------------------
print("****************************KNN***********************************")
knn_model_7 = KNeighborsClassifier(n_neighbors=7)
knn_model_7.fit(X_train, y_train)

# Generate predictions
knn_predictions_7 = knn_model_7.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, knn_predictions_7))
print(classification_report(y_test, knn_predictions_7,zero_division=0))

#heat map visualization on random forest confusion matrix
plot_confusion_matrix(y_test, knn_predictions_7, title="KNN Confusion Matrix(k=7)")
#-----------------------------------------------------------------------------------------------
print("****************************KNN***********************************")
knn_model_3 = KNeighborsClassifier(n_neighbors=3)
knn_model_3.fit(X_train, y_train)

# Generate predictions
knn_predictions_3 = knn_model_3.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, knn_predictions_3))
print(classification_report(y_test, knn_predictions_3,zero_division=0))

#heat map visualization on random forest confusion matrix
plot_confusion_matrix(y_test, knn_predictions_3, title="KNN Confusion Matrix(k=3)")