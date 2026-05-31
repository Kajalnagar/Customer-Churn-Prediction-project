
# CUSTOMER CHURN PREDICTION
# COMPLETE CAPSTONE PROJECT


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


# LOAD DATASET


print("\nLoading Dataset...\n")

df = pd.read_csv("customer_churn.csv")

print(df.head())

print("\nDataset Shape:")
print(df.shape)


# DATA INFORMATION


print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())


# MISSING VALUES


print("\nMissing Values:")
print(df.isnull().sum())

for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)


# DUPLICATES


print("\nDuplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)


# EDA


plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Contract vs Churn")
plt.show()

plt.figure(figsize=(10,5))
sns.countplot(x="PaymentMethod", hue="Churn", data=df)
plt.title("Payment Method vs Churn")
plt.xticks(rotation=20)
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(x="Churn", y="TotalCharges", data=df)
plt.title("Total Charges vs Churn")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["Tenure"], bins=20, kde=True)
plt.title("Tenure Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="SeniorCitizen", hue="Churn", data=df)
plt.title("Senior Citizen Analysis")
plt.show()


# PREPROCESSING


if "CustomerID" in df.columns:
    df.drop("CustomerID", axis=1, inplace=True)

encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])


# CORRELATION


plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.show()


# FEATURES & TARGET


X = df.drop("Churn", axis=1)

y = df["Churn"]


# TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# SCALING


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# LOGISTIC REGRESSION


print("\nTraining Logistic Regression...\n")

lr = LogisticRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print(
    "Logistic Regression Accuracy:",
    accuracy_score(y_test, lr_pred)
)


# RANDOM FOREST


print("\nTraining Random Forest...\n")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print(
    "Random Forest Accuracy:",
    accuracy_score(y_test, rf_pred)
)


# CLASSIFICATION REPORT


print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        rf_pred
    )
)


# CONFUSION MATRIX


cm = confusion_matrix(
    y_test,
    rf_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# ROC CURVE


prob = rf.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    prob
)

plt.figure(figsize=(7,5))

plt.plot(fpr, tpr)

plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.show()


# ROC AUC


auc = roc_auc_score(
    y_test,
    prob
)

print("\nROC AUC Score:", auc)


# FEATURE IMPORTANCE


importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:\n")

print(importance)

plt.figure(figsize=(8,5))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance
)

plt.title("Feature Importance")

plt.show()


# SAVE MODEL


joblib.dump(
    rf,
    "model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("\nModel Saved Successfully")


# SAMPLE PREDICTION


sample = pd.DataFrame({

    "Tenure":[24],
    "MonthlyCharges":[120],
    "TotalCharges":[3000],
    "Contract":[1],
    "PaymentMethod":[0],
    "PaperlessBilling":[1],
    "SeniorCitizen":[0]

})

sample_scaled = scaler.transform(sample)

prediction = rf.predict(sample_scaled)

print("\nSample Prediction:", prediction)

if prediction[0] == 1:
    print("Customer Likely To Churn")
else:
    print("Customer Likely To Stay")

print("\nProject Completed Successfully")
