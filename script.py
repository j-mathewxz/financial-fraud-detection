import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df_clean = pd.read_csv('Financial_Transactions.csv')


# 1. Statistical Summary
summary_stats = df_clean.describe()
print("Statistical Summary:")
print(summary_stats)

# 2. Fraud Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='isFraud', data=df_clean, palette='viridis')
plt.title('Distribution of Fraudulent vs Non-Fraudulent Transactions')
plt.xlabel('Is Fraud? (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.savefig('fraud_distribution.png')
plt.close()

# 3. Transaction Type vs Fraud
plt.figure(figsize=(10, 6))
sns.countplot(x='type', hue='isFraud', data=df_clean, palette='magma')
plt.title('Fraud Occurrence by Transaction Type')
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.savefig('fraud_by_type.png')
plt.close()

# model building
# install sckikit-learn package
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Prepare features and target
X = df_clean.drop('isFraud', axis=1)
y = df_clean['isFraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 1. Logistic Regression (Baseline)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# 2. Random Forest (Advanced)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("Logistic Regression Results:")
print(classification_report(y_test, lr_preds))

print("\nRandom Forest Results:")
print(classification_report(y_test, rf_preds))
