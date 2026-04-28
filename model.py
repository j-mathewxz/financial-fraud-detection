import pandas as pd
df_clean = pd.read_csv('Financial_Transactions.csv')

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


