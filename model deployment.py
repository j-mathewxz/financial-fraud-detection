import numpy as np
import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings('ignore')

# 1. Load the cleaned data
data = pd.read_csv('Financial_Transactions.csv')

# 2. Separate Features (x) and Target (y)
x = data.drop(['isFraud'], axis=1)
y = data['isFraud']

# 3. Split and Scale
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)  # Use transform, not fit_transform on test

# 4. Build and Train the Neural Network (MLP)
model = MLPClassifier(max_iter=1000, alpha=1)
model.fit(x_train_scaled, y_train)

print("Model Accuracy on Test Set:", model.score(x_test_scaled, y_test))


# 5. Define the Prediction Function for the GUI
def detect_fraud(step, amount, amount_cat, old_bal_org, new_bal_org, old_bal_dest, new_bal_dest, out, debit, pay,
                 transfer):
    # Create the array from inputs
    features = np.array(
        [step, amount, amount_cat, old_bal_org, new_bal_org, old_bal_dest, new_bal_dest, out, debit, pay, transfer])

    # Scale the input just like the training data
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Predict
    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        return "⚠️ FRAUD DETECTED"
    else:
        return "✅ TRANSACTION LEGITIMATE"


# 6. Build the Gradio Interface
# We have 11 features in the cleaned dataset
inputs = [
    'number', 'number', 'number', 'number', 'number',
    'number', 'number', 'checkbox', 'checkbox', 'checkbox', 'checkbox'
]

app = gr.Interface(
    fn=detect_fraud,
    inputs=inputs,
    outputs=gr.Textbox(),
    title="Financial Fraud Detection System",
    description="Enter transaction details below to check for potential fraud using the MLP Neural Network model."
)

app.launch(share=True)