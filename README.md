🫀 Heart Disease Prediction using Logistic Regression
This project builds a machine learning model to predict the likelihood of heart disease in patients based on medical attributes such as age, blood pressure, cholesterol levels, chest pain type, etc. The model is trained using Logistic Regression and evaluated on accuracy, confusion matrix, and classification report.

📂 Project Structure
bash
Copy
Edit
├── data/
│   └── heart.csv              # Dataset
├── models/
│   └── heart_model.pkl        # Trained Logistic Regression model (generated after training)
├── heart_disease_prediction.py # Main script
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
🚀 Features
Preprocesses dataset (encodes categorical variables).

Splits data into train/test sets.

Trains a Logistic Regression classifier.

Evaluates the model using accuracy, confusion matrix, and classification report.

Saves trained model (heart_model.pkl) with joblib.

Allows prediction for new patient data (with confidence percentage).

📊 Example Output
After training and testing, you will see results like:

yaml
Copy
Edit
Model Accuracy: 85.25%

Classification Report:
              precision    recall  f1-score   support
           0       0.87      0.83      0.85       123
           1       0.84      0.87      0.86       126
📑 Dataset
The dataset used is Heart Disease dataset, available publicly (e.g., Kaggle or UCI).
It contains patient medical attributes and a target label HeartDisease (0 = No, 1 = Yes).
