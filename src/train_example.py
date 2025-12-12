# Simple ML training script using scikit-learn
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv('../data/sample_data.csv')
x = data[['feature1', 'feature2']]
y = data['label']

# Initialize and train model
model = DecisionTreeClassifier()
model.fit(x, y)

# Predict on training data
predictions = model.predict(x)
print("Predictions on training data:", predictions)

