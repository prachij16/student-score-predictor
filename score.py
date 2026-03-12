import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("dataset.csv")
print(data.columns)

# Features and target
X = data[['Hours']]
y = data['Score']

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict score
hours = float(input('enter study hours:'))
prediction = model.predict([[hours]])

print("Predicted Score for hours study:", prediction[0])

# Visualization
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Student Score Predictor")
plt.show()