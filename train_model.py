import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
df = pd.read_csv('train.csv')

# Keep relevant columns
features = ['OverallQual', 'GrLivArea', 'YearBuilt', 'GarageCars', 'Neighborhood']
df = df[features + ['SalePrice']].dropna()

# Encode categorical 'Neighborhood'
le = LabelEncoder()
df['Neighborhood'] = le.fit_transform(df['Neighborhood'])

# Features and target
X = df[features]
y = df['SalePrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and label encoder
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(le, open('label_encoder.pkl', 'wb'))

print("✅ Model and label encoder saved.")
