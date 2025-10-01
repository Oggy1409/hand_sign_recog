import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
data_dict = pickle.load(open('data.pickle','rb'))
X = np.array(data_dict['data'])
y = np.array(data_dict['labels'])

# Encode labels (A,B,C,... → 0,1,2,...)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, shuffle=True, stratify=y_encoded
)

# RandomForest
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(f'{score*100:.2f}% of samples classified correctly!')

# Save model + label encoder
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'le': le}, f)

print("Model saved as 'model.p'")
