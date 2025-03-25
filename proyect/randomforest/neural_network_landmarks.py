import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load data
data_dict = pickle.load(open('data.pickle', 'rb'))

# Extract data
data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Ensure all elements in data have the same shape
# Find the maximum length of sequences
max_length = max(len(seq) for seq in data)

# Pad sequences to the same length
padded_data = np.array([np.pad(seq, (0, max_length - len(seq)), 'constant') for seq in data])

# Standardize the data
scaler = StandardScaler()
padded_data = scaler.fit_transform(padded_data)

# Split data
x_train, x_test, y_train, y_test = train_test_split(padded_data, labels, # arrays to split
                                                    test_size=0.2,  # 80% train, 20% test
                                                    shuffle=True, stratify=labels, random_state=42) # shuffle data

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Predict and evaluate
y_predict = model.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_predict)*100}')

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler, 'max_length': max_length}, f)