import os
import pickle
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Load dataset (matches the 30 features from the notebook)
data = load_breast_cancer()
X = data.data
# 1 = malignant, 0 = benign
y = np.where(data.target == 0, 1, 0)

# 2. Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# 3. Fit scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4. Train Regularized Logistic Regression (C=0.1)
model = LogisticRegression(penalty='l2', C=0.1, max_iter=1000)
model.fit(X_train_scaled, y_train)

# 5. Save BOTH model and fitted scaler into deploy.pkl
current_dir = os.path.dirname(__file__)
pkl_path = os.path.join(current_dir, 'deploy.pkl')

with open(pkl_path, 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)

print(f"✅ deploy.pkl created successfully at: {pkl_path}")
