# Random Forest Classification to make Feature Importances

import pandas as pd
import numpy as np
import sklearn as sk

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("mental_health_remote_workers.csv")

# map some of the qualitative data
data['Mental_Health_Status'] = data['Mental_Health_Status'].map({'Poor':0.1, 'Moderate':0.5, 'Good': 1.0, None:0.0})
data['Internet_Issues_Frequency'] = data['Internet_Issues_Frequency'].map({'Sometimes': 0.5, 'Often': 1.0, 'Never': 0.0, None: 0.0})

categories = ['Work_Mode']
labelEncoder = LabelEncoder()

for cat in categories:
    data[cat] = labelEncoder.fit_transform(data[cat])

X = data[['Mental_Health_Status', 'Hours_Worked_Per_Week', 'Work_Mode', 'Remote_Setup_Satisfaction', 'Internet_Issues_Frequency']]
y = data['Burnout_Score']

y = labelEncoder.fit_transform(y)

# split data
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=42)

# training a model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=True)

importances_df = pd.DataFrame({
    'Feature' : X.columns,
    'Importance' : rf.feature_importances_
}).sort_values('Importance', ascending=False)

importances_df.to_csv('feature_importances.csv', index=False)
