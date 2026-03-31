import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# load dataset
df = pd.read_csv("Bank_Churn.csv")

# drop unimortant column
df = df.drop(columns=["CustomerId","Surname","Geography"])


# split dataset traing and test
X = df.drop('Exited',axis=1)
y = df['Exited']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=43)

numerical_features = ['CreditScore','Age','Tenure','Balance','EstimatedSalary']
categorical_features = ['Gender']


# make pipeline
preproessesing = ColumnTransformer([
    ('scale', StandardScaler(),numerical_features),
    ('encoding',OneHotEncoder(drop='first'),categorical_features)
])

pipe = Pipeline([
    ('preprocessing',preproessesing),
    ('classifier',LogisticRegression(class_weight='balanced', random_state=42))
])

# train model 
pipe.fit(X_train,y_train)

# pridect test data
y_pred = pipe.predict(X_test)

# accueacy of the model
print(accuracy_score(y_test,y_pred))
# report of model
print(classification_report(y_test, y_pred))

# store the model
joblib.dump(pipe, "bank_churn.pkl")
