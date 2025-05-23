import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

#Note-The .pkl files are made from running this program, therefore run this program first before running 'main.py'
#Only three files are necessary for 'main.py','train_model.py','train.csv', the pkl files will be made by 'train_model.py' and be used to run 'main.py' without loading it each time

df = pd.read_csv("train.csv")

df['Gender'] = df['Gender'].fillna('Male')
df['Married'] = df['Married'].fillna('Yes')
df['Dependents'] = df['Dependents'].replace('3+', 3)
df['Dependents'] = df['Dependents'].fillna(0)
df['Self_Employed'] = df['Self_Employed'].fillna('No')
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(1.0)

# Drop Loan_ID, because to us, it is basically useless for training or testing data
if 'Loan_ID' in df.columns:
    df = df.drop(columns=["Loan_ID"])

le_dict = {}
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le  


X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(" Accuracy on validation set:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

joblib.dump(model, "loan_model.pkl")
joblib.dump(le_dict, "encoders.pkl")
