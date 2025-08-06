import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("rba-dataset.csv")
df.columns = df.columns.str.strip()

target = "Is Attack IP"

# excluding copy tuples with NaN in the target column
print(f"\n[{target}]")
print(df[target].value_counts(dropna=False))

# casting to string / boolean
if df[target].dtype == object:
    df[target] = df[target].str.lower().map({"true": True, "false": False})

# fifty-fiftying
df_true = df[df[target] == True]
df_false = df[df[target] == False].sample(n=len(df_true), random_state=42)

# messing up the order
df_balanced = pd.concat([df_true, df_false]).sample(frac=1).reset_index(drop=True)

print(f"\n✅ Dataset balanceado com {len(df_true)} ataques e {len(df_false)} normais.\n")

# pre-processing
features = [
    "Country", "Region", "City", "ASN",
    "OS Name and Version", "Browser Name and Version",
    "Device Type", "Round-Trip Time [ms]",
    "Login Successful"
]

df_balanced = df_balanced[features + [target]].dropna()

# Coodify categorical features
label_encoders = {}
for col in df_balanced.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_balanced[col] = le.fit_transform(df_balanced[col])
    label_encoders[col] = le

# Separate features and target
X = df_balanced[features]
y = df_balanced[target]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Aval
y_pred = model.predict(X_test)

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

importances = model.feature_importances_
feat_names = X.columns

plt.figure(figsize=(10, 5))
plt.barh(feat_names, importances)
plt.title("Importância das Features")
plt.xlabel("Peso")
plt.tight_layout()
plt.show()
