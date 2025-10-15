import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Memuat dataset
df = pd.read_csv('UCI_Credit_Card.csv')
print(df.info())
print(df.head())

# Cek missing value
print(df.isnull().sum())

# Visualisasi distribusi target
sns.countplot(x='default.payment.next.month', data=df)
plt.title('Distribusi Nasabah: Tidak Gagal Bayar vs Gagal Bayar')
plt.show()

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X = df.drop('default.payment.next.month', axis=1)
y = df['default.payment.next.month']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Terapkan SMOTE
smote = SMOTE(random_state=42, k_neighbors=1)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Scaling fitur numerik
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Model Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train_res)


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:,1]

print(classification_report(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Tidak Gagal Bayar', 'Gagal Bayar'])
disp.plot(cmap='Blues')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='orange', label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='darkblue')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Curve')
plt.show()


import joblib
joblib.dump(model, 'model_logreg.pkl')
joblib.dump(scaler, 'scaler.pkl')
