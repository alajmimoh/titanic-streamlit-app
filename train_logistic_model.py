
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# تحميل البيانات
data_path = "C:/Users/D/Downloads/cleaned_data.csv"
df = pd.read_csv(data_path)

# تعويض القيم المفقودة
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# حذف الأعمدة غير المفيدة
df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

# تحويل النصوص إلى أرقام
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# فصل المدخلات والمخرجات
X = df.drop(columns=['Survived'])
y = df['Survived']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# إنشاء وتدريب نموذج Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# التقييم
y_pred = model.predict(X_test)
print("📊 تقرير النموذج:")
print(classification_report(y_test, y_pred))

# حفظ النموذج
model_path = "C:/Users/D/Downloads/logistic_model.pkl"
joblib.dump(model, model_path)
print("✅ تم حفظ سكربت التدريب في مجلد التنزيلات.")
