import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from time import time
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'C:/Users/Asus/Desktop/data Cleaning/train.csv'
df = pd.read_csv(file_path, encoding='ascii')

df['Age'].fillna(df['Age'].mean(), inplace=True)

df = pd.get_dummies(df, columns=['Sex'], drop_first=True)


targets = ['Pclass', 'Sex_male', 'Age', 'Fare']
X = df[targets]
y = df['Survived']

# Standardizing the features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Applying PCA
pca = PCA(n_components=0.95) # Retain 95% of variance
X_pca = pca.fit_transform(X_scaled)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Initializing the Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Calculating accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print('Accuracy: ', accuracy)
print('Confusion Matrix:\
', conf_matrix)
print('Classification Report:\
', class_report)

print(f'Predictions: \n{y_pred}')

testdf = pd.read_csv('C:/Users/Asus/Desktop/data Cleaning/test.csv')
actualdf = pd.read_csv('C:/Users/Asus/Desktop/data Cleaning/gender_submission.csv')

testdf['Age'].fillna(testdf['Age'].mean(), inplace=True)
testdf['Fare'].fillna(testdf['Fare'].mean(), inplace=True)

testdf = pd.get_dummies(testdf, columns=['Sex'], drop_first=True)
print(testdf[['PassengerId', 'Sex_male']])

targets = ['Pclass', 'Sex_male', 'Age', 'Fare']

actuals = actualdf['Survived']

X_test_scaled = scaler.transform(testdf[targets])
X_test_pca = pca.transform(X_test_scaled)

#get predictions on the testing dataset
test_predictions = model.predict(X_test_pca)
accuracy_test = accuracy_score(actuals, test_predictions)

print(f'Trainset Accuracy: {accuracy}')
print(f'Testset Accuracy: {accuracy_test}')
print(f'Predictions: \n{test_predictions}')

pred_results = pd.DataFrame({'PassengerId': testdf['PassengerId'], 'Name': testdf['Name'], 'Age': testdf['Age'], 'Pclass': testdf['Pclass'], 'Sex': testdf['Sex_male'], 'Survived': test_predictions})
pred_results['Sex'] = pred_results['Sex'].map({True: 'Male', False: 'Female'})
pred_results.to_csv('C:/Users/Asus/Desktop/data Cleaning/titanic_predictions.csv', index=False)

# Distribution of predictions for survival status
plt.figure(figsize=(8, 5))
sns.countplot(x=y_pred)
plt.title('Distribution of Predictions for Survival Status')
plt.xlabel('Survival Status')
plt.ylabel('Count')
plt.show()

# Survival rate by Gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Sex_male', data=testdf)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['Not Survived', 'Survived'])
plt.show()

# Survival rate by Passenger Class
plt.figure(figsize=(8, 6))
sns.countplot(x='Pclass', data=testdf)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['Not Survived', 'Survived'])
plt.show()

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Accuracy
plt.figure(figsize=(6, 4))
sns.barplot(x=['Trainset', 'Testset'], y=[accuracy, accuracy_test])
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0)
plt.show()

