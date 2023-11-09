import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load the Iris dataset and create a DataFrame
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

# Images of Iris Sepal and Petal
st.image("iris_petal_sepal.png", width=450)

# Visualizations
st.title('Iris Dataset Exploration and K-Nearest Neighbors Classifier')

# Display dataset summary and first few rows
st.subheader('Iris Dataset Summary')
st.write(df.iloc[0:141:20, :])

# Data Visualization - Scatter Plot
st.subheader('Scatter Plot of Sepal and Petal Measurements')
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='flower_name', palette='Set1', ax=ax)
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='flower_name', palette='Set2', ax=ax)
st.pyplot(fig)

# Model Building and Evaluation
st.subheader('K-Nearest Neighbors Classifier')

# Train-test split
X = df.drop(['target', 'flower_name'], axis=1)
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Fit the KNN model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
st.write(f'Model Accuracy: {accuracy*100:.2f}')

# Confusion Matrix
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Visualize Confusion Matrix
st.subheader('Confusion Matrix')
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Truth')
st.pyplot(fig)

# Classification Report
st.subheader('Classification Report')
report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
df_report = pd.DataFrame(report).transpose()
st.write(df_report[['precision', 'recall', 'f1-score', 'support']])

# Prediction
st.sidebar.header('Make Predictions')
input_data = []
for feature in iris.feature_names:
    val = st.sidebar.number_input(f'Enter {feature}', step=0.1)
    input_data.append(val)

if st.sidebar.button('Predict'):
    prediction = knn.predict([input_data])
    st.sidebar.write(f'Predicted Class: {iris.target_names[prediction[0]]}')
    if prediction[0] == 0:
        st.sidebar.image("setosa.jpg", caption="Setosa", use_column_width=True)
    elif prediction[0] == 1:
        st.sidebar.image("versicolor.jpg", caption="Versicolor", use_column_width=True)
    else:
        st.sidebar.image("virginica.jpg", caption="Virginica", use_column_width=True)
