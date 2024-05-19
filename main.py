# # symptom_checker_ml.py

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline


# data = {
#     'symptom1': ['fever', 'fever', 'headache', 'fatigue', 'cough'],
#     'symptom2': ['cough', 'cough', 'fatigue', 'headache', 'fever'],
#     'symptom3': ['headache', 'fatigue', 'sore throat', 'sore throat', 'headache'],
#     'condition': ['Flu', 'COVID-19', 'Cold', 'Chronic Fatigue Syndrome', 'Malaria']
# }

# df = pd.DataFrame(data)

# df['all_symptoms'] = df['symptom1'] + ' ' + df['symptom2'] + ' ' + df['symptom3']
# X = df['all_symptoms']
# y = df['condition']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create a pipeline that combines CountVectorizer and MultinomialNB
# pipeline = Pipeline([
#     ('vectorizer', CountVectorizer()),
#     ('classifier', MultinomialNB())
# ])

# # Train the model
# pipeline.fit(X_train, y_train)

# # Function to predict condition based on symptoms
# def predict_condition(symptoms):
#     return pipeline.predict([symptoms])[0]

# def main():
    
#     print("Welcome to the Symptom Checker!")
#     symptoms = input("Enter your symptoms separated by spaces (e.g., fever cough headache): ")
    
#     # # Predict the condition
#     condition = predict_condition(symptoms)
    
#     # Print the diagnosis
#     print(f"\nBased on your symptoms, the possible condition is: {condition}")
#     medicine = {
#     'Flu': 'Tamiflu',
#     'COVID-19': 'Remdesivir',
#     'Cold': 'Coldrex',
#     'Chronic Fatigue Syndrome': 'Vitamin B12 supplements',
#     'Malaria': 'Chloroquine'
# }
#     if condition in medicine:
#         print(f"The recommended medicine for {condition} is: {medicine[condition]}")
#     else:
#         print("No specific medicine recommendation for this condition.")

# if __name__ == '__main__':
#     main()
# ////////////
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# df= pd.read_csv("da.csv")
# # print(df)
# # shape= df.shape
# # print(shape)
# # shows data in arry
# # print(df.values)
# X_price = df[['age']]
# y_price = df['price']

# model_price = DecisionTreeClassifier()
# model_price.fit(X_price, y_price)
# X_review = df[['price']]
# y_review = df['review']
# model_review = DecisionTreeClassifier()
# model_review.fit(X_review, y_review)
# age= input("enter age: ")
# def predict_price_and_review(age):

#     input_data_price = pd.DataFrame({'age': [age]})
#     predicted_price = model_price.predict(input_data_price)[0]
    

#     input_data_review = pd.DataFrame({'price': [predicted_price]})
#     predicted_review = model_review.predict(input_data_review)[0]
    
#     return predicted_price, predicted_review


# predicted_price = predict_price_and_review(age)
# price , review =predicted_price
# print(f"The predicted price for age {age} is {price} and review is {review}")
# ////////

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score

# # Read the dataset
# df = pd.read_csv("sympt.csv")

# # Separate features and target variable
# x = df.drop(columns=['Disease', 'Outcome Variable'])
# y = df['Outcome Variable']

# # # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline

# df=pd.read_csv("dataset.csv")
# symptom_columns = [f'Symptom_{i}' for i in range(1, 18)]
# df['All_Symptoms'] = df[symptom_columns].fillna('').apply(lambda row: ' '.join(row).strip(), axis=1)
# df.drop_duplicates(subset=['All_Symptoms', 'Disease'], inplace=True)
# x = df['All_Symptoms']
# y = df['Disease']
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# pipeline = Pipeline([
#     ('vectorizer', CountVectorizer()),
#     ('classifier', MultinomialNB())
# ])

# pipeline.fit(x_train, y_train)
# def predict_condition(symptoms):
#     return pipeline.predict([symptoms])[0]

# print(predict_condition( "fever,vomiting,head hurts"))
# print(f"Training accuracy: {pipeline.score(x_train, y_train)}")
# print(f"Testing accuracy: {pipeline.score(x_test, y_test)}")

# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score
# import joblib

# # Read the dataset
# df = pd.read_csv("dataset.csv")

# # Combine all symptom columns into a single column of text data
# symptom_columns = [f'Symptom_{i}' for i in range(1, 18)]
# df['All_Symptoms'] = df[symptom_columns].fillna('').apply(lambda row: ' '.join(row).strip(), axis=1)

# # Define features and target
# x = df['All_Symptoms']
# y = df['Disease']

# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=42)

# # Define the pipeline
# pipeline = Pipeline([
#     ('vectorizer', CountVectorizer()),
#     ('classifier', MultinomialNB())
# ])

# # Define hyperparameters for GridSearchCV
# param_grid = {
#     'vectorizer__ngram_range': [(1, 1), (1, 2)],
#     'classifier__alpha': [0.1, 0.5, 1.0, 5.0, 10.0]
# }

# # Use GridSearchCV to find the best parameters
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(x_train, y_train)

# # Get the best estimator
# best_pipeline = grid_search.best_estimator_

# # Predict and evaluate the model
# y_pred_train = best_pipeline.predict(x_train)
# y_pred_test = best_pipeline.predict(x_test)

# print(f"Training accuracy: {accuracy_score(y_train, y_pred_train)}")
# print(f"Testing accuracy: {accuracy_score(y_test, y_pred_test)}")

# # Function to preprocess symptoms
# def preprocess_symptoms(symptoms):
#     return ' '.join(symptoms.strip().lower().split())

# # Define a function to predict conditions with probabilities
# def predict_condition(symptoms, top_n=1):
#     preprocessed_symptoms = preprocess_symptoms(symptoms)
#     probabilities = best_pipeline.predict_proba([preprocessed_symptoms])[0]
#     classes = best_pipeline.classes_
#     sorted_indices = probabilities.argsort()[::-1]
#     top_classes = classes[sorted_indices][:top_n]
#     top_probabilities = probabilities[sorted_indices][:top_n]
#     return list(zip(top_classes, top_probabilities))

# # Example prediction
# example_symptoms = "fever,Cough, sneezing,vomiting, body hurts"
# predictions = predict_condition(example_symptoms, top_n=1)

# print(f"Predictions for '{example_symptoms}':")
# for disease, probability in predictions:
#     print(f"{disease}: {probability:.4f}")
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from flask import Flask, request, jsonify
app = Flask(__name__)

df = pd.read_csv("dataset.csv")


symptom_columns = [f'Symptom_{i}' for i in range(1, 18)]
df['All_Symptoms'] = df[symptom_columns].fillna('').apply(lambda row: ' '.join(row).strip(), axis=1).str.replace('_',' ')


df.drop_duplicates(subset=['All_Symptoms', 'Disease'], inplace=True)


x = df['All_Symptoms']
y = df['Disease']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', RandomForestClassifier())
])


param_grid = {
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30]
}


cv = StratifiedKFold(n_splits=3)


grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy')
grid_search.fit(x_train, y_train)


best_pipeline = grid_search.best_estimator_


y_pred_train = best_pipeline.predict(x_train)
y_pred_test = best_pipeline.predict(x_test)

print(f"Training accuracy: {accuracy_score(y_train, y_pred_train)}")
print(f"Testing accuracy: {accuracy_score(y_test, y_pred_test)}")


def preprocess_symptoms(symptoms):
    return ' '.join(symptoms.strip().lower().split())


def predict_condition(symptoms, top_n=1):
    preprocessed_symptoms = preprocess_symptoms(symptoms)
    probabilities = best_pipeline.predict_proba([preprocessed_symptoms])[0]
    classes = best_pipeline.classes_
    sorted_indices = probabilities.argsort()[::-1]
    top_classes = classes[sorted_indices][:top_n]
    top_probabilities = probabilities[sorted_indices][:top_n]
    return list(zip(top_classes, top_probabilities))


@app.route('/predict', methods=['POST'])
def predict():

    data = request.json
    
    if not data or 'symptoms' not in data:
        return jsonify({'error': 'Invalid request'}), 400

    symptoms = data['symptoms']
    predictions = predict_condition(symptoms, top_n=2)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=8080)