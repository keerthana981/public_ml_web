import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random

app = Flask(__name__)

# Define the GymRecommendationModel class (must match the class used when pickling)
class GymRecommendationModel:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.data = None
        self.user_features_cols = ['Sex', 'Age', 'Height', 'Weight', 'Hypertension', 'Diabetes', 'BMI', 'Level', 'Fitness Goal', 'Fitness Type']
        self.output_cols = ['Exercises', 'Diet', 'Equipment']

    def fit(self, data):
        """Preprocess the data (encode categorical features and scale numerical features)."""
        # Drop unnecessary columns
        self.data = data.drop(columns=['ID'], errors='ignore')

        # Encode categorical features
        categorical_cols = ['Sex', 'Hypertension', 'Diabetes', 'Level', 'Fitness Goal', 'Fitness Type']
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            self.data[col] = self.label_encoders[col].fit_transform(self.data[col])

        # Scale numerical features
        numerical_cols = ['Age', 'Height', 'Weight', 'BMI']
        self.data[numerical_cols] = self.scaler.fit_transform(self.data[numerical_cols])

    def _preprocess_user_input(self, user_input):
        """Preprocess user input to match the model's data format."""
        user_df = pd.DataFrame([user_input], columns=self.user_features_cols)
        
        # Scale numerical features
        numerical_cols = ['Age', 'Height', 'Weight', 'BMI']
        user_df[numerical_cols] = self.scaler.transform(user_df[numerical_cols])
        
        return user_df

    def predict(self, user_input, top_n=3):
        """Generate personalized workout and diet recommendations based on user input."""
        # Preprocess user input
        user_df = self._preprocess_user_input(user_input)

        # Calculate similarity scores
        user_features = self.data[self.user_features_cols]
        similarity_scores = cosine_similarity(user_features, user_df).flatten()

        # Retrieve top similar users and get the first recommendation
        similar_user_indices = similarity_scores.argsort()[-5:][::-1]
        similar_users = self.data.iloc[similar_user_indices]
        recommendation_1 = similar_users[self.output_cols].mode().iloc[0].to_dict()

        # Simulate two additional recommendations by modifying input values
        simulated_recommendations = []
        for _ in range(2):
            modified_input = user_input.copy()
            modified_input['Age'] += random.randint(-5, 5)
            modified_input['Weight'] += random.uniform(-5, 5)
            modified_input['BMI'] += random.uniform(-1, 1)

            # Preprocess modified input
            modified_user_df = self._preprocess_user_input(modified_input)

            # Calculate similarity scores for modified input
            modified_similarity_scores = cosine_similarity(user_features, modified_user_df).flatten()
            modified_similar_user_indices = modified_similarity_scores.argsort()[-5:][::-1]
            modified_similar_users = self.data.iloc[modified_similar_user_indices]
            recommendation = modified_similar_users[self.output_cols].mode().iloc[0].to_dict()

            # Ensure unique recommendations
            if not any(
                rec['Exercises'] == recommendation['Exercises'] and
                rec['Diet'] == recommendation['Diet'] and
                rec['Equipment'] == recommendation['Equipment']
                for rec in simulated_recommendations
            ):
                simulated_recommendations.append(recommendation)

        return [recommendation_1] + simulated_recommendations

# Load the pickled GymRecommendationModel
with open('/Users/keerthanagc/Downloads/AI/model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form inputs
    user_input = {
        'Sex': int(request.form['Sex']),
        'Age': float(request.form['Age']),
        'Height': float(request.form['Height']),
        'Weight': float(request.form['Weight']),
        'Hypertension': int(request.form['Hypertension']),
        'Diabetes': int(request.form['Diabetes']),
        'BMI': float(request.form['BMI']),
        'Level': int(request.form['Level']),
        'Fitness Goal': int(request.form['Fitness Goal']),
        'Fitness Type': int(request.form['Fitness Type'])
    }

    # Get recommendations
    recommendations = model.predict(user_input, top_n=3)

    # Pass recommendations to the template
    return render_template('index.html', 
                         recommendations=recommendations,
                         predict_text="Here are your personalized workout and diet recommendations:")

if __name__ == '__main__':
    app.run(debug=True)