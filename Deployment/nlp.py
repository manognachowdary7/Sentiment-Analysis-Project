import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image

# Load the Decision Tree model
model_path = 'best_classifier_decision_tree.pkl'
with open(model_path, 'rb') as file:
    best_classifier_decision_tree = pickle.load(file)

# Load the CountVectorizer used during training
vectorizer_path = 'vectorizer.pkl'
with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)

# Function to preprocess input text
def preprocess_text(text):
    # Implement your text preprocessing logic here
    return text

# Streamlit UI
st.title('Sentiment Analysis with Decision Tree Model')

# Input text box for the user to enter a review
user_input = st.text_area('Enter a review:', '')

# Preprocess the input text
preprocessed_input = preprocess_text(user_input)

# Make predictions on user input
if st.button('Predict'):
    if not user_input or not vectorizer.transform([preprocessed_input]).nnz:
        st.warning('Please enter a valid review. 🤔✍️')
    else:
        try:
            # Vectorize the preprocessed text using the loaded vectorizer
            user_input_vectorized = vectorizer.transform([preprocessed_input])

            # Make a prediction using the loaded model
            prediction = best_classifier_decision_tree.predict(user_input_vectorized)

            # Display the prediction label
            sentiment_labels = ['negative', 'neutral', 'positive']

            if prediction[0] in [0, 1, 2]:  # Check if the prediction is within the expected range
                st.write('Prediction:', sentiment_labels[prediction[0]])

                # Show a local image based on sentiment
                if prediction[0] == 0:  # Negative
                    image_path = r"C:\Users\ds_sa\Desktop\Project_P340\Gif\0.jpg"
                    caption = "Sad___Negative___😥"
                elif prediction[0] == 1:  # Neutral
                    image_path = r"C:\Users\ds_sa\Desktop\Project_P340\Gif\1.jpg"
                    caption = "Silent___Neutral___😐"
                elif prediction[0] == 2:  # Positive
                    image_path = r"C:\Users\ds_sa\Desktop\Project_P340\Gif\2.jpg"
                    caption = "Always___Positive___☺️"

            # Open the local image using Pillow (PIL)
            image = Image.open(image_path)

            # Display the image
            st.image(image, caption=f"Your Sentiment Image: {caption}", use_column_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
