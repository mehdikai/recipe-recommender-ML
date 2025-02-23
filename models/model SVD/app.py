import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess dataset
@st.cache_data
def load_and_preprocess(file_path):
    """Load, clean, and prepare the dataset for recommendations."""
    try:
        df = pd.read_csv(file_path)
        st.success(f"✅ Successfully loaded {len(df)} recipes.")
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

    # Cleaning functions
    def clean_name(text):
        return re.sub(r'[^\w\s]', '', str(text).lower()).strip()
    
    def clean_ingredients(text):
        """Standardizes ingredient text: removes quantities, formats names."""
        text = str(text).lower()
        ingredients = re.split(r',|\s+and\s+', text)
        cleaned = [re.sub(r'\d+[^\s]*', '', ing).strip().replace(' ', '_') for ing in ingredients if ing.strip()]
        return cleaned

    # Apply cleaning
    df['cleaned_name'] = df['name'].apply(clean_name)
    df['cleaned_ingredients'] = df['ingredients_name'].apply(clean_ingredients)

    # Create text representations
    df['name_text'] = df['cleaned_name']
    df['ingredient_text'] = df['cleaned_ingredients'].apply(' '.join)

    return df


# SVD Recommender Model
class SVDRecommender:
    """Performs recipe recommendations using SVD + cosine similarity."""
    
    def __init__(self, n_components=50):
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.terms = []
        self.term_indices = {}
    
    def create_term_matrix(self, texts):
        """Builds a binary term-document matrix for ingredient-based similarity."""
        self.terms = list(set(" ".join(texts).split()))
        self.term_indices = {term: idx for idx, term in enumerate(self.terms)}

        matrix = np.zeros((len(texts), len(self.terms)))
        for i, text in enumerate(texts):
            for term in text.split():
                if term in self.term_indices:
                    matrix[i, self.term_indices[term]] = 1
        return matrix
    
    def fit(self, texts):
        term_matrix = self.create_term_matrix(texts)
        self.svd.fit(term_matrix)
    
    def transform(self, text):
        vec = np.zeros((1, len(self.terms)))
        for term in text.split():
            if term in self.term_indices:
                vec[0, self.term_indices[term]] = 1
        return self.svd.transform(vec)
    
    def recommend(self, query, dataset, top_n=5):
        """Finds top N similar recipes based on query."""
        query_vec = self.transform(query)
        doc_vectors = self.svd.transform(self.create_term_matrix(dataset))
        similarities = cosine_similarity(query_vec, doc_vectors)[0]
        indices = np.argsort(similarities)[-top_n:][::-1]
        return dataset.iloc[indices], similarities[indices]


# Load dataset
file_path = '/content/Food_Recipe.csv'  # Adjust if necessary
dataset = load_and_preprocess(file_path)

if dataset is not None:
    # Train models
    name_model = SVDRecommender(n_components=50)
    ingredient_model = SVDRecommender(n_components=100)
    
    name_model.fit(dataset['name_text'])
    ingredient_model.fit(dataset['ingredient_text'])

    # Streamlit UI
    st.title("🍽 Recipe Recommendation System")
    st.sidebar.header("🔍 Search Options")
    
    option = st.sidebar.radio("Select Recommendation Type:", ["By Ingredients", "By Similar Recipe"])
    
    if option == "By Ingredients":
        user_ingredients = st.text_input("Enter ingredients (comma-separated):")
        exclude_ingredients = st.text_input("Exclude ingredients (comma-separated, optional):")

        if st.button("🔍 Find Recipes"):
            # Process user input
            ingredients = [re.sub(r'\s+', '_', re.sub(r'\d+[^\s]*', '', ing.strip())) for ing in user_ingredients.split(',')]
            query = ' '.join(ingredients)

            # Get recommendations
            recommendations, scores = ingredient_model.recommend(query, dataset['ingredient_text'])

            if not recommendations.empty:
                st.write("### 🍲 Recommended Recipes")
                for i, idx in enumerate(recommendations.index):
                    recipe_name = dataset.loc[idx, 'name']
                    ingredients_list = dataset.loc[idx, 'ingredients_name'].split(', ')
                    st.write(f"**{i+1}. {recipe_name}**")
                    st.write(f"*Ingredients:* {', '.join(ingredients_list[:3])}...")
                    st.write(f"⭐ Match Score: {scores[i]:.2f}\n")
            else:
                st.warning("No recommendations found.")
    
    elif option == "By Similar Recipe":
        recipe_name = st.selectbox("Choose a recipe:", dataset['name'].unique())

        if st.button("🔍 Find Similar Recipes"):
            # Clean and process recipe name
            query = re.sub(r'[^\w\s]', '', recipe_name.lower())
            recommendations, scores = name_model.recommend(query, dataset['name_text'])

            if not recommendations.empty:
                st.write("### 🔥 Similar Recipes")
                for i, idx in enumerate(recommendations.index):
                    recipe_name = dataset.loc[idx, 'name']
                    ingredients_list = dataset.loc[idx, 'ingredients_name'].split(', ')
                    st.write(f"**{i+1}. {recipe_name}**")
                    st.write(f"*Ingredients:* {', '.join(ingredients_list[:3])}...")
                    st.write(f"⭐ Match Score: {scores[i]:.2f}\n")
            else:
                st.warning("No recommendations found.")

    # Ingredient frequency visualization
    def plot_ingredient_frequency(dataset):
        """Displays a bar chart of the top 10 most used ingredients."""
        all_ingredients = dataset['ingredients_name'].dropna().str.lower().str.split(', ')
        all_ingredients = [ingredient for sublist in all_ingredients for ingredient in sublist]  # Flatten list

        ingredient_counts = pd.Series(all_ingredients).value_counts().head(10)  # Get top 10 ingredients

        fig, ax = plt.subplots(figsize=(10, 5))
        ingredient_counts.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title("Top 10 Most Used Ingredients")
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Ingredients")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.sidebar.subheader("📊 Ingredient Popularity")
    if st.sidebar.button("Show Ingredient Trends"):
        plot_ingredient_frequency(dataset)
