import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load dataset
@st.cache_data
def load_dataset(file_path):
    """Load and preprocess the dataset."""
    try:
        dataset = pd.read_csv(file_path)
        required_columns = {'name', 'ingredients_name'}
        missing_columns = required_columns - set(dataset.columns)

        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            return None

        dataset['ingredients_combined'] = dataset['ingredients_name'].astype(str).str.strip()
        st.success(f"✅ Dataset loaded successfully! ({len(dataset)} recipes)")
        return dataset

    except Exception as e:
        st.error(f"⚠️ Error loading dataset: {e}")
        return None

# Build the TF-IDF matrix
@st.cache_data
def build_tfidf_matrix(dataset, column):
    """Transform the ingredient data into TF-IDF vectors."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataset[column])
    return tfidf_matrix, vectorizer

# Train KNN model (🚀 Fix: Remove caching for sparse matrices)
def train_knn_model(tfidf_matrix, n_neighbors=5):
    """Train the KNN model for recommendation."""
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(tfidf_matrix)
    return knn

# Ingredient-based recommendation using KNN
def recommend_by_ingredients(user_ingredients, dataset, vectorizer, knn, top_n=5):
    """Finds the most similar recipes based on user-input ingredients using KNN."""
    
    if not user_ingredients.strip():
        st.warning("⚠️ No ingredients entered! Please try again.")
        return None

    # Convert user input into a TF-IDF vector
    user_tfidf = vectorizer.transform([user_ingredients])

    # Find the closest recipes using KNN
    distances, indices = knn.kneighbors(user_tfidf, n_neighbors=top_n)

    # Retrieve recipe names and cuisines
    recommendations = dataset.iloc[indices[0]][['name', 'cuisine']].copy()
    recommendations['similarity_score'] = (1 - distances[0]).round(2)  # Convert cosine distance to similarity score

    return recommendations

# Recipe-based recommendation using KNN
def recommend_by_recipe(recipe_name, dataset, tfidf_matrix, knn, top_n=5):
    """Finds recipes similar to a given recipe using KNN."""
    
    recipe_index = dataset.index[dataset['name'].str.lower() == recipe_name.lower()]

    if recipe_index.empty:
        st.warning("⚠️ Recipe not found in the dataset!")
        return None

    recipe_index = recipe_index[0]
    
    # Find the closest recipes using KNN
    distances, indices = knn.kneighbors(tfidf_matrix[recipe_index], n_neighbors=top_n + 1)

    # Exclude the recipe itself and retrieve results
    similar_recipes = dataset.iloc[indices[0][1:]][['name', 'cuisine']].copy()
    similar_recipes['similarity_score'] = (1 - distances[0][1:]).round(2)  # Convert distance to similarity score

    return similar_recipes

# Ingredient frequency visualization
def plot_ingredient_frequency(dataset):
    """Display a bar chart of the top 10 most used ingredients."""
    all_ingredients = dataset['ingredients_combined'].dropna().str.lower().str.split(', ')
    all_ingredients = [ingredient for sublist in all_ingredients for ingredient in sublist]  # Flatten list

    ingredient_counts = pd.Series(all_ingredients).value_counts().head(10)  # Get top 10 ingredients

    fig, ax = plt.subplots(figsize=(10, 5))
    ingredient_counts.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title("Top 10 Most Used Ingredients")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Ingredients")
    plt.xticks(rotation=45)
    st.pyplot(fig)


# Main Streamlit app
st.title("🍽 Recipe Recommendation System (KNN Model)")
st.sidebar.header("🔍 Search Options")

file_path = 'Food_Recipe.csv'  # Adjust the path as needed
dataset = load_dataset(file_path)

if dataset is not None:
    tfidf_matrix, vectorizer = build_tfidf_matrix(dataset, 'ingredients_combined')
    knn_model = train_knn_model(tfidf_matrix)  # ✅ Fix: Remove caching for sparse matrices

    option = st.sidebar.radio("Select Recommendation Type:", ["By Ingredients", "By Similar Recipe"])

    if option == "By Ingredients":
        user_ingredients = st.text_input("Enter ingredients (comma-separated):")

        if st.button("🔍 Find Recipes"):
            recommendations = recommend_by_ingredients(user_ingredients, dataset, vectorizer, knn_model)

            if recommendations is not None and not recommendations.empty:
                st.write("### 🍲 Recommended Recipes")
                st.dataframe(recommendations)
            else:
                st.warning("No recommendations found.")

    elif option == "By Similar Recipe":
        recipe_name = st.selectbox("Choose a recipe:", dataset['name'].unique())

        if st.button("🔍 Find Similar Recipes"):
            recommendations = recommend_by_recipe(recipe_name, dataset, tfidf_matrix, knn_model)

            if recommendations is not None and not recommendations.empty:
                st.write("### 🔥 Similar Recipes")
                st.dataframe(recommendations)
            else:
                st.warning("No recommendations found.")

    # Display ingredient frequency visualization
    st.sidebar.subheader("📊 Ingredient Popularity")
    if st.sidebar.button("Show Ingredient Trends"):
        plot_ingredient_frequency(dataset)
