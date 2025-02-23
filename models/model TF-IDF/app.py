import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data
def load_dataset(file_path):
    """Load the recipe dataset and validate required columns."""
    try:
        dataset = pd.read_csv(file_path)

        required_columns = {'name', 'ingredients_name'}
        missing_columns = required_columns - set(dataset.columns)

        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            return None

        st.success(f"✅ Dataset loaded successfully! ({len(dataset)} recipes)")
        return dataset

    except Exception as e:
        st.error(f"⚠️ Error loading dataset: {e}")
        return None

# Preprocess ingredients
@st.cache_data
def preprocess_ingredients(dataset):
    """Combine ingredients into a single string per recipe."""
    dataset['ingredients_combined'] = dataset['ingredients_name'].astype(str).str.strip()
    return dataset

# Build the TF-IDF matrix
@st.cache_data
def build_tfidf_matrix(dataset, column):
    """Build a TF-IDF matrix based on the given column."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataset[column])
    return tfidf_matrix, vectorizer

# Ingredient-based recommendation
def recommend_by_ingredients(user_ingredients, tfidf_matrix, dataset, vectorizer, top_n=5, exclude_ingredients=None):
    """Recommend recipes based on user-entered ingredients, with optional filtering before selecting top recipes."""
    
    if not user_ingredients.strip():
        st.warning("⚠️ No ingredients entered! Please try again.")
        return None

    # Filter dataset BEFORE computing similarity
    filtered_dataset = dataset.copy()

    if exclude_ingredients:
        exclude_ingredients = [ing.strip().lower() for ing in exclude_ingredients if ing.strip()]
        if exclude_ingredients:
            filtered_dataset = filtered_dataset[~filtered_dataset['ingredients_combined']
                                                .str.lower()
                                                .str.contains('|'.join(exclude_ingredients), case=False, na=False)]

    # Check if we have recipes left after filtering
    if filtered_dataset.empty:
        st.warning("⚠️ No recipes found after applying exclusion filters.")
        return None

    # Compute TF-IDF matrix for the filtered dataset
    filtered_tfidf_matrix = vectorizer.transform(filtered_dataset['ingredients_combined'])
    
    # Compute similarity between user ingredients and filtered dataset
    user_tfidf = vectorizer.transform([user_ingredients])
    cosine_similarities = cosine_similarity(user_tfidf, filtered_tfidf_matrix).flatten()
    
    # Select top N recommended recipes
    recommendations = cosine_similarities.argsort()[-top_n:][::-1]
    
    return filtered_dataset.iloc[recommendations][['name', 'cuisine']] if not filtered_dataset.iloc[recommendations].empty else None
    
# Recipe-based recommendation
def recommend_by_recipe(recipe_name, tfidf_matrix, dataset, top_n=5):
    """Recommend similar recipes based on a given recipe."""
    recipe_index = dataset.index[dataset['name'].str.lower() == recipe_name.lower()]
    
    if recipe_index.empty:
        st.warning("⚠️ Recipe not found in the dataset!")
        return None

    recipe_index = recipe_index[0]
    cosine_similarities = cosine_similarity(tfidf_matrix[recipe_index], tfidf_matrix).flatten()
    similar_recipes = cosine_similarities.argsort()[-(top_n + 1):][::-1][1:]  # Exclude the recipe itself

    return dataset.iloc[similar_recipes][['name', 'cuisine']]

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
st.title("🍽 Recipe Recommendation System")
st.sidebar.header("🔍 Search Options")

file_path = 'Food_Recipe.csv'  # Adjust the path as needed
dataset = load_dataset(file_path)

if dataset is not None:
    dataset = preprocess_ingredients(dataset)
    tfidf_matrix, vectorizer = build_tfidf_matrix(dataset, 'ingredients_combined')

    option = st.sidebar.radio("Select Recommendation Type:", ["By Ingredients", "By Similar Recipe"])

    if option == "By Ingredients":
        user_ingredients = st.text_input("Enter ingredients (comma-separated):")
        exclude_ingredients = st.text_input("Exclude ingredients (comma-separated, optional):").split(',')

        if st.button("🔍 Find Recipes"):
            recommendations = recommend_by_ingredients(user_ingredients, tfidf_matrix, dataset, vectorizer, exclude_ingredients=exclude_ingredients)

            if recommendations is not None and not recommendations.empty:
                st.write("### 🍲 Recommended Recipes")
                st.dataframe(recommendations)
            else:
                st.warning("No recommendations found.")

    elif option == "By Similar Recipe":
        recipe_name = st.selectbox("Choose a recipe:", dataset['name'].unique())

        if st.button("🔍 Find Similar Recipes"):
            recommendations = recommend_by_recipe(recipe_name, tfidf_matrix, dataset)

            if recommendations is not None and not recommendations.empty:
                st.write("### 🔥 Similar Recipes")
                st.dataframe(recommendations)
            else:
                st.warning("No recommendations found.")

    # Display ingredient frequency visualization
    st.sidebar.subheader("📊 Ingredient Popularity")
    if st.sidebar.button("Show Ingredient Trends"):
        plot_ingredient_frequency(dataset)
