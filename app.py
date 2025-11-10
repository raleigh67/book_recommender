import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ---------------------------------
# Load Data
# ---------------------------------
df = pd.read_csv('/Users/raleightognela/Documents/book_recommender/goodreads_data.csv')

# Clean
df.drop_duplicates(subset=['Book', 'Author'], inplace=True)
df.dropna(subset=['Description', 'Genres'], inplace=True)
df['Num_Ratings'] = df['Num_Ratings'].str.replace(',', '').astype(int)
df['Popularity'] = df['Avg_Rating'] * np.log10(df['Num_Ratings'] + 1)


# Clean Genres column
def clean_genres(genres_str):
    
    cleaned = re.sub(r'[\[\]\'\"]', '', genres_str)
    
    genres_list = [g.strip() for g in cleaned.split(',')]
    
    seen = set()
    unique_genres = []
    for g in genres_list:
        if g and g.lower() not in seen:
            seen.add(g.lower())
            unique_genres.append(g)
    return ', '.join(unique_genres)

df['Genres'] = df['Genres'].apply(clean_genres)
genre_counts = (
    df['Genres']
    .dropna()
    .str.split(',')
    .explode()
    .str.strip()
    .value_counts()
)
top_genres = set(genre_counts.head(10).index)
df = df[df['Genres'].apply(lambda x: any(g.strip() in top_genres for g in x.split(',')))].reset_index(drop=True)

# TF-IDF for description similarity
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['Description'].fillna(""))
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['Book']).drop_duplicates()

# ---------------------------------
# Helper Functions
# ---------------------------------
def top_books_by_genre(df, genre, n=5):
    subset = df[df['Genres'].str.contains(genre, case=False, na=False)]
    return subset.sort_values('Popularity', ascending=False).head(n)[['Book', 'Author', 'Avg_Rating', 'Num_Ratings', 'URL']]

def get_similar_books(title, n=5):
    if title not in indices:
        return pd.DataFrame(columns=['Book', 'Author', 'Genres', 'Avg_Rating', 'URL'])
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:n+1]]
    return df.iloc[sim_indices][['Book', 'Author', 'Genres', 'Avg_Rating', 'URL']]

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.title("📚 Book Recommendation System")

st.sidebar.header("🔍 Options")
mode = st.sidebar.radio("Choose Recommendation Type:", ["By Genre", "Similar Books"])

# Initialize session state for selected_book
if 'selected_book' not in st.session_state:
    st.session_state.selected_book = None

if mode == "By Genre":
    # Genre-based recommendations
    all_genres = sorted(set(
        g for sublist in df['Genres'].dropna().apply(lambda x: [genre.strip() for genre in x.split(',')])
        for g in sublist
    ))
    genre = st.selectbox("Select a Genre:", all_genres)
    top_books = top_books_by_genre(df, genre)
    st.subheader(f"Top {genre} Books:")

    for _, row in top_books.iterrows():
        book_name = row['Book']
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(f"📖 {book_name}", key=book_name):
                st.session_state.selected_book = book_name
            st.markdown(f"**Author:** {row['Author']}")
            st.write(f"⭐ {row['Avg_Rating']} | {int(row['Num_Ratings']):,} ratings")
            st.markdown(f"[Goodreads Page]({row['URL']})")
            st.write("---")

        # 💡 Show similar books right below the clicked one
        if st.session_state.selected_book == book_name:
            st.markdown(f"**Books similar to _{book_name}_:**")
            results = get_similar_books(book_name)
            for _, sim_row in results.iterrows():
                st.markdown(f"- [{sim_row['Book']}]({sim_row['URL']}) by {sim_row['Author']} ⭐ {sim_row['Avg_Rating']}")
            st.write("---")

elif mode == "Similar Books":
    st.subheader("Find Similar Books by Description")
    book_list = df['Book'].tolist()
    selected_book = st.selectbox("Select a Book:", book_list)
    
    if st.button("Find Similar Books"):
        results = get_similar_books(selected_book)
        st.write(f"📚 Books similar to **{selected_book}**:")
        for _, row in results.iterrows():
            st.markdown(f"**[{row['Book']}]({row['URL']})** by {row['Author']}")
            st.write(f"⭐ {row['Avg_Rating']}")
            st.write("---")