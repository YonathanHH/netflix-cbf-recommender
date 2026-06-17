import streamlit as st
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

# ─── Load & Preprocess Data ─────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('netflix_titles.csv')
    movies = df[df['type'] == 'Movie'].copy().reset_index(drop=True)

    genre_col = 'listed_in' if 'listed_in' in movies.columns else 'genre'

    movies['description'] = movies['description'].fillna('')
    movies['country'] = movies['country'].fillna('Unknown')
    movies[genre_col] = movies[genre_col].fillna('Unknown')

    return movies, genre_col

@st.cache_data
def build_features(movies, genre_col):
    # TF-IDF on description
    tfidf = TfidfVectorizer(stop_words='english', max_features=500)
    tfidf_matrix = tfidf.fit_transform(movies['description'])

    # Multi-hot Genre
    mlb_genre = MultiLabelBinarizer()
    genre_matrix = mlb_genre.fit_transform(movies[genre_col].str.split(', '))

    # Multi-hot Country
    mlb_country = MultiLabelBinarizer()
    country_matrix = mlb_country.fit_transform(movies['country'].str.split(', '))

    # Combine features
    feature_matrix = sp.hstack([
        tfidf_matrix,
        sp.csr_matrix(genre_matrix),
        sp.csr_matrix(country_matrix)
    ])

    # Cosine Similarity
    cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

    # Title index mapping (lowercase)
    title_to_idx = pd.Series(movies.index, index=movies['title'].str.lower()).drop_duplicates()

    return cosine_sim, title_to_idx


def recommend(title, movies, cosine_sim, title_to_idx, genre_col, top_n=10):
    title_lower = title.lower()
    if title_lower not in title_to_idx:
        return None
    idx = title_to_idx[title_lower]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]
    movie_indices = [s[0] for s in sim_scores]
    scores = [round(s[1], 4) for s in sim_scores]
    result = movies[['title', genre_col, 'country', 'description']].iloc[movie_indices].copy()
    result['similarity_score'] = scores
    return result.reset_index(drop=True)


# ─── App UI ─────────────────────────────────────────────────
st.title('🎬 Netflix Movie Recommender')
st.markdown('**Content-Based Filtering** using TF-IDF + Cosine Similarity')
st.markdown('---')

# Load
movies, genre_col = load_data()
cosine_sim, title_to_idx = build_features(movies, genre_col)

# Search box with live suggestions
all_titles = movies['title'].sort_values().tolist()

st.subheader('🔍 Find Similar Movies')
query = st.text_input('Type a movie title you like:', placeholder='e.g. Inception, Bird Box, Mank...')

# Live suggestions as user types
if query:
    suggestions = [t for t in all_titles if query.lower() in t.lower()]
    if suggestions:
        selected_title = st.selectbox(
            f'Showing {len(suggestions)} match(es) — select a title:',
            options=suggestions
        )
    else:
        st.warning('No matching title found. Try a different keyword.')
        selected_title = None
else:
    selected_title = None

# Number of recommendations
top_n = st.slider('Number of recommendations:', min_value=3, max_value=15, value=5)

# Show recommendations
if selected_title:
    st.markdown(f'### 🎯 Movies similar to **"{selected_title}"**')
    results = recommend(selected_title, movies, cosine_sim, title_to_idx, genre_col, top_n=top_n)

    if results is not None:
        for i, row in results.iterrows():
            with st.expander(f"#{i+1} — {row['title']}  |  Score: {row['similarity_score']}"):
                st.markdown(f"**Genre:** {row[genre_col]}")
                st.markdown(f"**Country:** {row['country']}")
                st.markdown(f"**Description:** {row['description']}")
    else:
        st.error('Movie not found in the dataset.')

st.markdown('---')
st.caption('Dataset: Netflix Movies & TV Shows (Kaggle) | Method: TF-IDF + Cosine Similarity | Built with Streamlit')
