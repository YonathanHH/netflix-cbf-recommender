import streamlit as st
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

# ─── Load & Preprocess ────────────────────────────────────────
@st.cache_data
def load_and_build():
    df = pd.read_csv('netflix_titles.csv')
    movies = df[df['type'] == 'Movie'].copy().reset_index(drop=True)

    genre_col = 'listed_in' if 'listed_in' in movies.columns else 'genre'
    movies['description'] = movies['description'].fillna('')
    movies['country']     = movies['country'].fillna('Unknown')
    movies[genre_col]     = movies[genre_col].fillna('Unknown')

    # TF-IDF
    tfidf        = TfidfVectorizer(stop_words='english', max_features=500)
    tfidf_matrix = tfidf.fit_transform(movies['description'])

    # Multi-hot Genre
    mlb_genre    = MultiLabelBinarizer()
    genre_matrix = mlb_genre.fit_transform(movies[genre_col].str.split(', '))

    # Multi-hot Country
    mlb_country    = MultiLabelBinarizer()
    country_matrix = mlb_country.fit_transform(movies['country'].str.split(', '))

    # Combine → dense (needed for rating multiplication)
    feature_matrix = sp.hstack([
        tfidf_matrix,
        sp.csr_matrix(genre_matrix),
        sp.csr_matrix(country_matrix)
    ]).toarray()

    # Title index
    title_to_idx = pd.Series(
        movies.index,
        index=movies['title'].str.lower()
    ).drop_duplicates()

    return movies, genre_col, feature_matrix, title_to_idx


# ─── Core CBF Logic ───────────────────────────────────────────
def build_user_vector(watched_ratings, feature_matrix, title_to_idx):
    """
    Rating-weighted user feature vector (lecture pipeline):
      1. feature_row × rating  for each watched movie
      2. Sum all weighted rows
      3. Normalize ÷ total rating sum
    """
    weighted_sum = np.zeros(feature_matrix.shape[1])
    total_rating = 0

    for title, rating in watched_ratings.items():
        if rating > 0 and title.lower() in title_to_idx:
            idx = title_to_idx[title.lower()]
            weighted_sum += feature_matrix[idx] * rating
            total_rating += rating

    if total_rating == 0:
        return weighted_sum
    return weighted_sum / total_rating


def recommend(watched_ratings, movies, genre_col, feature_matrix, title_to_idx, top_n=10):
    user_vec   = build_user_vector(watched_ratings, feature_matrix, title_to_idx)
    sim_scores = cosine_similarity([user_vec], feature_matrix)[0]

    watched_indices = set(
        title_to_idx[t.lower()] for t in watched_ratings
        if t.lower() in title_to_idx and watched_ratings[t] > 0
    )

    filtered = [(i, s) for i, s in enumerate(sim_scores) if i not in watched_indices]
    filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_n]

    indices = [s[0] for s in filtered]
    scores  = [round(s[1], 4) for s in filtered]

    result = movies[['title', genre_col, 'country', 'description']].iloc[indices].copy()
    result['similarity_score'] = scores
    return result.reset_index(drop=True)


# ─── Load Data ────────────────────────────────────────────────
movies, genre_col, feature_matrix, title_to_idx = load_and_build()
all_titles = movies['title'].sort_values().tolist()

# ─── UI ───────────────────────────────────────────────────────
st.title('🎬 Netflix Movie Recommender')
st.markdown('**Content-Based Filtering** — Rating-Weighted User Feature Vector + Cosine Similarity')
st.markdown('---')

# ── Session state for watch list ──────────────────────────────
if 'watch_list' not in st.session_state:
    st.session_state.watch_list = {}  # {title: rating}

# ── Add a movie ───────────────────────────────────────────────
st.subheader('📋 Your Watch History')
st.markdown('Search and add movies you have watched, then rate them.')

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input('Search movie title:', placeholder='e.g. Inception, Bird Box...')

if query:
    suggestions = [t for t in all_titles if query.lower() in t.lower()]
    if suggestions:
        selected = st.selectbox(
            f'{len(suggestions)} match(es) found — pick one:',
            options=suggestions
        )
    else:
        st.warning('No matching title found. Try a different keyword.')
        selected = None
else:
    selected = None

if selected:
    with col2:
        rating = st.slider('Your rating:', min_value=1, max_value=10, value=8)
    if st.button(f'➕ Add "{selected}"'):
        st.session_state.watch_list[selected] = rating
        st.success(f'Added "{selected}" with rating {rating}/10')

# ── Show current watch list ───────────────────────────────────
if st.session_state.watch_list:
    st.markdown('**🎞️ Movies in your list:**')
    watch_df = pd.DataFrame(
        list(st.session_state.watch_list.items()),
        columns=['Title', 'Rating']
    )
    st.dataframe(watch_df, use_container_width=True)

    if st.button('🗑️ Clear Watch List'):
        st.session_state.watch_list = {}
        st.rerun()

    st.markdown('---')
    top_n = st.slider('Number of recommendations:', min_value=3, max_value=15, value=5)

    if st.button('🎯 Get Recommendations'):
        results = recommend(
            st.session_state.watch_list,
            movies, genre_col, feature_matrix, title_to_idx,
            top_n=top_n
        )
        st.subheader('🍿 Recommended for You')
        for i, row in results.iterrows():
            with st.expander(f"#{i+1} — {row['title']}  |  Score: {row['similarity_score']}"):
                st.markdown(f"**Genre:** {row[genre_col]}")
                st.markdown(f"**Country:** {row['country']}")
                st.markdown(f"**Description:** {row['description']}")
else:
    st.info('👆 Search and add at least one movie to get started!')

st.markdown('---')
st.caption('Dataset: Netflix Movies & TV Shows (Kaggle) | Method: Rating-Weighted CBF + Cosine Similarity | Built with Streamlit')
