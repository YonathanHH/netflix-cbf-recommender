# Netflix Content-Based Filtering Recommender

A portfolio project demonstrating a **Content-Based Filtering (CBF) Recommendation System** built on the Netflix Movies Dataset using **TF-IDF Vectorization** and **Cosine Similarity**.

---

## Project Overview

This project simulates a Netflix-style recommendation engine that suggests movies based on:
- **Genre**
- **Country of Production**
- **Movie Description (TF-IDF)**

Two recommendation modes are implemented:
1. **Single User** – Input a movie title + your rating → Build a **rating-weighted user feature vector** → Get top similar movies
2. **Multiple Users** – Simulated user-rating table → Build **rating-weighted feature vectors per user** → Generate personalized recommendations for each user

---

## Methodology

Both modes follow the **same rating-weighted CBF pipeline**, consistent with the Content-Based Filtering lecture framework:

```
Item-Feature Matrix (TF-IDF + Genre + Country)
        ×  User Rating
        ↓
Item-Feature Matrix with Rating
        ↓
Sum per feature column  →  User Feature Vector
        ↓
Normalize (divide by total rating sum)
        ↓
Cosine Similarity against unseen movies
        ↓
Top-N Recommendations
```

| Step | Technique |
|------|----------|
| Text Feature Extraction | TF-IDF Vectorizer on `description` |
| Categorical Features | Multi-hot Encoding (`genre`, `country`) |
| Feature Weighting | Item-Feature Matrix × User Rating |
| User Profile | Normalized weighted sum → User Feature Vector |
| Similarity | Cosine Similarity |
| Recommendation | Single User & Multi-User CBF |

---

## Project Structure

```
netflix-cbf-recommender/
│
├── netflix_cbf.ipynb             # Main analysis notebook
├── NetFlix.csv                   # raw csv data used for streamlit and notebook
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Dataset
- **Source**: [Netflix Movies & TV Shows – Kaggle](https://www.kaggle.com/datasets/imtkaggleteam/netflix)
- **Features Used**: `title`, `type`, `listed_in` (genre), `country`, `description`

---

## Tech Stack
- Python 3.10+
- Pandas, NumPy, SciPy
- Scikit-learn (TF-IDF, Cosine Similarity, MultiLabelBinarizer)
- Seaborn, Matplotlib
- Streamlit

---

## 👤 Author
**Yonathan Hary Hutagalung**  
[GitHub](https://github.com/YonathanHH) | Jakarta, Indonesia
