# 🎬 Netflix Content-Based Filtering Recommender

A portfolio project demonstrating a **Content-Based Filtering (CBF) Recommendation System** built on the Netflix Movies Dataset using **TF-IDF Vectorization** and **Cosine Similarity**.

---

## 📌 Project Overview

This project simulates a Netflix-style recommendation engine that suggests movies based on:
- **Genre**
- **Country of Production**
- **Movie Description (TF-IDF)**

Two recommendation modes are implemented:
1. **Single User** – Input a movie title you like → Get top similar movies
2. **Multiple Users** – Simulated user-rating table → Generate personalized recommendations for each user

---

## 🧠 Methodology

| Step | Technique |
|------|----------|
| Text Feature Extraction | TF-IDF Vectorizer |
| Categorical Features | One-Hot Encoding (Genre, Country) |
| Similarity Measurement | Cosine Similarity |
| Recommendation Logic | Content-Based Filtering (Single & Multi-User) |

---

## 📁 Project Structure

```
netflix-cbf-recommender/
│
├── notebook/
│   └── netflix_cbf.ipynb        # Main analysis notebook
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YonathanHH/netflix-cbf-recommender.git
cd netflix-cbf-recommender
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download the Netflix dataset from [Kaggle](https://www.kaggle.com/datasets/imtkaggleteam/netflix) and place the CSV file in the root directory as `netflix_titles.csv`.

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

---

## 📊 Dataset
- **Source**: [Netflix Movies & TV Shows – Kaggle](https://www.kaggle.com/datasets/imtkaggleteam/netflix)
- **Features Used**: `title`, `type`, `genre`, `country`, `description`

---

## 🛠️ Tech Stack
- Python 3.10+
- Pandas, NumPy
- Scikit-learn (TF-IDF, Cosine Similarity)
- Seaborn, Matplotlib
- Streamlit

---

## 👤 Author
**Yonathan Hary Hutagalung**  
[GitHub](https://github.com/YonathanHH) | Jakarta, Indonesia
