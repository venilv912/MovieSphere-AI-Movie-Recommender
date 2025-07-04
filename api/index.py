from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import scipy.sparse
from rapidfuzz import process
import requests
import os
from dotenv import load_dotenv
load_dotenv()

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")  # Use env variable or fallback

app = Flask(__name__, template_folder="../templates")

# Load components
tfidf = joblib.load("tfidf_vectorizer.pkl")
nn = joblib.load("nearest_neighbors_model.pkl")
df = joblib.load("movies_dataframe.pkl")
tfidf_matrix = scipy.sparse.load_npz("tfidf_matrix.npz")

def fuzzy_match_title(query, title_series, limit=5, threshold=70):
    matches = process.extract(query, title_series, limit=limit, score_cutoff=threshold)
    return matches

def get_movie_poster(tmdb_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            print(f"No poster found for ID {tmdb_id}")
            return None
    except Exception as e:
        print(f"Error fetching poster for ID {tmdb_id}: {e}")
        return None

def recommend(title, n=5):
    title = title.lower().strip()
    matches = fuzzy_match_title(title, df['title'].str.lower())

    if not matches:
        return { "error": f"No similar title found for '{title}'" }

    best_title, score, idx = matches[0]
    input_movie = df.iloc[idx][['id', 'title', 'genres', 'cast', 'directors']].copy()
    input_movie['poster_url'] = get_movie_poster(input_movie['id'])

    vec = tfidf_matrix.getrow(idx)
    distances, indices = nn.kneighbors(vec, n_neighbors=n+1)
    recommended_indices = [i for i in indices[0] if i != idx]

    recommendations = []
    for i in recommended_indices:
        movie = df.iloc[i][['id', 'title', 'genres', 'cast', 'directors']].copy()
        movie['poster_url'] = get_movie_poster(movie['id'])
        recommendations.append(movie)

    return {
        "input_movie": input_movie.to_dict(),
        "recommendations": [m.to_dict() for m in recommendations]
    }


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        movie_title = request.form.get("title")
        result = recommend(movie_title, n=5)
        return render_template("index.html", input_movie=result.get("input_movie"), recommendations=result.get("recommendations", []), error=result.get("error"))
    return render_template("index.html")

@app.route("/recommend", methods=["GET"])
def recommend_api():
    title = request.args.get("title", "")
    n = int(request.args.get("n", 5))
    result = recommend(title, n)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)