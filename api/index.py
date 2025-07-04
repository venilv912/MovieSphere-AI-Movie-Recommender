from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import scipy.sparse
from rapidfuzz import process

app = Flask(__name__, template_folder="../templates")

# Load components
tfidf = joblib.load("tfidf_vectorizer.pkl")
nn = joblib.load("nearest_neighbors_model.pkl")
df = joblib.load("movies_dataframe.pkl")
tfidf_matrix = scipy.sparse.load_npz("tfidf_matrix.npz")

def fuzzy_match_title(query, title_series, limit=5, threshold=70):
    matches = process.extract(query, title_series, limit=limit, score_cutoff=threshold)
    return matches

def recommend(title, n=5):
    title = title.lower().strip()
    matches = fuzzy_match_title(title, df['title'].str.lower())

    if not matches:
        return { "error": f"No similar title found for '{title}'" }

    best_title, score, idx = matches[0]
    input_movie = df.iloc[idx][['title', 'genres', 'cast', 'directors']]
    vec = tfidf_matrix.getrow(idx)
    distances, indices = nn.kneighbors(vec, n_neighbors=n+1)
    recommended_indices = [i for i in indices[0] if i != idx]
    recommendations = df.iloc[recommended_indices][['title', 'genres', 'cast', 'directors']]

    return {
        "input_movie": input_movie.to_dict(),
        "recommendations": recommendations.to_dict(orient='records')
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