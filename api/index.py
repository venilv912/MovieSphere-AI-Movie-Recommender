from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import scipy.sparse
from rapidfuzz import process
import requests
import os
from dotenv import load_dotenv
import asyncio
import aiohttp
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


async def fetch_single_poster(session, tmdb_id, api_key):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}"
    try:
        async with session.get(url, timeout=5) as response:
            if response.status == 200:
                data = await response.json()
                poster_path = data.get('poster_path')
                if poster_path:
                    return tmdb_id, f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print(f"Error fetching poster for ID {tmdb_id}: {e}")
    return tmdb_id, None


async def fetch_posters_concurrently(tmdb_ids):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_single_poster(session, tid, TMDB_API_KEY) for tid in tmdb_ids]
        results = await asyncio.gather(*tasks)
        return dict(results)


def get_popular_movies(n=20):
    """Get popular movies from your dataset"""
    # Sort by popularity or rating - adjust based on your dataframe columns
    # You can modify this logic based on what columns you have in your df
    popular_movies = df.nlargest(n, 'popularity') if 'popularity' in df.columns else df.head(n)
    
    # Get posters for popular movies
    tmdb_ids = popular_movies['id'].tolist()
    posters = asyncio.run(fetch_posters_concurrently(tmdb_ids))
    
    # Add poster URLs to movies
    popular_movies_list = []
    for _, movie in popular_movies.iterrows():
        movie_dict = movie.to_dict()
        movie_dict['poster_url'] = posters.get(movie['id'])
        popular_movies_list.append(movie_dict)
    
    return popular_movies_list


def recommend(title, n=5):
    title = title.lower().strip()
    matches = fuzzy_match_title(title, df['title'].str.lower())

    if not matches:
        return { "error": f"No similar title found for '{title}'" }

    best_title, score, idx = matches[0]
    input_movie = df.iloc[idx].copy()

    vec = tfidf_matrix.getrow(idx)
    distances, indices = nn.kneighbors(vec, n_neighbors=n+1)
    recommended_indices = [i for i in indices[0] if i != idx]

    recommended_tmdb_ids = [df.iloc[i]['id'] for i in recommended_indices]

    # Parallel poster fetching for input + recommendations
    posters = asyncio.run(fetch_posters_concurrently([input_movie['id']] + recommended_tmdb_ids))

    input_movie['poster_url'] = posters.get(input_movie['id'])

    recommendations = []
    for i in recommended_indices:
        movie = df.iloc[i].copy()
        movie['poster_url'] = posters.get(movie['id'])
        recommendations.append(movie)

    return {
        "input_movie": input_movie.to_dict(),
        "recommendations": [m.to_dict() for m in recommendations]
    }


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        movie_title = request.form.get("title")
        result = recommend(movie_title, n=8)
        return render_template("index.html", 
                               input_movie=result.get("input_movie"), 
                               recommendations=result.get("recommendations", []), 
                               error=result.get("error"),
                               show_search_results=True)
    
    # GET request - show home page with popular movies
    popular_movies = get_popular_movies(20)
    return render_template("index.html", popular_movies=popular_movies, show_search_results=False)


@app.route("/recommend", methods=["GET"])
def recommend_api():
    title = request.args.get("title", "")
    n = int(request.args.get("n", 8))
    result = recommend(title, n)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    