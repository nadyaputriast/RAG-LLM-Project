import requests
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from apiKey import TMDB_API_KEY, MONGO_CONNECTION_STRING
import logging

logging.basicConfig(level=logging.INFO)

# 1. Connect to MongoDB (vector DB)
client = MongoClient(MONGO_CONNECTION_STRING)
db = client["RecommendationMovie"]  # Nama database MongoDB
movies_collection = db["movies"]  # Nama collection
model = SentenceTransformer("all-MiniLM-L6-v2")  # Model untuk embedding

# 2. Get data movies from TMDB API
def fetch_tmdb_genres():
    url = "https://api.themoviedb.org/3/genre/movie/list"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}

    response = requests.get(url, params=params)

    if response.status_code == 200:
        genres = response.json().get("genres", [])
        return {genre["id"]: genre["name"] for genre in genres}
    else:
        logging.error("Failed to fetch genres from TMDB API")
        return {}

def fetch_tmdb_movies(page=1):
    url = "https://api.themoviedb.org/3/movie/popular"
    params = {"api_key": TMDB_API_KEY, "language": "en-US", "page": page}

    response = requests.get(url, params=params)

    if response.status_code == 200:
        movies = response.json().get("results", [])
        filtered_movies = [movie for movie in movies if not movie.get("adult", False)]
        return filtered_movies
    else:
        logging.error(f"Failed to fetch popular movies from TMDB API (Page {page})")
        return []

# 3. Embedding data movies
def seed_movies(movies, genres):
    for movie in movies:
        genre_ids = movie.get("genre_ids", [])
        genre_names = [genres.get(genre_id, "Unknown") for genre_id in genre_ids]

        movie_text = f"{movie['title']} {movie.get('overview', '')}"
        movie_embedding = model.encode(movie_text)

        movie_doc = {
            "tmdb_id": movie["id"],
            "overview": movie.get("overview", ""),
            "popularity": movie.get("popularity", 0),
            "poster_path": movie.get("poster_path", ""),
            "release_date": movie.get("release_date", ""),
            "title": movie.get("title", ""),
            "vote_average": movie.get("vote_average", 0),
            "vote_count": movie.get("vote_count", 0),
            "genre_ids": genre_ids,
            "genre_names": genre_names,
            "movie_embedding": movie_embedding.tolist()  # Convert NumPy array to list
        }

        # Insert/update in MongoDB
        movies_collection.update_one(
            {"tmdb_id": movie["id"]},
            {"$set": movie_doc},
            upsert=True
        )

# Store embedding data to MongoDB (vector DB)
def seed_database_from_tmdb(pages=1):
    genres = fetch_tmdb_genres()
    
    for page in range(1, pages+1):
        movies = fetch_tmdb_movies(page)
        if not movies:
            break
        seed_movies(movies, genres)

seed_database_from_tmdb(pages=500)
print("Database seeding completed")

# Setelah menghubungkan ke MongoDB dan sebelum fungsi lainnya
def create_vector_index():
    try:
        movies_collection.create_index([
            ("movie_embedding", "vector")
        ], 
        name="movie_index", 
        dimension=384)  # Dimensi embedding dari model all-MiniLM-L6-v2
        logging.info("Vector index created successfully")
    except Exception as e:
        logging.error(f"Error creating vector index: {e}")

create_vector_index()