from flask import Flask, request, jsonify
from pymongo import MongoClient
from pymongo import TEXT
from bson import ObjectId
from sentence_transformers import SentenceTransformer

from generator import converse_with_llm

from apiKey import MONGO_CONNECTION_STRING

from flask_cors import CORS
import spacy
import logging
import time
logging.basicConfig(level=logging.INFO)

client = MongoClient(MONGO_CONNECTION_STRING)
db = client["RecommendationMovie"]
movies_collection = db["movies"]
history_collection = db["search_history"]

nlp = spacy.load("en_core_web_sm")

model = SentenceTransformer("all-MiniLM-L6-v2")

app = Flask(__name__)
CORS(app)

GENRE_SYNONYMS = {
    "romance": ["romance", "romantic", "love", "rom-com"],
    "action": ["action", "adventure", "fight", "combat"],
    "comedy": ["comedy", "funny", "humor", "satire"],
    "horror": ["horror", "scary", "thriller", "fear"],
    "sci-fi": ["sci-fi", "science fiction", "space", "alien"],
    "family": ["family", "kids", "children", "child-friendly", "child"],
    "animation": ["animation", "animated", "cartoon", "pixar", "disney"],
    "drama": ["drama", "emotional", "tearjerker", "melodrama"],
    "fantasy": ["fantasy", "magic", "myth", "supernatural", "wizard", "fairy tale"],
    "mystery": ["mystery", "detective", "crime", "suspense", "whodunit"],
    "documentary": ["documentary", "docu", "real-life", "non-fiction", "biopic"],
    "musical": ["musical", "music", "singing", "dance"],
    "war": ["war", "battle", "military", "army", "soldier"],
    "history": ["history", "historical", "period drama", "biographical"],
    "sports": ["sports", "athletic", "competition", "soccer", "basketball", "football"],
    "crime": ["crime", "gangster", "mafia", "noir", "heist"]
}

def parse_advanced_filters(query):
    """
    Parse the query to extract advanced filter for movie search.
 
    Args:
        query (str): User input query
    Returns:
        dict: Filters like minimum rating or vote count
    """
    filters = {}

    if "top" in query.lower() or "high-rated" in query.lower():
        filters["vote_average"] = {"$gte": 8.5}
        
    if "popular" in query.lower():
        filters["vote_count"] = {"$gte": 500}
        
    if "recent" in query.lower():
        filters["release_date"] = {"$gte": "2020-01-01"}

    if "old" in query.lower():
        filters["release_date"] = {"$lt": "2000-01-01"}
        
    # Add time range filter for specific years
    if "between 2020-2025" in query.lower():
        filters["release_date"] = {"$gte": "2020-01-01", "$lte": "2025-12-31"}

    return filters

def clean_document(doc):
    doc["_id"] = str(doc["_id"])
    return doc

def retrieve_similar_movies(query, n=5):
    """
    Search for similar movies based on query using embeddings.
    Falls back to text search if vector search fails.
    """
    query_embedding = model.encode(query).tolist()
    filters = parse_advanced_filters(query)
    
    try:
        # Try vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "movie_index",
                    "queryVector": query_embedding,
                    "path": "movie_embedding",
                    "k": n,
                    "numCandidates": 1000,
                    "limit": 20,
                }
            }
        ]
        
        # Add filters if any
        if filters:
            pipeline.append({"$match": filters})
            
        pipeline.append({
            "$project": {
                "title": 1,
                "overview": 1,
                "poster_path": 1,
                "vote_average": 1,
                "vote_count": 1,
                "release_date": 1,
                "genre_names": 1,
                "score": {"$meta": "searchScore"}
            }
        })
        
        similar_movies_search = list(movies_collection.aggregate(pipeline))
        
        # If vector search returns no results, use fallback
        if not similar_movies_search:
            logging.info("Vector search returned no results, using fallback")
            raise Exception("No vector search results")
            
        return [clean_document(movie) for movie in similar_movies_search]
        
    except Exception as e:
        logging.error(f"Vector search failed: {e}. Using fallback method")
        
        # Fallback: Use simple text search
        # Search by genre or keywords in title/overview
        search_terms = query.split()
        text_query = {"$or": []}
        
        # Add search based on keywords in title
        for term in search_terms:
            if len(term) > 3:  # Ignore short words
                text_query["$or"].append({"title": {"$regex": term, "$options": "i"}})
                text_query["$or"].append({"overview": {"$regex": term, "$options": "i"}})
                
        # Add search based on genre if any
        genre_match = match_genre([query])
        if genre_match:
            text_query["$or"].append({"genre_names": {"$regex": genre_match, "$options": "i"}})
        
        # If no search criteria, get popular movies
        if not text_query["$or"]:
            fallback_movies = list(movies_collection.find(
                filters if filters else {},
                {
                    "title": 1,
                    "overview": 1,
                    "poster_path": 1,
                    "vote_average": 1,
                    "vote_count": 1,
                    "release_date": 1,
                    "genre_names": 1,
                }
            ).sort("vote_count", -1).limit(n))
        else:
            # Combine with additional filters if any
            if filters:
                final_query = {"$and": [text_query, filters]}
            else:
                final_query = text_query
                
            fallback_movies = list(movies_collection.find(
                final_query,
                {
                    "title": 1,
                    "overview": 1,
                    "poster_path": 1,
                    "vote_average": 1,
                    "vote_count": 1,
                    "release_date": 1,
                    "genre_names": 1,
                }
            ).sort("vote_count", -1).limit(n))
            
        return [clean_document(movie) for movie in fallback_movies]

def retrieve_similar_movies_by_genre(genre, n=5, query=""):
    """Search for movies by genre with fallback"""
    try:
        # Log genre being searched for debugging
        logging.info(f"Searching for movies with genre: {genre}")
        
        # Check data structure with sample document
        sample_doc = movies_collection.find_one({})
        if sample_doc and "genre_names" in sample_doc:
            logging.info(f"Sample genre_names format: {type(sample_doc['genre_names'])} - {sample_doc['genre_names']}")
        
        # Get advanced filters from query
        filters = parse_advanced_filters(query)
        logging.info(f"Advanced filters parsed: {filters}")
        
        # Create base query that will hold both filters and genre conditions
        base_query = {}
        
        # Add all filters from parse_advanced_filters to base query
        for key, value in filters.items():
            base_query[key] = value
        
        # Now add genre condition to the base query
        if genre:
            base_query["$or"] = [
                {"genre_names": {"$in": [genre]}},  # If genre_names is array
                {"genre_names": {"$regex": genre, "$options": "i"}},  # If genre_names is string
                {"genre_names": genre}  # Direct match
            ]
        
        logging.info(f"Using combined query: {base_query}")
        
        # Use the combined query to find matching movies
        matching_movies = list(
            movies_collection.find(
                base_query,  # Using the combined query
                {
                    "title": 1,
                    "overview": 1,
                    "poster_path": 1,
                    "vote_average": 1,
                    "vote_count": 1,
                    "release_date": 1,
                    "genre_names": 1,
                },
            )
            .sort("popularity", -1)
            .limit(n)
        )
        
        logging.info(f"Found {len(matching_movies)} movies with combined query")
        
        # If no results, try with more relaxed query but keep date filters
        if not matching_movies and filters:
            logging.info("No results with combined query, trying to keep filters but relax genre matching")
            
            # Create a new query that keeps filters but is more flexible with genre
            relaxed_query = {}
            
            # Keep all the important filters (especially date)
            for key, value in filters.items():
                relaxed_query[key] = value
            
            # Add a more flexible genre match
            if genre:
                relaxed_query["$or"] = [
                    {"genre_names": {"$regex": genre, "$options": "i"}},
                    {"overview": {"$regex": genre, "$options": "i"}},
                    {"title": {"$regex": genre, "$options": "i"}}
                ]
            
            matching_movies = list(
                movies_collection.find(
                    relaxed_query,
                    {
                        "title": 1,
                        "overview": 1,
                        "poster_path": 1,
                        "vote_average": 1,
                        "vote_count": 1,
                        "release_date": 1,
                        "genre_names": 1,
                    },
                )
                .sort("popularity", -1)
                .limit(n)
            )
            
            logging.info(f"Found {len(matching_movies)} movies with relaxed query keeping filters")
        
        # If still no results with combined approach, try genre synonyms
        if not matching_movies:
            logging.info(f"Trying with genre synonyms for {genre}")
            genre_synonyms = GENRE_SYNONYMS.get(genre.lower(), [])
            logging.info(f"Synonyms for {genre}: {genre_synonyms}")
            
            if genre_synonyms:
                # Create a new query with both filters and synonyms
                synonym_query = {}
                
                # Keep all the filters (especially date filters)
                for key, value in filters.items():
                    synonym_query[key] = value
                
                # Add synonym search
                or_conditions = []
                
                # For genre_names field
                or_conditions.append({"genre_names": {"$in": genre_synonyms}})
                
                # For title and overview
                for synonym in genre_synonyms:
                    or_conditions.append({"title": {"$regex": synonym, "$options": "i"}})
                    or_conditions.append({"overview": {"$regex": synonym, "$options": "i"}})
                
                synonym_query["$or"] = or_conditions
                logging.info(f"Synonym query with filters: {synonym_query}")
                
                matching_movies = list(
                    movies_collection.find(
                        synonym_query,
                        {
                            "title": 1,
                            "overview": 1,
                            "poster_path": 1,
                            "vote_average": 1,
                            "vote_count": 1,
                            "release_date": 1,
                            "genre_names": 1,
                        },
                    )
                    .sort("popularity", -1)
                    .limit(n)
                )
                
                logging.info(f"Found {len(matching_movies)} movies with synonym query and filters")
        
        # Log found movie titles for debugging
        if matching_movies:
            movie_titles = [movie.get('title', 'Unknown') for movie in matching_movies]
            logging.info(f"Found movies: {', '.join(movie_titles)}")
        
        return [clean_document(movie) for movie in matching_movies]
    
    except Exception as e:
        logging.error(f"Error in retrieve_similar_movies_by_genre: {e}")
        
        # Try fallback to popular movies in genre while keeping filters
        try:
            logging.info(f"Trying fallback to popular movies in genre: {genre}")
            
            # Create fallback query with both genre and filters
            fallback_query = {}
            
            # Keep all the filters (especially date filters)
            filters = parse_advanced_filters(query)
            for key, value in filters.items():
                fallback_query[key] = value
            
            # Add simple genre regex
            if genre:
                fallback_query["genre_names"] = {"$regex": genre, "$options": "i"}
            
            popular_in_genre = list(
                movies_collection.find(
                    fallback_query,
                    {
                        "title": 1,
                        "overview": 1,
                        "poster_path": 1,
                        "vote_average": 1,
                        "vote_count": 1,
                        "release_date": 1,
                        "genre_names": 1,
                    },
                )
                .sort("vote_count", -1)  # Sort by popularity as fallback
                .limit(n)
            )
            
            if popular_in_genre:
                logging.info(f"Found {len(popular_in_genre)} movies in genre fallback with filters")
                return [clean_document(movie) for movie in popular_in_genre]
        except Exception as fallback_error:
            logging.error(f"Error in genre fallback: {fallback_error}")
        
        # Final fallback - popular movies with just filters
        logging.info("Using final fallback: popular movies with just filters")
        
        # Keep the filters but drop genre requirements
        final_filters = parse_advanced_filters(query)
        
        popular_movies = list(
            movies_collection.find(
                final_filters if final_filters else {},  # Use filters if available
                {
                    "title": 1,
                    "overview": 1,
                    "poster_path": 1,
                    "vote_average": 1,
                    "vote_count": 1,
                    "release_date": 1,
                    "genre_names": 1,
                },
            )
            .sort("vote_count", -1)
            .limit(n)
        )
        
        if popular_movies:
            movie_titles = [movie.get('title', 'Unknown') for movie in popular_movies]
            logging.info(f"Fallback movies with filters: {', '.join(movie_titles)}")
        
        return [clean_document(movie) for movie in popular_movies]
    
def process_query(query):
    doc = nlp(query)
    keywords = [chunk.text.lower() for chunk in doc.noun_chunks] + [
        token.text.lower()
        for token in doc
        if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop
    ]
    return list(set(keywords))

def match_genre(keywords):
    """Better function to match genre from keywords"""
    if not keywords:
        return None
        
    # Log for debugging
    logging.info(f"Trying to match genre from keywords: {keywords}")
        
    # Check if any keyword directly matches a genre
    for keyword in keywords:
        keyword = keyword.lower().strip()
        for genre, synonyms in GENRE_SYNONYMS.items():
            if keyword == genre or any(keyword == synonym.lower() for synonym in synonyms):
                logging.info(f"Direct genre match found: {genre}")
                return genre
    
    # Check for partial matches if no direct match
    for keyword in keywords:
        keyword = keyword.lower().strip()
        for genre, synonyms in GENRE_SYNONYMS.items():
            if any(synonym.lower() in keyword or keyword in synonym.lower() for synonym in synonyms):
                logging.info(f"Partial genre match found: {genre}")
                return genre
    
    logging.info("No genre match found")
    return None

@app.route("/api/query", methods=["POST"])
def handle_query():
    data = request.json
    query = data.get("query", "")
    
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    # Generate a unique request ID based on query and timestamp
    request_id = f"{query}_{int(time.time())}"
    
    # No caching - we'll always generate a fresh response
    input_prompt = process_query(query)
    genre_match = match_genre(input_prompt)
    
    # Log the detected genre and filters for debugging
    filters = parse_advanced_filters(query)
    logging.info(f"Detected genre: {genre_match}, Filters: {filters}")
        
    # Search for similar movies
    try:
        if genre_match:
            similar_movies = retrieve_similar_movies_by_genre(genre_match, query=query)
        else:
            cleaned_query = " ".join(input_prompt) if input_prompt else query
            similar_movies = retrieve_similar_movies(cleaned_query)
        
        # If still empty, use fallback to popular movies with filters
        if not similar_movies:
            logging.warning(f"No similar movies found for: {query}. Using popular movies with filters.")
            
            # Parse any filters from the query
            filters = parse_advanced_filters(query)
            
            popular_movies = list(movies_collection.find(
                filters if filters else {},
                {
                    "title": 1,
                    "overview": 1, 
                    "poster_path": 1,
                    "vote_average": 1,
                    "vote_count": 1,
                    "release_date": 1,
                    "genre_names": 1,
                }
            ).sort("popularity", -1).limit(5))
            
            similar_movies = [clean_document(movie) for movie in popular_movies]
    except Exception as e:
        logging.error(f"Error finding similar movies: {e}")
        # Fallback to popular movies if error occurs, but still try to maintain filters
        filters = parse_advanced_filters(query)
        
        popular_movies = list(movies_collection.find(
            filters if filters else {},  # Use filters if available
            {
                "title": 1,
                "overview": 1,
                "poster_path": 1, 
                "vote_average": 1,
                "vote_count": 1,
                "release_date": 1,
                "genre_names": 1,
            }
        ).sort("popularity", -1).limit(5))
        
        similar_movies = [clean_document(movie) for movie in popular_movies]
    
    # Ensure similar_movies array is not empty before sending to LLM
    if not similar_movies:
        logging.error(f"Critical: No movies found even after fallback for query: {query}")
        # Create dummy data to avoid empty array
        similar_movies = [{
            "_id": "fallback_id",
            "title": "The Hangover",
            "overview": "When three friends finally wake up after a wild bachelor party, they can't locate their best friend, who's supposed to be tying the knot.",
            "vote_average": 7.7,
            "vote_count": 6575,
            "release_date": "2009-06-05",
            "genre_names": ["Comedy"]
        }]
    
    # Log the movies found for debugging
    movie_titles = [movie.get('title', 'Unknown') for movie in similar_movies]
    movie_years = [movie.get('release_date', 'Unknown').split('-')[0] if 'release_date' in movie else 'Unknown' for movie in similar_movies]
    logging.info(f"Final movies found: {', '.join([f'{title} ({year})' for title, year in zip(movie_titles, movie_years)])}")
    
    # Generate recommendation with LLM - improved prompt for better explanations
    prompt = f"""
    You are a movie recommendation assistant. Respond directly to this user query: "{query}"
    
    The search returned these movies that match the query:
    {", ".join([f"{movie['title']} ({movie['release_date'].split('-')[0] if 'release_date' in movie else 'Unknown'})" for movie in similar_movies])}
    
    Your task:
    1. Provide a direct, fresh response to ONLY the current query
    2. DO NOT refer to any previous interactions or previous searches
    3. DO NOT mention "based on the list you provided" or similar phrases
    4. DO NOT begin with phrases like "I see what's happening here"
    5. Recommend 3-5 appropriate movies from the list that best match the query "{query}"
    6. For EACH recommended movie, provide a detailed explanation (2-3 sentences) about WHY it's a good match
    7. Mention specific elements like plot, actors, director, or style that make the movie suitable
    8. If the query mentions kids or children, ensure recommendations are family-friendly
    9. If the query mentions "old movies", emphasize movies released before 2000
    10. If the query mentions "recent movies", emphasize movies released after 2020
    11. If the query mentions "high-rated" or "top", emphasize movies with high ratings
    
    Important: Start your response with a direct recommendation relevant to the query, then follow with detailed explanations for each movie.
    """
    
    try:
        recommendation = converse_with_llm(prompt)
    except Exception as e:
        logging.error(f"Error getting recommendation from LLM: {e}")
        # Fallback if LLM fails
        recommendation = f"Based on your search for '{query}', I recommend {similar_movies[0]['title']}. It's a great match for what you're looking for because of its engaging storyline and excellent direction. The film features {similar_movies[0].get('overview', 'an interesting plot')} which aligns perfectly with your interests."
    
    result = {"similar_movies": similar_movies, "recommendation": recommendation}
    
    # Store in history with the unique request ID to avoid conflicting with previous entries
    history_collection.insert_one({"query": request_id, "original_query": query, "result": result, "timestamp": time.time()})
    
    return jsonify(result)

@app.route("/api/history", methods=["GET"])
def get_search_history():
    # fetches all previous search queries and their results
    history = history_collection.find({}, {"_id": 0, "original_query": 1}).sort("timestamp", -1).limit(10)
    return jsonify([entry.get("original_query", "") for entry in history if "original_query" in entry])

@app.route("/api/clear_history", methods=["POST"])
def clear_history():
    """Clear all search history"""
    history_collection.delete_many({})
    return jsonify({"message": "Search history cleared successfully"})

if __name__ == "__main__":
    app.run(debug=True, port=5001)