"""
Cineverse Backend - OpenAI GPT-4o Movie Recommendation API
Provides streaming chat responses for movie recommendations.
"""

import os
import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Cineverse API",
    description="Movie recommendation chatbot powered by GPT-4o"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# TMDB API configuration
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# System prompt for movie recommendations
MOVIE_SYSTEM_PROMPT = """You are CineBot, an expert movie recommendation assistant for Cineverse.

Your expertise includes:
- Deep knowledge of films across all genres, eras, and countries
- Understanding of director styles, cinematography, and storytelling techniques
- Awareness of streaming availability and film history
- Ability to match movies to user moods and preferences

IMPORTANT: Distinguish between two types of queries:

1. SPECIFIC MOVIE QUERY (user asks about a particular movie by name):
   - If the user mentions a specific movie title (like "Barbie", "Inception", "The Dark Knight")
   - Provide detailed information about ONLY that specific movie
   - Include: title, year, genre, director, main cast, plot summary, and your rating
   - Do NOT recommend other similar movies unless the user specifically asks
   - Format: **Movie Title** (Year, Genre)

2. RECOMMENDATION REQUEST (user asks for suggestions):
   - If the user asks for recommendations, trending movies, or movies based on mood/genre
   - Provide 3-5 thoughtful recommendations
   - Include the movie title, year, and genre for each
   - Give a brief, engaging description (2-3 sentences)
   - Mention why the user might enjoy it based on their query
   - Add an IMDb-style rating when relevant

Format your responses in a clean, readable way. Use **bold** for movie titles.
Always include the year in parentheses after the movie title for proper card display.
Be enthusiastic but not overwhelming. If asked about non-movie topics, 
politely redirect to movie-related discussions.

Remember: You're helping users find their next favorite film! 🎬"""


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str
    stream: bool = True
    is_recommendation: bool = True


async def generate_stream(message: str, is_recommendation: bool):
    """Generate streaming response from OpenAI GPT-4o"""
    try:
        system_prompt = MOVIE_SYSTEM_PROMPT if is_recommendation else "You are a helpful assistant."
        
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            stream=True,
            max_tokens=1000,
            temperature=0.7
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                data = json.dumps({"text": text})
                yield f"data: {data}\n\n"
        
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_data = json.dumps({"error": str(e)})
        yield f"data: {error_data}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint that streams responses from GPT-4o.
    
    Returns Server-Sent Events (SSE) stream for real-time text generation.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured. Please add OPENAI_API_KEY to .env file."
        )
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    return StreamingResponse(
        generate_stream(request.message, request.is_recommendation),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Cineverse API",
        "model": "gpt-4o",
        "tmdb_configured": bool(TMDB_API_KEY)
    }


class MovieSearchRequest(BaseModel):
    """Request model for movie search endpoint"""
    title: str
    year: Optional[int] = None


class MovieData(BaseModel):
    """Response model for movie data"""
    id: int
    title: str
    year: str
    poster_url: Optional[str]
    rating: float
    overview: str
    genres: List[str]


@app.post("/search-movie")
async def search_movie(request: MovieSearchRequest):
    """
    Search TMDB for a movie and return its details.
    Returns poster URL, rating, year, overview, and genres.
    """
    if not TMDB_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="TMDB API key not configured. Please add TMDB_API_KEY to .env file."
        )
    
    try:
        async with httpx.AsyncClient() as http_client:
            # Search for the movie
            search_params = {
                "api_key": TMDB_API_KEY,
                "query": request.title,
                "include_adult": False
            }
            if request.year:
                search_params["year"] = request.year
            
            search_response = await http_client.get(
                f"{TMDB_BASE_URL}/search/movie",
                params=search_params
            )
            search_data = search_response.json()
            
            if not search_data.get("results"):
                return {"found": False, "message": f"Movie '{request.title}' not found"}
            
            # Get the first (most relevant) result
            movie = search_data["results"][0]
            movie_id = movie["id"]
            
            # Get full movie details including genres
            details_response = await http_client.get(
                f"{TMDB_BASE_URL}/movie/{movie_id}",
                params={"api_key": TMDB_API_KEY}
            )
            details = details_response.json()
            
            # Build poster URL
            poster_url = None
            if movie.get("poster_path"):
                poster_url = f"{TMDB_IMAGE_BASE_URL}{movie['poster_path']}"
            
            # Extract year from release date
            year = ""
            if movie.get("release_date"):
                year = movie["release_date"][:4]
            
            # Get genre names
            genres = [genre["name"] for genre in details.get("genres", [])]
            
            return {
                "found": True,
                "movie": {
                    "id": movie_id,
                    "title": movie.get("title", request.title),
                    "year": year,
                    "poster_url": poster_url,
                    "rating": round(movie.get("vote_average", 0), 1),
                    "overview": movie.get("overview", ""),
                    "genres": genres[:3]  # Limit to 3 genres
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching movie data: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("🎬 Starting Cineverse API...")
    print("📍 Server running at http://localhost:8000")
    print("📚 API docs at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
