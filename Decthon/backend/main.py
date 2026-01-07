"""
Cineverse Backend - OpenAI GPT-4o Movie Recommendation API
Provides streaming chat responses for movie recommendations.
"""

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
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

# System prompt for movie recommendations
MOVIE_SYSTEM_PROMPT = """You are CineBot, an expert movie recommendation assistant for Cineverse.

Your expertise includes:
- Deep knowledge of films across all genres, eras, and countries
- Understanding of director styles, cinematography, and storytelling techniques
- Awareness of streaming availability and film history
- Ability to match movies to user moods and preferences

When recommending movies:
1. Provide 3-5 thoughtful recommendations unless asked for more
2. Include the movie title, year, and genre
3. Give a brief, engaging description (2-3 sentences)
4. Mention why the user might enjoy it based on their query
5. Add an IMDb-style rating when relevant

Format your responses in a clean, readable way. Use **bold** for movie titles.
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
        "model": "gpt-4o"
    }


if __name__ == "__main__":
    import uvicorn
    print("🎬 Starting Cineverse API...")
    print("📍 Server running at http://localhost:8000")
    print("📚 API docs at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
