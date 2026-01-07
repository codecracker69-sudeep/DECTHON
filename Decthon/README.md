# Cineverse 🎬

AI-powered movie recommendation platform with a beautifully designed interface.

![Cineverse Screenshot](https://via.placeholder.com/800x400/0A0A0A/D4A7A7?text=Cineverse)

## Features

- **AI Movie Recommendations** - Powered by OpenAI GPT-4o
- **Streaming Responses** - Word-by-word text generation
- **Research-Visualization Design** - Elegant scholarly styling with Playfair Display typography
- **Dark Mode** - Beautiful dark theme by default
- **Responsive** - Works on all screen sizes

## Quick Start

### 1. Backend Setup

```bash
cd backend
cp .env.example .env
# Add your OpenAI API key to .env
pip install -r requirements.txt
python main.py
```

### 2. Open Frontend

Open `structure.html` in your browser, or:

```bash
open structure.html
```

## Project Structure

```
Decthon/
├── structure.html     # Main frontend
├── backend/
│   ├── main.py        # FastAPI server with OpenAI integration
│   ├── requirements.txt
│   └── .env.example   # Environment template
└── research-visualization/  # Design reference
```

## Tech Stack

- **Frontend**: HTML, Tailwind CSS, Vanilla JS
- **Backend**: Python, FastAPI, OpenAI API
- **Fonts**: Playfair Display, Inter

## Team

- **Creator**: Sagar
- **Visual Design**: Sudeep
- **Development**: Aditya

---

© 2026 Cineverse. AI-powered movie recommendations.
