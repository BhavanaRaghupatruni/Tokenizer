from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tiktoken
from typing import Dict

app = FastAPI(
    title="Bhavana's Tokenizer API", 
    description="ChatGPT Tokenizer Backend API by Bhavana",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class TokenizeRequest(BaseModel):
    text: str
    model: str = "gpt-3.5-turbo"

class VocabularyRequest(BaseModel):
    start: int = 0
    limit: int = 100
    search: str = ""

# Helper functions
def tokenize_text_helper(text: str, model: str) -> Dict:
    """Tokenize text and return details"""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    token_strings = [encoding.decode([token]) for token in tokens]
    
    token_details = []
    for idx, (token_id, token_str) in enumerate(zip(tokens, token_strings)):
        token_details.append({
            "position": idx + 1,
            "token_id": token_id,
            "token_string": token_str,
            "byte_length": len(token_str.encode('utf-8'))
        })
    
    return {
        "tokens": tokens,
        "token_strings": token_strings,
        "count": len(tokens),
        "token_details": token_details,
        "model": model,
        "original_text": text
    }

def get_vocabulary_helper(start: int, limit: int, search: str) -> Dict:
    """Get vocabulary with pagination and search"""
    encoding = tiktoken.get_encoding("cl100k_base")
    total_vocab_size = encoding.n_vocab
    
    vocab_list = []
    end = min(start + limit, total_vocab_size)
    
    for i in range(start, end):
        try:
            token_str = encoding.decode([i])
            if search == "" or search.lower() in token_str.lower():
                vocab_list.append({
                    "token_id": i,
                    "token_string": token_str,
                    "length": len(token_str)
                })
        except Exception:
            if search == "":
                vocab_list.append({
                    "token_id": i,
                    "token_string": "[SPECIAL_TOKEN]",
                    "length": 0
                })
    
    return {
        "vocabulary": vocab_list,
        "total_size": total_vocab_size,
        "showing_start": start,
        "showing_end": end,
        "count": len(vocab_list),
        "search_term": search
    }

def get_stats_helper() -> Dict:
    """Get encoding statistics"""
    encoding = tiktoken.get_encoding("cl100k_base")
    return {
        "encoding_name": encoding.name,
        "total_tokens": encoding.n_vocab,
        "description": "GPT-3.5/GPT-4 vocabulary (cl100k_base)",
        "supported_models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    }

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Bhavana's Tokenizer API is running! 😎",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": ["/tokenize", "/vocabulary", "/stats"]
    }

@app.post("/tokenize")
async def tokenize_text(request: TokenizeRequest):
    """Tokenize input text"""
    try:
        result = tokenize_text_helper(request.text, request.model)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

@app.post("/vocabulary")
async def get_vocabulary(request: VocabularyRequest):
    """Get vocabulary with pagination and search"""
    try:
        result = get_vocabulary_helper(request.start, request.limit, request.search)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get vocabulary statistics"""
    try:
        stats = get_stats_helper()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Bhavana's Tokenizer API",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)