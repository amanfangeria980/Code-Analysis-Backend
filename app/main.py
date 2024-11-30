from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from .services.code_analyzer import CodeAnalyzer
from .config import OPENAI_API_KEY

app = FastAPI(title="Code Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class AnalysisRequest(BaseModel):
    repo_url: str
    question: str

class AnalysisResponse(BaseModel):
    answer: str

analyzers = {}

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_code(request: AnalysisRequest):
    try:
        # Create or get analyzer for the repository
        if request.repo_url not in analyzers:
            analyzer = CodeAnalyzer(request.repo_url, OPENAI_API_KEY)
            await analyzer.initialize()
            analyzers[request.repo_url] = analyzer
        
        # Get analysis
        answer = await analyzers[request.repo_url].analyze_code(request.question)
        
        return AnalysisResponse(answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 