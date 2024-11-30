from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
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
    repository_analysis: Dict[str, Any]
    analysis_summary: Dict[str, Any]
    ai_insights: Dict[str, Any]

analyzers = {}

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_code(request: AnalysisRequest):
    try:
        # Create a unique key for the analyzer
        analyzer_key = request.repo_url
        
        # Create or get analyzer for the repository
        if analyzer_key not in analyzers:
            analyzer = CodeAnalyzer(
                request.repo_url,
                OPENAI_API_KEY,
                model="gpt-4"
            )
            await analyzer.initialize()
            analyzers[analyzer_key] = analyzer
        
        # Get analysis
        answer = await analyzers[analyzer_key].analyze_code(request.question)
        
        return answer
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 