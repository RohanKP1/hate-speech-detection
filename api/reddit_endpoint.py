from fastapi import APIRouter, HTTPException
from schemas.reddit_schema import RedditCommentRequest, RedditCommentResponse
from agents.reddit_scraping_agent import RedditScrapingAgent

router = APIRouter()

# Initialize the agent once (adjust as needed for your app structure)
reddit_agent = RedditScrapingAgent(policy_docs="data/policy_docs")


@router.post("/analyze", response_model=RedditCommentResponse)
def analyze_reddit_comments(request: RedditCommentRequest):
    try:
        results = reddit_agent.analyze_comments(
            reddit_url=str(request.reddit_url),  # Convert HttpUrl to string
            num_comments=request.num_comments,
        )
        if isinstance(results, dict) and "error" in results:
            return RedditCommentResponse(results=[], error=results["error"])
        return RedditCommentResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
