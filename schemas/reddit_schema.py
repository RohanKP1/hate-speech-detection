from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Any


class RedditCommentRequest(BaseModel):
    reddit_url: HttpUrl = Field(..., description="URL of the Reddit post to analyze")
    num_comments: int = Field(
        10, ge=1, le=100, description="Number of comments to analyze"
    )


class RedditCommentAnalysis(BaseModel):
    comment_id: str
    author: Optional[str]
    text: str
    analysis_result: Any


class RedditCommentResponse(BaseModel):
    results: List[RedditCommentAnalysis]
    error: Optional[str] = None
