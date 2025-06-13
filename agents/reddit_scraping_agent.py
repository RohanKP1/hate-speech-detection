import praw
from core.config import Config
from controllers.text_controller import TextController
from utils.custom_logger import CustomLogger


class RedditScrapingAgent:
    def __init__(self, policy_docs):
        self.logger = CustomLogger("RedditScrapingAgent")
        self.reddit = praw.Reddit(
            client_id=Config.REDDIT_CLIENT_ID,
            client_secret=Config.REDDIT_CLIENT_SECRET,
            user_agent=Config.REDDIT_USER_AGENT,
        )
        self.text_controller = TextController(policy_docs)
        self.logger.info(
            "Initialized RedditScrapingAgent with provided credentials and TextController."
        )

    def analyze_comments(self, reddit_url: str, num_comments: int = 10):
        """
        Scrape comments from a Reddit post and analyze each for hate speech and policy using TextController.

        Args:
            reddit_url (str): The URL of the Reddit post.
            num_comments (int): Number of top-level comments to analyze.

        Returns:
            List[dict]: List of dicts with comment text and analysis results.
        """
        try:
            self.logger.info(f"Fetching submission from URL: {reddit_url}")
            submission = self.reddit.submission(url=reddit_url)
            submission.comments.replace_more(limit=0)
            comments = submission.comments.list()[:num_comments]
            self.logger.info(f"Fetched {len(comments)} comments for analysis.")

            results = []
            for idx, comment in enumerate(comments, 1):
                text = comment.body
                self.logger.info(
                    f"Analyzing comment {idx}/{len(comments)} (ID: {comment.id})"
                )
                analysis = self.text_controller.analyze_text(text)
                results.append(
                    {
                        "comment_id": comment.id,
                        "author": str(comment.author),
                        "text": text,
                        "analysis_result": analysis,
                    }
                )
            self.logger.info("Completed analysis of all comments.")
            return results
        except Exception as e:
            self.logger.error(f"Error during Reddit comment analysis: {str(e)}")
            return {"error": str(e)}
