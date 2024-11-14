from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from typing import List, Tuple, Optional
import logging
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class YouTubeSummarizer:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the YouTubeSummarizer with necessary components.

        Args:
            api_key: YouTube Data API key. If not provided, tries to get from environment.
        """
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            logger.error(f"Failed to download NLTK data: {e}")
            raise

        self.api_key = api_key or os.getenv('YOUTUBE_API_KEY')
        if not self.api_key:
            raise ValueError("YouTube API key is required")

        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def search_youtube(self, topic: str, max_results: int = 10) -> List[Tuple[str, str]]:
        """
        Search YouTube for videos on a specific topic.

        Args:
            topic: Search query
            max_results: Maximum number of results to return

        Returns:
            List of tuples containing video IDs and titles
        """
        try:
            request = self.youtube.search().list(
                q=topic,
                part='snippet',
                type='video',
                maxResults=max_results
            )
            response = request.execute()
            return [(item['id']['videoId'], item['snippet']['title'])
                for item in response['items']]
        except Exception as e:
            logger.error(f"YouTube search failed: {e}")
            return []

    def get_transcript(self, video_id: str) -> Optional[str]:
        """
        Get the transcript of a YouTube video.

        Args:
            video_id: YouTube video ID

        Returns:
            Complete transcript text or None if unavailable
        """
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return ' '.join([entry['text'] for entry in transcript])
        except Exception as e:
            logger.warning(f"Could not get transcript for video {video_id}: {e}")
            return None

    def summarize_text(self, text: str, max_length: int = 130,
                    chunk_size: int = 1024) -> str:
        """
        Summarize text using BART model with chunking for long texts.

        Args:
            text: Input text to summarize
            max_length: Maximum length of each summary chunk
            chunk_size: Maximum size of text chunks to process

        Returns:
            Summarized text
        """
        try:
            sentences = nltk.sent_tokenize(text)
            chunks = []
            current_chunk = ""

            for sentence in sentences:
                if len(nltk.word_tokenize(current_chunk + " " + sentence)) <= chunk_size:
                    current_chunk += " " + sentence
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk.strip())

            summaries = []
            for chunk in chunks:
                summary = self.summarizer(chunk,
                                        max_length=max_length,
                                        min_length=50,
                                        do_sample=False)[0]['summary_text']
                summaries.append(summary)

            return " ".join(summaries)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return ""

    def evaluate_summaries(self, summaries: List[str], topic: str) -> List[float]:
        """
        Evaluate summaries using cosine similarity with the topic.

        Args:
            summaries: List of summary texts
            topic: Original search topic

        Returns:
            List of similarity scores
        """
        try:
            vectorizer = TfidfVectorizer().fit_transform([topic] + summaries)
            vectors = vectorizer.toarray()
            topic_vector = vectors[0]
            return [cosine_similarity([topic_vector], [vec])[0][0]
                for vec in vectors[1:]]
        except Exception as e:
            logger.error(f"Summary evaluation failed: {e}")
            return [0.0] * len(summaries)

    def process_topic(self, topic: str, max_results: int = 10) -> List[dict]:
        """
        Process a topic by searching videos, getting transcripts, and creating summaries.

        Args:
            topic: Search topic
            max_results: Maximum number of videos to process

        Returns:
            List of dictionaries containing video information and summaries
        """
        logger.info(f"Processing topic: {topic}")
        video_results = self.search_youtube(topic, max_results)
        summaries = []

        for video_id, title in video_results:
            logger.info(f"Processing video: {title}")
            transcript = self.get_transcript(video_id)
            if transcript:
                summary = self.summarize_text(transcript)
                summaries.append({
                    'title': title,
                    'video_id': video_id,
                    'summary': summary,
                    'url': f'https://www.youtube.com/watch?v={video_id}'
                })

        if summaries:
            summary_texts = [s['summary'] for s in summaries]
            scores = self.evaluate_summaries(summary_texts, topic)

            # Add scores to summaries and sort
            for summary, score in zip(summaries, scores):
                summary['relevance_score'] = score

            summaries.sort(key=lambda x: x['relevance_score'], reverse=True)

        return summaries

def main():
    # Example usage
    try:
        summarizer = YouTubeSummarizer()
        topic = "Israel-Iran War: What Will Be Israel's Targets In Iran Now?"
        results = summarizer.process_topic(topic)

        print(f"\nSummaries for topic: {topic}\n")
        for result in results:
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Relevance Score: {result['relevance_score']:.2f}")
            print(f"Summary: {result['summary']}\n")

    except Exception as e:
        logger.error(f"Program failed: {e}")

if __name__ == "__main__":
    main()