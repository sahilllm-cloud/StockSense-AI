
from dotenv import load_dotenv
import os
import cohere

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY is not configured in .env")

co = cohere.Client(api_key=COHERE_API_KEY)


def rerank_search_results(company_name: str, search_results: list[str], top_n: int):
    query = (
        f"Filter and prioritize news headlines explicitly discussing factors "
        f"influencing {company_name}'s stock price, including market trends, "
        f"financial reports, corporate announcements, and industry-specific "
        f"news impacting {company_name}'s performance."
    )

    results = co.rerank(
        model="rerank-english-v2.0",
        query=query,
        documents=search_results,
        top_n=top_n
    )

    return [result.document['text'] for result in results]
