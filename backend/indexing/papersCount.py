import feedparser
import requests

BASE_URL = "http://export.arxiv.org/api/query"

categories = "(cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.MA)"
keywords = '(ti:"retrieval augmented" OR ti:"RAG" OR ti:"retrieval-augmented")'

query = f"{categories} AND {keywords}"


def get_total_count(q):
    params = {"search_query": q, "start": 0, "max_results": 1}
    resp = requests.get(BASE_URL, params=params)
    feed = feedparser.parse(resp.text)
    return int(feed.feed.opensearch_totalresults)


print("Query:", query)
print("Total unique papers:", get_total_count(query))
