from flask import Flask,render_template,jsonify
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676df8o#5(z^p@kjakwtw-'

client = Elasticsearch("http://localhost:9200")

# Build a search finction
def search(exclude_sections, query, index = ""):  
    """
    We use a boolean query to exclude irrelevant sections, but you choose another query type
    if you feel like it returns better results or easier to use
    """
    query_body = {"size":50,
                  "query": {
                    "bool": {
                        "should": { 
                            "match": { "text": query }
                        },
                        "must_not": {
                            "terms" : { "section_title.keyword" : exclude_sections }
                      },
                    }
                  }
                }
    # Full text search within an ElasticSearch index (''=all indexes) for the indicated text
    docs = client.search(index=index, body=query_body)
    # Reshape search results to prepare them for sentence embeddings
    texts = []
    section_titles = []
    article_titles = []
    for h in docs['hits']['hits']:
        texts.append(h["_source"]["text"])
        section_titles.append(h["_source"]["section_title"])
        article_titles.append(h["_source"]["article_title"])  
    return texts, article_titles, section_titles


def compute_embeddings(query, es_results, model, top_k=10):
    
    texts = es_results[0]
    article_titles = es_results[1]
    seaction_titles = es_results[2]
    
    embedder = SentenceTransformer(model)
    corpus_embeddings = embedder.encode(texts, convert_to_tensor=True) 
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    reranked_results = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
    
    reranked_results_list = []
    
    for item in reranked_results:
        
        idx = item['corpus_id']
        reranked_results_dict = {
            'bert_score': item['score'],
            'article_title': article_titles[idx],
            'section_title': article_titles[idx],
            'text': texts[idx]
        }
        
        reranked_results_list.append(reranked_results_dict)
    return reranked_results_list

@app.route("/")
def home():
	query = "what is disease X?"
	exclude_sections = ["See also", 'Further reading', 'Data and graphs', 'Medical journals', "External links"]
	es_results = search(exclude_sections = exclude_sections, 
	                    index = "pandemics", 
	                    query = query)
	reranked = compute_embeddings(query, es_results, model = 'distilbert-base-nli-stsb-mean-tokens')
	objJson=jsonify(reranked)
	return objJson

if "__main__" == __name__:
	app.run(debug=True)