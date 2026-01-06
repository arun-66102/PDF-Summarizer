import chromadb
import os
from sentence_transformers import SentenceTransformer
from department_corpus import build_department_documents

# Global variables for caching
_embedding_model = None
_collection = None
_initialized = False

def get_embedding_model():
    """Lazy load embedding model"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

def get_collection():
    """Lazy load and initialize collection"""
    global _collection, _initialized
    if _collection is None and not _initialized:
        try:
            chroma_client = chromadb.Client()
            _collection = chroma_client.get_or_create_collection(
                name="department_corpus"
            )
            
            # Check if collection has data
            try:
                count = _collection.count()
                if count == 0:
                    print(" Initializing department corpus...")
                    documents, metadatas, ids = build_department_documents()
                    embeddings = get_embedding_model().encode(documents).tolist()
                    
                    _collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        embeddings=embeddings,
                        ids=ids
                    )
                    print(" Department corpus initialized")
                else:
                    print(f" Using existing corpus ({count} departments)")
            except:
                # Collection might be empty, add data
                print(" Initializing department corpus...")
                documents, metadatas, ids = build_department_documents()
                embeddings = get_embedding_model().encode(documents).tolist()
                
                _collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    ids=ids
                )
                print(" Department corpus initialized")
            
            _initialized = True
        except Exception as e:
            print(f" Error initializing collection: {e}")
            raise
    
    return _collection

def classify_text_to_department(text: str, top_k: int = 1):
    """
    Classify input text to most relevant department using embeddings
    Returns list of department codes only (CSE, EEE, MECH, CIVIL)
    """
    try:
        collection = get_collection()
        embedding_model = get_embedding_model()
        
        query_embedding = embedding_model.encode([text]).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # Extract department codes only
        department_codes = []
        
        if results and results.get('metadatas') and results.get('distances'):
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            # Sort by distance (lower is better)
            sorted_results = sorted(zip(metadatas, distances), key=lambda x: x[1])
            
            for metadata, distance in sorted_results:
                department_codes.append(metadata['department_code'])
        
        return department_codes
        
    except Exception as e:
        print(f" Classification error: {e}")
        return []

def classify_text_to_department_detailed(text: str, top_k: int = 1):
    """
    Classify input text to most relevant department using embeddings
    Returns detailed results with scores (for debugging)
    """
    try:
        collection = get_collection()
        embedding_model = get_embedding_model()
        
        query_embedding = embedding_model.encode([text]).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # Convert to detailed format
        formatted_results = []
        
        if results and results.get('metadatas') and results.get('distances'):
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            # Sort by distance (lower is better)
            sorted_results = sorted(zip(metadatas, distances), key=lambda x: x[1])
            
            for metadata, distance in sorted_results:
                formatted_results.append({
                    'department_code': metadata['department_code'],
                    'department_name': metadata['department_name'],
                    'score': 1 - distance,  # Convert distance to similarity score
                    'distance': distance,
                    'method': 'embedding'
                })
        
        return formatted_results
        
    except Exception as e:
        print(f" Classification error: {e}")
        return []