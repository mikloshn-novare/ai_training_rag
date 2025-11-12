import os
from ollama import embed
from nltk.tokenize import sent_tokenize
from database_connect_embeddings import get_psql_session, insert_embeddings

def populate_vector_db(articles_dir="all_articles", model="deepseek-r1:14b", batch_size=100):
    # Populate the vector database with embeddings from text files in the specified directory.

    session = get_psql_session()
    
    for filename in os.listdir(articles_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(articles_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            sentences = sent_tokenize(content)
            sentence_embeddings = []
            sentence_contents = []
            sentence_file_names = []
            
            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i:i + batch_size]
                embeddings = embed(model=model, input=batch_sentences)['embeddings']
                
                sentence_embeddings.extend(embeddings)
                sentence_contents.extend(batch_sentences)
                sentence_file_names.extend([filename] * len(batch_sentences))
            
            insert_embeddings(sentence_embeddings, sentence_contents, sentence_file_names, session)
            print(f"Inserted embeddings for file: {filename}")