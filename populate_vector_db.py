import os
import torch
from ollama import embed
from nltk.tokenize import sent_tokenize
from database_connect_embeddings import get_psql_session, TextEmbedding
from sentence_transformers import SentenceTransformer

import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# import nltk
# nltk.download("punkt")
# nltk.download("punkt_tab")

def populate_vector_database(folder_path='all_articles'):

    # Check if the directory exists, if not create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    session = get_psql_session()
    #model = SentenceTransformer("deepseek-r1:14b", device=device) #https://huggingface.co/Qwen/Qwen3-Embedding-0.6B #https://huggingface.co/Salesforce/SFR-Embedding-Mistral
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        print("Trying: {}".format(file_path))
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            sentences = sent_tokenize(content)
            # embeddings = embed(model="deepseek-coder:6.7b", input=sentences)["embeddings"]
            embeddings = embed(model="mxbai-embed-large", input=sentences)['embeddings']
            # embeddings = model.encode(sentences)
            # embeddings = model.encode(sentences, normalize_embeddings=True)

            for i, (embedding, content) in enumerate(zip(embeddings, sentences)):
                embedding_list = embedding.tolist()
                new_embedding = TextEmbedding(embedding=embedding_list, content=content, file_name=filename, sentence_number=i+1)
                session.add(new_embedding)
            session.commit()
            session.close()

            print("Succesfully generated embeddings for: {}".format(file_path))

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    return

if __name__=='__main__':
    populate_vector_database()
