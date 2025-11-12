import os
import torch
import nltk
from sentence_transformers import SentenceTransformer
from database_connect_embeddings import get_psql_session, TextEmbedding

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize


def populate_vector_database(folder_path='all_articles'):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Use a valid, efficient embedding model
    model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5",
        device=device
    )

    session = get_psql_session()

    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {file_path}")

            if not os.path.isfile(file_path):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                if not content.strip():
                    print(f"Empty file: {filename}")
                    continue

                sentences = sent_tokenize(content)
                if not sentences:
                    continue

                print(f"  → {len(sentences)} sentences")

                # Batch encode all sentences
                embeddings = model.encode(
                    sentences,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )

                for i, (embedding, sentence) in enumerate(zip(embeddings, sentences)):
                    # Check for duplicates
                    exists = session.query(TextEmbedding).filter_by(
                        file_name=filename,
                        sentence_number=i + 1
                    ).first()

                    if exists:
                        continue

                    embedding_list = embedding.tolist()

                    new_embedding = TextEmbedding(
                        embedding=embedding_list,
                        content=sentence,
                        file_name=filename,
                        sentence_number=i + 1
                    )
                    session.add(new_embedding)

                session.commit()
                print(f"Successfully embedded: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                session.rollback()
                continue

    finally:
        session.close()


if __name__ == '__main__':
    populate_vector_database()