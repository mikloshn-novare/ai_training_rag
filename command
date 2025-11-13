\l
\c text_embeddings 
sudo -u postgres psql
CREATE DATABASE text_embeddings WITH TEMPLATE template0 ENCODING 'utf8' OWNER postgres;
GRANT ALL PRIVILEGES ON DATABASE text_embeddings TO postgres;
CREATE EXTENSION IF NOT EXISTS vector;
screen -S rag_db_session
source rag_env/bin/activate