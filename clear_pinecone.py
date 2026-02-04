import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("kaizen-index")

index.delete(delete_all=True, namespace="producao_v1")

print("🔥 Índice do Pinecone limpo com sucesso")
