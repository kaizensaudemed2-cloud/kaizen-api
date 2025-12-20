# index_products.py

import voyageai 
from pinecone import Pinecone, ServerlessSpec
import os
import pandas as pd

# 🔑 CONFIGURAÇÕES (substitua com suas chaves reais)
VOYAGE_API_KEY = "pa-mL3exuk-YHYEJVO1Fup8Mmh8Vm6y_jmln8ifoYtwCgb"
PINECONE_API_KEY = "pcsk_4qiBEA_SqccbsbWmMZXCkMi21mqNEYMFbbjZqbqKK8KFz55CoMjREjLQ8vABuAWHsVLQaj"
PINECONE_ENV = "us-east-1"  # ou o que aparecer como "Region" no seu painel
INDEX_NAME = "kaizen-index"  # pode ser o nome do índice que você criou no Pinecone

# 🔧 Inicializar clientes
voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# 🧱 Criar índice se não existir
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,  # compatível com o modelo de embedding da OpenAI
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
    )

index = pc.Index(INDEX_NAME)

# 🧾 Exemplo de produtos (você vai substituir isso pelas descrições reais)
produtos = [
    {
        "id": "1",
        "nome": "Óleo de Coco Extra Virgem",
        "descricao": "Óleo de coco 100% natural, prensado a frio, ideal para culinária e cuidados com o cabelo e pele."
    },
    {
        "id": "2",
        "nome": "Chá de Hibisco",
        "descricao": "Chá natural de hibisco rico em antioxidantes, auxilia na digestão e contribui para o bem-estar geral."
    },
    {
        "id": "3",
        "nome": "Mel Orgânico Puro",
        "descricao": "Mel puro e orgânico, direto do produtor, sem adição de açúcar ou conservantes."
    },
]

# 🔄 Gerar embeddings e enviar para o Pinecone
for produto in produtos:
    texto = f"{produto['nome']} - {produto['descricao']}"
    response = voyage.embed(
    texts=[texto],
    model="voyage-lite-01"  # modelo gratuito
)
    embedding = response.embeddings[0]

    index.upsert(
        vectors=[{
            "id": produto["id"],
            "values": embedding,
            "metadata": {"nome": produto["nome"], "descricao": produto["descricao"]}
        }]
    )

print("✅ Produtos indexados com sucesso no Pinecone!")
