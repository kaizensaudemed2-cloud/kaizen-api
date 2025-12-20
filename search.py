import pandas as pd
from pinecone import Pinecone
import voyageai

# ============================
# 🔑 CONFIGURAÇÕES
# ============================

VOYAGE_API_KEY = "pa-mL3exuk-YHYEJVO1Fup8Mmh8Vm6y_jmln8ifoYtwCgb"
PINECONE_API_KEY = "pcsk_4qiBEA_SqccbsbWmMZXCkMi21mqNEYMFbbjZqbqKK8KFz55CoMjREjLQ8vABuAWHsVLQaj"
INDEX_NAME = "kaizen-index"  # o mesmo nome usado no index_products.py

# ============================
# 🔧 CLIENTES
# ============================

voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ============================
# 🔍 FUNÇÃO DE BUSCA
# ============================

def buscar_produtos(query, top_k=5):
    # 1. Gerar embedding da consulta
    embedding = voyage.embed(
        texts=[query],
        model="voyage-lite-01"
    ).embeddings[0]

    # 2. Consultar Pinecone
    resultados = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )

    return resultados

# ============================
# ▶️ EXECUÇÃO (TESTE)
# ============================

if __name__ == "__main__":
    print("\n=== 🔍 BUSCA INTELIGENTE KAIZEN ===\n")

    query = input("Digite o que você procura: ")

    resultados = buscar_produtos(query)

    print("\n=== RESULTADOS ===\n")

    for match in resultados["matches"]:
        score = match["score"]
        meta = match["metadata"]

        print(f"📌 Produto: {meta.get('nome', meta.get('Nome', 'Sem nome'))}")
        print(f"📄 Descrição: {meta.get('descricao', meta.get('Descrição', ''))}")
        print(f"⭐ Similaridade: {score:.4f}")
        print("-" * 50)
