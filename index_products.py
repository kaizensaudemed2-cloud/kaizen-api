import os
import time
import pandas as pd
from pinecone import Pinecone
import voyageai
from dotenv import load_dotenv

# ============================
# 🔐 VARIÁVEIS
# ============================

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "kaizen-index"

# ============================
# 🔧 CLIENTES
# ============================

voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ============================
# 📥 CSV
# ============================

df = pd.read_csv("data/produtos.csv")
print(f"📦 Produtos encontrados: {len(df)}")

# ============================
# ⚠️ CONFIGURAÇÕES DE SEGURANÇA
# ============================

BATCH_SIZE = 5          # 🔴 EXTREMAMENTE CONSERVADOR
DELAY = 70              # segundos (garante < 3 RPM)
MAX_PRODUTOS = 20   # use 100 para teste, ou None para todos

# ============================
# 🚀 INDEXAÇÃO
# ============================

produtos_processados = 0

for i, row in df.iterrows():
    if MAX_PRODUTOS and produtos_processados >= MAX_PRODUTOS:
        break

    titulo = str(row.get("Title", "")).strip()
    short_desc = str(row.get("Short Description", "")).strip()
    categoria = str(row.get("Categorias de produto", "")).strip()

    texto = f"{titulo}. {short_desc}. Categoria: {categoria}"

    try:
        response = voyage.embed(
            texts=[texto],
            model="voyage-lite-01"
        )

        embedding = response.embeddings[0]

        index.upsert(
            vectors=[{
                "id": f"produto-{i}",
                "values": embedding,
                "metadata": {
                    "nome": titulo,
                    "descricao": short_desc,
                    "categoria": categoria
                }
            }]
        )

        produtos_processados += 1
        print(f"✅ Produto {produtos_processados} indexado")

        print("⏳ Aguardando rate limit...")
        time.sleep(DELAY)

    except Exception as e:
        print("❌ Erro ao indexar produto:", e)
        print("⏸️ Aguardando 120s antes de tentar novamente...")
        time.sleep(120)

print("🎉 Indexação finalizada!")
