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
NAMESPACE = "producao_v1"

# ============================
# 🔧 CLIENTES
# ============================

voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ============================
# 📥 xlsx
# ============================

df = pd.read_excel("data/produtos.xlsx")
print(f"📦 Produtos encontrados: {len(df)}")

# ============================
# ⚠️ CONFIGURAÇÕES
# ============================

BATCH_SIZE = 5          # 🔴 EXTREMAMENTE CONSERVADOR
DELAY = 70              # segundos (garante < 3 RPM)
MAX_PRODUTOS = 10  # use 100 para teste, ou None para todos

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

    if not titulo:
        continue

    texto = f"{titulo}. {short_desc}. Categoria: {categoria}"

    try:
        response = voyage.embed(
            texts=[texto],
            model="voyage-lite-01"
        )

        embedding = response.embeddings[0]

        id_produto = str(row.get("ID", f"produto-{i}")).strip()

        index.upsert(
            vectors=[{
                "id": id_produto,
                "values": embedding,
                "metadata": {
                    "nome": titulo,
                    "descricao": short_desc,
                    "categoria": categoria
                }
            }],
            namespace=NAMESPACE
        )

        produtos_processados += 1
        print(f"✅ Produto {produtos_processados} indexado")

        time.sleep(DELAY)

    except Exception as e:
        print("❌ Erro ao indexar produto:", e)
        time.sleep(120)

print("🎉 Indexação finalizada!")
