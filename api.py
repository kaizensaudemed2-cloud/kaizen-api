from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone
import voyageai
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ============================
# 🔑 CONFIGURAÇÕES
# ============================

VOYAGE_API_KEY = "pa-mL3exuk-YHYEJVO1Fup8Mmh8Vm6y_jmln8ifoYtwCgb"
PINECONE_API_KEY = "pcsk_4qiBEA_SqccbsbWmMZXCkMi21mqNEYMFbbjZqbqKK8KFz55CoMjREjLQ8vABuAWHsVLQaj"
INDEX_NAME = "kaizen-index"

# ============================
# 🔧 CLIENTES
# ============================

voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ============================
# 🚀 FASTAPI
# ============================

app = FastAPI(title="API Inteligente Kaizen")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # libera qualquer site (depois podemos restringir)
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, OPTIONS etc
    allow_headers=["*"],  # Content-Type, Authorization etc
)


class QueryRequest(BaseModel):
    pergunta: str
    top_k: int = 5

@app.post("/buscar")
def buscar_produtos(req: QueryRequest):
    # 1. Gerar embedding da pergunta
    embedding = voyage.embed(
        texts=[req.pergunta],
        model="voyage-lite-01"
    ).embeddings[0]

    # 2. Buscar no Pinecone
    resultados = index.query(
        vector=embedding,
        top_k=req.top_k,
        include_metadata=True
    )

    produtos = []
    nomes = []

    for match in resultados["matches"]:
        meta = match["metadata"]
        produtos.append({
            "nome": meta.get("nome") or meta.get("Nome"),
            "descricao": meta.get("descricao") or meta.get("Descrição"),
            "score": round(match["score"], 4)
        })

        if meta.get("nome") or meta.get("Nome"):
            nomes.append(meta.get("nome") or meta.get("Nome"))

    # 3. Criar mensagem humanizada
    if produtos:
        mensagem = (
            f"Com base no que você procura ({req.pergunta}), "
            f"estes produtos naturais da Kaizen podem ser relevantes:\n\n"
            + "\n".join([f"• {nome}" for nome in nomes[:5]])
            + "\n\n⚠️ Importante: os produtos naturais auxiliam o bem-estar, "
              "mas não substituem orientação médica ou tratamento profissional."
        )
    else:
        mensagem = (
            "No momento não encontramos produtos diretamente relacionados "
            "à sua busca. Caso queira, tente usar outras palavras."
        )

    return {
        "pergunta": req.pergunta,
        "mensagem": mensagem,
        "produtos": produtos
    }
