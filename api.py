from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
import voyageai
from openai import OpenAI
from dotenv import load_dotenv
import os

# ============================
# 🔑 VARIÁVEIS DE AMBIENTE
# ============================

load_dotenv()  # 🔥 ESSENCIAL PARA AMBIENTE LOCAL

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_NAME = "kaizen-index"
SCORE_MINIMO = 0.6
MAX_PRODUTOS_RESPOSTA = 3

# ============================
# 🔒 VALIDAÇÃO DE CHAVES
# ============================

if not VOYAGE_API_KEY:
    raise RuntimeError("VOYAGE_API_KEY não configurada")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY não configurada")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY não configurada")

# ============================
# 🔧 CLIENTES
# ============================

voyage = voyageai.Client(api_key=VOYAGE_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ============================
# 🚀 FASTAPI
# ============================

app = FastAPI(title="API Inteligente Kaizen")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# 📩 MODELO DE REQUEST
# ============================

class QueryRequest(BaseModel):
    pergunta: str
    top_k: int = 5

# ============================
# 🔍 ENDPOINT /buscar
# ============================

@app.post("/buscar")
def buscar_produtos(req: QueryRequest):

    pergunta = req.pergunta.strip()

    if len(pergunta) < 3:
        return {
            "found": False,
            "pergunta": pergunta,
            "mensagem": "Pode me explicar um pouco melhor sua dúvida?",
            "produtos": []
        }

    # 1️⃣ Embedding
    embedding = voyage.embed(
        texts=[pergunta],
        model="voyage-lite-01"
    ).embeddings[0]

    # 2️⃣ Busca Pinecone
    resultados = index.query(
        vector=embedding,
        top_k=req.top_k,
        include_metadata=True,
        namespace="producao_v1"
    )

    produtos = []

    for match in resultados.get("matches", []):
        score = match.get("score", 0)

        if score < SCORE_MINIMO:
            continue

        meta = match.get("metadata", {})

        produtos.append({
            "nome": meta.get("nome") or meta.get("Nome"),
            "descricao": meta.get("descrição") 
            or meta.get("Descrição")
            or meta.get("descricao")
            or meta.get("description")
            or meta.get("descricao curta"),
            "score": round(score, 4)
        })

    produtos.sort(key=lambda x: x["score"], reverse=True)

    if not produtos:
        return {
            "found": False,
            "pergunta": pergunta,
            "mensagem": (
                "Não encontrei produtos do nosso catálogo que estejam "
                "realmente relacionados à sua dúvida."
            ),
            "produtos": []
        }

    produtos = produtos[:MAX_PRODUTOS_RESPOSTA]

    # ============================
    # 🧠 RAG
    # ============================

    contexto = "\n".join([
        f"- {p['nome']}: {p['descricao']}"
        for p in produtos
    ])

    prompt = f"""
Você é um assistente virtual da Kaizen Saúde Integral.
Fale de forma humanizada, acessível e profissional.

Pergunta do cliente:
"{pergunta}"

Produtos disponíveis no catálogo:
{contexto}

Regras obrigatórias:
- Responda SOMENTE com base nos produtos listados
- Não invente benefícios ou indicações
- Linguagem simples e acolhedora
- Inclua aviso de que produtos naturais não substituem orientação médica
"""

    resposta = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Você é um especialista em bem-estar natural e atendimento humanizado."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.4
    )

    mensagem_final = resposta.choices[0].message.content.strip()

    return {
        "found": True,
        "pergunta": pergunta,
        "mensagem": mensagem_final,
        "produtos": produtos
    }
