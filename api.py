from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
import voyageai
from openai import OpenAI
import os

# ============================
# 🔑 VARIÁVEIS DE AMBIENTE
# ============================

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_NAME = "kaizen-index"

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
# 🔍 ENDPOINT COM RAG
# ============================

@app.post("/buscar")
def buscar_produtos(req: QueryRequest):

    # 🔎 0️⃣ Validação rápida da pergunta
    if len(req.pergunta.strip()) < 3:
        return {
            "pergunta": req.pergunta,
            "mensagem": "Pode me explicar melhor sua dúvida?",
            "produtos": []
        }

    # 1️⃣ Embedding da pergunta
    embedding = voyage.embed(
        texts=[req.pergunta],
        model="voyage-lite-01"
    ).embeddings[0]

    # 2️⃣ Busca no Pinecone
    resultados = index.query(
        vector=embedding,
        top_k=req.top_k,
        include_metadata=True
    )

    SCORE_MINIMO = 0.78
    produtos = []

    for match in resultados["matches"]:
        if match["score"] < SCORE_MINIMO:
            continue

        meta = match["metadata"]

        produtos.append({
            "nome": meta.get("nome") or meta.get("Nome"),
            "descricao": meta.get("descricao") or meta.get("Descrição"),
            "score": round(match["score"], 4)
        })

    # 🚫 Nenhum produto relevante
    if not produtos:
        return {
            "pergunta": req.pergunta,
            "mensagem": (
                "Não encontrei produtos do nosso catálogo relacionados a essa dúvida. "
                "Se quiser, tente usar outras palavras ou perguntar sobre bem-estar, chás, suplementos ou produtos naturais."
            ),
            "produtos": []
        }

    # 🔽 Limitar a 3 produtos
    produtos = produtos[:3]

    # 3️⃣ Contexto RAG
    contexto = "\n".join([
        f"- {p['nome']}: {p['descricao']}"
        for p in produtos
    ])

    prompt = f"""
Você é um assistente virtual da Kaizen Saúde Integral.

Pergunta do cliente:
"{req.pergunta}"

Produtos disponíveis no catálogo:
{contexto}

Regras obrigatórias:
- Responda SOMENTE com base nos produtos listados
- Se os produtos não resolverem diretamente a dúvida, explique isso
- NÃO invente benefícios
- Linguagem simples, acolhedora e profissional
- Inclua aviso de que produtos naturais não substituem orientação médica
"""

    # 4️⃣ OpenAI
    resposta = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um especialista em bem-estar natural e atendimento humanizado."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    mensagem_final = resposta.choices[0].message.content.strip()

    return {
        "pergunta": req.pergunta,
        "mensagem": mensagem_final,
        "produtos": produtos
    }
