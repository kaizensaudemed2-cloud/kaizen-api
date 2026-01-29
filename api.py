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
SCORE_MINIMO = 0.65
MAX_PRODUTOS_RESPOSTA = 3

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

    # 0️⃣ Validação mínima
    if len(pergunta) < 3:
        return {
            "found": False,
            "pergunta": pergunta,
            "mensagem": "Pode me explicar um pouco melhor sua dúvida?",
            "produtos": []
        }

    # 1️⃣ Gerar embedding da pergunta
    embedding = voyage.embed(
        texts=[pergunta],
        model="voyage-lite-01"
    ).embeddings[0]

    # 2️⃣ Buscar no Pinecone
    resultados = index.query(
        vector=embedding,
        top_k=req.top_k,
        include_metadata=True
    )

    produtos = []

    for match in resultados.get("matches", []):
        score = match.get("score", 0)

        # 🔒 Regra dura: score mínimo
        if score < SCORE_MINIMO:
            continue

        meta = match.get("metadata", {})

        produtos.append({
            "nome": meta.get("nome") or meta.get("Nome"),
            "descricao": meta.get("descricao") or meta.get("Descrição"),
            "score": round(score, 4)
        })

    # 🔽 Ordenar por relevância
    produtos.sort(key=lambda x: x["score"], reverse=True)

    # 🚫 Nenhum produto realmente relacionado
    if not produtos:
        return {
            "found": False,
            "pergunta": pergunta,
            "mensagem": (
                "Não encontrei produtos do nosso catálogo que estejam realmente "
                "relacionados à sua dúvida."
            ),
            "produtos": []
        }

    # 🔢 Limitar quantidade final
    produtos = produtos[:MAX_PRODUTOS_RESPOSTA]

    # ============================
    # 🧠 RAG (somente se houver produto relevante)
    # ============================

    contexto = "\n".join([
        f"- {p['nome']}: {p['descricao']}"
        for p in produtos
    ])

    prompt = f"""
Você é um assistente virtual da Kaizen Saúde Integral, precisa se comportar como um especialista em produtos naturais, sua linguagem não pode ser de doutor mas tem que passar uma impressão de conversa mais humanizada.

Pergunta do cliente:
"{pergunta}"

Produtos disponíveis no catálogo:
{contexto}

Regras obrigatórias:
- Responda SOMENTE com base nos produtos listados
- Se os produtos não resolverem diretamente a dúvida, explique isso com clareza
- NÃO invente benefícios, indicações ou efeitos
- Linguagem simples, acolhedora e profissional
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
