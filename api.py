from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
import voyageai
from openai import OpenAI
from dotenv import load_dotenv
import os
from supabase import create_client, Client

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

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError("Supabase não configurado")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

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

class ChatRequest(BaseModel):
    user_id: str
    pergunta: str
    conversation_id: str | None = None
    top_k: int = 5

# ============================
# 🔍 ENDPOINT /buscar
# ============================

@app.post("/chat")
def chat(req: ChatRequest):

    pergunta = req.pergunta.strip()

    if len(pergunta) < 3:
        return {
            "found": False,
            "mensagem": "Pode me explicar um pouco melhor sua dúvida?"
        }

    # =========================
    # Criar conversa se necessário
    # =========================
    conversation_id = req.conversation_id

    if not conversation_id:
        response = supabase.table("conversations").insert({
            "user_id": req.user_id
        }).execute()
        conversation_id = response.data[0]["id"]

    # =========================
    # Salvar pergunta do usuário
    # =========================
    supabase.table("messages").insert({
        "conversation_id": conversation_id,
        "role": "user",
        "content": pergunta
    }).execute()

    # =========================
    # Buscar histórico
    # =========================
    historico = buscar_historico(conversation_id)

    historico_formatado = "\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in historico
    ])

    # =========================
    # Embedding da pergunta
    # =========================
    embedding = voyage.embed(
        texts=[pergunta],
        model="voyage-lite-01"
    ).embeddings[0]

    # =========================
    # Busca Pinecone
    # =========================
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

        descricao = (
            meta.get("descricao")
            or meta.get("descricao curta")
            or ""
        )

        produtos.append({
            "nome": meta.get("nome"),
            "descricao": descricao,
            "score": round(score, 4)
        })

    produtos.sort(key=lambda x: x["score"], reverse=True)
    produtos = produtos[:MAX_PRODUTOS_RESPOSTA]

    # =========================
    # Fallback
    # =========================
    if not produtos:
        mensagem = (
            "Não encontrei produtos específicos para isso, "
            "mas posso te ajudar melhor se me contar um pouco mais "
            "sobre o que está sentindo 😊"
        )

        supabase.table("messages").insert({
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": mensagem
        }).execute()

        return {
            "conversation_id": conversation_id,
            "found": False,
            "mensagem": mensagem
        }

    # =========================
    # Montar contexto
    # =========================
    contexto = "\n".join([
        f"- {p['nome']}: {p['descricao']}"
        for p in produtos
    ])

    prompt = f"""
Histórico da conversa:
{historico_formatado}

Pergunta atual:
"{pergunta}"

Produtos disponíveis:
{contexto}

Regras:
- Responda somente com base nos produtos listados
- Não invente benefícios
- Linguagem humanizada
- Inclua aviso de que não substitui orientação médica
"""

    resposta = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um especialista em bem-estar natural e atendimento acolhedor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    mensagem_final = resposta.choices[0].message.content.strip()

    # =========================
    # Salvar resposta
    # =========================
    supabase.table("messages").insert({
        "conversation_id": conversation_id,
        "role": "assistant",
        "content": mensagem_final
    }).execute()

    return {
        "conversation_id": conversation_id,
        "found": True,
        "mensagem": mensagem_final,
        "produtos": produtos
    }

def salvar_mensagem(conversation_id, role, content):
    supabase.table("messages").insert({
        "conversation_id": conversation_id,
        "role": role,
        "content": content
    }).execute()

def criar_conversa(user_id):
    response = supabase.table("conversations").insert({
        "user_id": user_id
    }).execute()

    return response.data[0]["id"]

def buscar_historico(conversation_id, limite=6):
    response = (
        supabase
        .table("messages")
        .select("*")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=False)
        .limit(limite)
        .execute()
    )
    
    return response.data