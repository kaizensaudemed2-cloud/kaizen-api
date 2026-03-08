from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
import voyageai
from openai import OpenAI
from dotenv import load_dotenv
import os
import uuid
import json
from supabase import create_client, Client

# ============================
# 🔑 VARIÁVEIS DE AMBIENTE
# ============================

load_dotenv()

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
# 🔍 ENDPOINT /chat (streaming)
# ============================

@app.post("/chat")
def chat(req: ChatRequest):

    pergunta = req.pergunta.strip()

    if len(pergunta) < 3:
        def resposta_curta():
            yield f"data: {json.dumps({'type': 'meta', 'conversation_id': '', 'produtos': []})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'token': 'Pode me explicar um pouco melhor sua dúvida?'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(resposta_curta(), media_type="text/event-stream")

    # =========================
    # Criar conversa se necessário
    # =========================

    conversation_id = req.conversation_id
    user_id = req.user_id

    if not conversation_id or conversation_id == "string":
        conversation_id = str(uuid.uuid4())
        supabase.table("conversations").insert({
            "id": conversation_id,
            "user_id": user_id
        }).execute()

    # Buscar histórico ANTES de salvar nova mensagem
    historico = buscar_historico(conversation_id)

    # Salvar pergunta do usuário
    salvar_mensagem(conversation_id, "user", pergunta)

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
    # Montar mensagens para o modelo
    # =========================

    messages = [
        {
            "role": "system",
            "content": (
                "Você é o Assistente Kaizen, especialista em bem-estar natural e saúde. "
                "Responda de forma humanizada, acolhedora e clara. "
                "Quando houver produtos relevantes, apresente-os com seus benefícios reais. "
                "Você também pode responder dúvidas gerais sobre saúde e bem-estar, mesmo sem produtos específicos. "
                "Nunca invente benefícios que não existam. "
                "Sempre inclua um aviso de que suas respostas não substituem orientação médica profissional."
            )
        }
    ]

    # Adiciona histórico real da conversa no formato OpenAI
    for msg in historico:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Monta pergunta atual com contexto de produtos (se houver)
    if produtos:
        contexto = "\n".join([
            f"- {p['nome']}: {p['descricao']}"
            for p in produtos
        ])
        conteudo_usuario = (
            f"{pergunta}\n\n"
            f"[Produtos disponíveis no catálogo Kaizen relacionados a essa pergunta:\n{contexto}]"
        )
    else:
        conteudo_usuario = pergunta

    messages.append({
        "role": "user",
        "content": conteudo_usuario
    })

    # =========================
    # Streaming da resposta
    # =========================

    def stream_resposta():
        # Primeiro envia metadados (conversation_id e produtos encontrados)
        yield f"data: {json.dumps({'type': 'meta', 'conversation_id': conversation_id, 'produtos': produtos})}\n\n"

        resposta_completa = ""

        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
            stream=True
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                resposta_completa += token
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        # Salva a resposta completa no banco após o streaming terminar
        salvar_mensagem(conversation_id, "assistant", resposta_completa)

        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_resposta(), media_type="text/event-stream")

# ============================
# 🛠️ FUNÇÕES AUXILIARES
# ============================

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

def buscar_historico(conversation_id, limite=10):
    response = (
        supabase
        .table("messages")
        .select("role, content")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=False)
        .limit(limite)
        .execute()
    )
    return response.data