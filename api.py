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
# 📩 MODELOS DE REQUEST
# ============================

class ChatRequest(BaseModel):
    user_id: str
    pergunta: str
    conversation_id: str | None = None
    top_k: int = 5

class RenomearRequest(BaseModel):
    titulo: str

# ============================
# 🔍 SYSTEM PROMPT
# ============================

SYSTEM_PROMPT = """Você é Kai, o assistente virtual da Kaizen — uma marca de produtos naturais voltada para saúde e bem-estar.

Sua personalidade:
- Acolhedora, empática e genuinamente curiosa sobre o que o usuário está sentindo
- Conversa de forma natural, como um amigo que entende muito de saúde natural
- Nunca parece um catálogo de produtos ou uma lista de vendas
- Faz perguntas abertas no final das respostas para aprofundar a conversa e entender melhor a necessidade do usuário

Como responder:
- Responda de forma fluida e humanizada, como uma conversa real
- Quando houver produtos relevantes no catálogo, mencione-os naturalmente dentro do texto — não como uma lista separada no final
- Se não encontrar produto específico, responda sobre o tema de saúde/bem-estar de forma útil e pergunte mais sobre a situação da pessoa
- Sempre termine com uma pergunta ou convite para continuar a conversa
- Use parágrafos curtos, linguagem leve e evite listas excessivas
- Nunca invente benefícios de produtos que não existam
- Sempre inclua de forma natural (não robótica) um lembrete de que suas orientações não substituem acompanhamento médico profissional

Lembre-se: você está tendo uma conversa, não fazendo uma apresentação de produtos."""

# ============================
# 💬 ENDPOINT /chat (streaming)
# ============================

@app.post("/chat")
def chat(req: ChatRequest):

    pergunta = req.pergunta.strip()

    if len(pergunta) < 3:
        def resposta_curta():
            yield f"data: {json.dumps({'type': 'meta', 'conversation_id': '', 'produtos': []})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'token': 'Pode me contar um pouco mais? Quero entender melhor como posso te ajudar 😊'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(resposta_curta(), media_type="text/event-stream")

    conversation_id = req.conversation_id
    user_id = req.user_id
    nova_conversa = False

    if not conversation_id or conversation_id == "string":
        conversation_id = str(uuid.uuid4())
        nova_conversa = True
        supabase.table("conversations").insert({
            "id": conversation_id,
            "user_id": user_id,
            "titulo": pergunta[:60]  # Título automático = primeira pergunta
        }).execute()

    historico = buscar_historico(conversation_id)
    salvar_mensagem(conversation_id, "user", pergunta)

    # Embedding + Busca Pinecone
    embedding = voyage.embed(
        texts=[pergunta],
        model="voyage-lite-01"
    ).embeddings[0]

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
        descricao = meta.get("descricao") or meta.get("descricao curta") or ""
        produtos.append({
            "nome": meta.get("nome"),
            "descricao": descricao,
            "score": round(score, 4)
        })

    produtos.sort(key=lambda x: x["score"], reverse=True)
    produtos = produtos[:MAX_PRODUTOS_RESPOSTA]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in historico:
        messages.append({"role": msg["role"], "content": msg["content"]})

    if produtos:
        contexto = "\n".join([f"- {p['nome']}: {p['descricao']}" for p in produtos])
        conteudo_usuario = (
            f"{pergunta}\n\n"
            f"[Contexto interno — produtos do catálogo Kaizen relacionados, mencione-os naturalmente se fizer sentido:\n{contexto}]"
        )
    else:
        conteudo_usuario = pergunta

    messages.append({"role": "user", "content": conteudo_usuario})

    def stream_resposta():
        yield f"data: {json.dumps({'type': 'meta', 'conversation_id': conversation_id, 'nova_conversa': nova_conversa, 'titulo': pergunta[:60], 'produtos': produtos})}\n\n"

        resposta_completa = ""
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            stream=True
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                resposta_completa += token
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        salvar_mensagem(conversation_id, "assistant", resposta_completa)
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_resposta(), media_type="text/event-stream")

# ============================
# 📋 ENDPOINT GET /conversas/{user_id}
# ============================

@app.get("/conversas/{user_id}")
def listar_conversas(user_id: str):
    response = (
        supabase
        .table("conversations")
        .select("id, titulo, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(30)
        .execute()
    )
    return {"conversas": response.data}

# ============================
# 📨 ENDPOINT GET /mensagens/{conversation_id}
# ============================

@app.get("/mensagens/{conversation_id}")
def carregar_mensagens(conversation_id: str):
    response = (
        supabase
        .table("messages")
        .select("role, content, created_at")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=False)
        .execute()
    )
    return {"mensagens": response.data}

# ============================
# ✏️ ENDPOINT PATCH /conversas/{conversation_id}/titulo
# ============================

@app.patch("/conversas/{conversation_id}/titulo")
def renomear_conversa(conversation_id: str, req: RenomearRequest):
    supabase.table("conversations").update({
        "titulo": req.titulo
    }).eq("id", conversation_id).execute()
    return {"ok": True}

# ============================
# 🗑️ ENDPOINT DELETE /conversas/{conversation_id}
# ============================

@app.delete("/conversas/{conversation_id}")
def deletar_conversa(conversation_id: str):
    supabase.table("messages").delete().eq("conversation_id", conversation_id).execute()
    supabase.table("conversations").delete().eq("id", conversation_id).execute()
    return {"ok": True}

# ============================
# 🛠️ FUNÇÕES AUXILIARES
# ============================

def salvar_mensagem(conversation_id, role, content):
    supabase.table("messages").insert({
        "conversation_id": conversation_id,
        "role": role,
        "content": content
    }).execute()

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