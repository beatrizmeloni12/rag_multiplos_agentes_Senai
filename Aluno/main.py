import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configurações para a Groq
API_KEY = os.getenv("GROQ_API_KEY")
MODELO = os.getenv("MODELO", "openai/gpt-oss-120b")  # Modelo padrão recomendado na Groq

# Inicializa o cliente apontando para a base_url da Groq
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# ==========================================
# BASE DE CONHECIMENTO FICTÍCIA (Simulando o RAG)
# ==========================================
BASE_CONHECIMENTO = {
    "groq": "A Groq é uma empresa de hardware e nuvem focada em IA, conhecida por sua unidade de processamento de linguagem (LPU) que oferece inferência extremamente rápida.",
    "precos": "Os custos da Groq variam de acordo com o modelo de linguagem utilizado, medidos pelo volume de tokens de entrada e saída."
}

# ==========================================
# AGENTE 1: O RECUPERADOR (Retriever Agent)
# ==========================================
def agente_recuperador(pergunta: str) -> str:
    """Busca informações relevantes na base de conhecimento com base na intenção."""
    print(f"\n[Agente 1 - Recuperador]: Buscando contexto para: '{pergunta}'...")
    
    pergunta_lower = pergunta.lower()
    contextos_encontrados = []
    
    for chave, conteudo in BASE_CONHECIMENTO.items():
        if chave in pergunta_lower:
            contextos_encontrados.append(conteudo)
            
    if not contextos_encontrados:
        return "Nenhum documento interno específico encontrado na base de conhecimento."
    
    return "\n".join(contextos_encontrados)

# ==========================================
# AGENTE 2: O GERADOR (Writer/RAG Agent)
# ==========================================
def agente_gerador(pergunta: str, contexto: str) -> str:
    """Usa o modelo na Groq para sintetizar a resposta com base no contexto."""
    print("[Agente 2 - Gerador]: Sintetizando resposta final via Groq...")
    
    prompt_sistema = (
        "Você é um assistente especialista. Responda à pergunta do usuário "
        "estritamente com base no contexto fornecido abaixo. Se a informação não estiver "
        "no contexto, diga que não sabe."
    )
    
    prompt_usuario = f"Contexto:\n{contexto}\n\nPergunta: {pergunta}"
    
    response = client.chat.completions.create(
        model=MODELO,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt_usuario}
        ]
    )
    
    return response.choices[0].message.content

# ==========================================
# ORQUESTRAÇÃO DO SISTEMA
# ==========================================
if __name__ == "__main__":
    pergunta_usuario = input('Faça a pergunta: ')
    
    if not pergunta_usuario.strip():
        print("Nenhuma pergunta foi informada.")
    else:
        print(f"\n--- INÍCIO DO FLUXO MULTI-AGENTE ---")
        print(f"Pergunta do Usuário: {pergunta_usuario}")
        
        # Passo 1: O Agente Recuperador busca os dados
        contexto_recuperado = agente_recuperador(pergunta_usuario)
        print(f"[Contexto Obtido]: {contexto_recuperado}")
        
        # Passo 2: O Agente Gerador processa o RAG e responde
        resposta_final = agente_gerador(pergunta_usuario, contexto_recuperado)
        
        print(f"\n--- RESPOSTA FINAL DO SISTEMA ---")
        print(resposta_final)