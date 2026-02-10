"""
Agente LangGraph con Telegram y LangSmith
==========================================
Este ejercicio crea un agente conversacional que:
1. Responde consultas por Telegram
2. Usa Gemini 2.5-flash como LLM
3. Es observado por LangSmith para trazabilidad
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from operator import add

# LangChain y LangGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Cargar variables de entorno
load_dotenv()

# ==========================================
# CONFIGURACIÓN DE LANGSMITH
# ==========================================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "telegram-agent-langgraph"
# LANGCHAIN_API_KEY se carga automáticamente del .env

# ==========================================
# DEFINICIÓN DEL ESTADO DEL AGENTE
# ==========================================
class AgentState(TypedDict):
    """Estado del agente que mantiene el historial de mensajes"""
    messages: Annotated[Sequence[BaseMessage], add]
    user_id: str

# ==========================================
# CONFIGURACIÓN DEL MODELO LLM
# ==========================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=1024
)

# ==========================================
# NODOS DEL GRAFO
# ==========================================

def process_message(state: AgentState) -> dict:
    """
    Nodo principal que procesa el mensaje del usuario.
    Añade contexto del sistema y genera la respuesta.
    """
    messages = state["messages"]
    
    # Mensaje del sistema con instrucciones para el agente
    system_message = SystemMessage(content="""
    Eres un asistente virtual inteligente y amigable que responde consultas por Telegram.
    
    Instrucciones:
    - Responde de manera clara, concisa y útil
    - Usa emojis ocasionalmente para hacer la conversación más amigable
    - Si no sabes algo, admítelo honestamente
    - Puedes responder en español o inglés según el idioma del usuario
    - Mantén un tono profesional pero cercano
    """)
    
    # Construir la lista de mensajes para el LLM
    full_messages = [system_message] + list(messages)
    
    # Invocar el modelo
    response = llm.invoke(full_messages)
    
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    Nodo de decisión: determina si continuar o terminar.
    Por ahora siempre termina después de una respuesta.
    """
    return END

# ==========================================
# CONSTRUCCIÓN DEL GRAFO CON LANGGRAPH
# ==========================================

def create_agent_graph():
    """Crea y compila el grafo del agente."""
    
    # Crear el grafo de estados
    workflow = StateGraph(AgentState)
    
    # Añadir nodos
    workflow.add_node("process", process_message)
    
    # Definir el punto de entrada
    workflow.set_entry_point("process")
    
    # Añadir aristas
    workflow.add_edge("process", END)
    
    # Compilar con memoria para mantener el historial por usuario
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    
    return graph

# Crear la instancia del agente
agent = create_agent_graph()

# ==========================================
# HANDLERS DE TELEGRAM
# ==========================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /start - Mensaje de bienvenida"""
    welcome_message = """
👋 ¡Hola! Soy un agente inteligente powered by LangGraph y Gemini.

Puedo ayudarte con:
• Responder preguntas generales
• Mantener conversaciones contextuales
• Asistirte con diversas consultas

💡 Simplemente escríbeme tu pregunta y te responderé.

Comandos disponibles:
/start - Mostrar este mensaje
/clear - Limpiar el historial de conversación
/help - Obtener ayuda
    """
    await update.message.reply_text(welcome_message)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /clear - Limpia el historial de conversación"""
    user_id = str(update.effective_user.id)
    context.user_data.clear()
    await update.message.reply_text("🗑️ Historial de conversación limpiado. ¡Empecemos de nuevo!")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /help - Muestra ayuda"""
    help_text = """
📚 **Guía de uso**

Este bot utiliza LangGraph para mantener conversaciones inteligentes.

**Características:**
- Memoria contextual por usuario
- Respuestas generadas por Gemini 2.5-flash
- Trazabilidad completa en LangSmith

**Tips:**
- Haz preguntas claras y específicas
- Usa /clear si quieres empezar una conversación nueva
- El bot recuerda el contexto de la conversación
    """
    await update.message.reply_text(help_text, parse_mode='Markdown')


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Maneja los mensajes de texto del usuario.
    Invoca el agente LangGraph y devuelve la respuesta.
    """
    user_id = str(update.effective_user.id)
    user_message = update.message.text
    
    # Mostrar indicador de escritura
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )
    
    try:
        # Configuración del thread para mantener memoria por usuario
        config = {"configurable": {"thread_id": user_id}}
        
        # Crear el mensaje de entrada
        input_state = {
            "messages": [HumanMessage(content=user_message)],
            "user_id": user_id
        }
        
        # Invocar el agente
        result = agent.invoke(input_state, config=config)
        
        # Obtener la última respuesta del agente
        ai_response = result["messages"][-1].content
        
        # Enviar respuesta al usuario
        await update.message.reply_text(ai_response)
        
    except Exception as e:
        error_message = f"❌ Ocurrió un error: {str(e)}"
        print(f"Error procesando mensaje: {e}")
        await update.message.reply_text(error_message)


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja errores globales del bot"""
    print(f"Error: {context.error}")


# ==========================================
# FUNCIÓN PRINCIPAL
# ==========================================

def main():
    """Inicializa y ejecuta el bot de Telegram."""
    
    # Obtener el token del bot
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not telegram_token:
        print("❌ Error: TELEGRAM_BOT_TOKEN no está configurado en el archivo .env")
        print("Por favor, añade tu token de Telegram al archivo .env:")
        print("TELEGRAM_BOT_TOKEN=tu_token_aquí")
        return
    
    print("🚀 Iniciando bot de Telegram con LangGraph...")
    print("📊 LangSmith está configurado para observabilidad")
    print("🧠 Usando Gemini 2.5-flash como modelo")
    
    # Crear la aplicación
    application = Application.builder().token(telegram_token).build()
    
    # Añadir handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Handler de errores
    application.add_error_handler(error_handler)
    
    # Ejecutar el bot
    print("✅ Bot iniciado. Presiona Ctrl+C para detener.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
