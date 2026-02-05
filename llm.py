from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# Las API keys se cargarán desde las variables de entorno o archivo .env
# LANGCHAIN_API_KEY debe estar definida como variable de entorno
os.environ["LANGCHAIN_PROJECT"] = "lanchain-prueba"  


# 1. Estructura de salida deseada
class Task(BaseModel):
    action: str = Field(description="Tarea a Realizar")
    priority: str = Field(description="Alta, Media o Baja")

class TaskList(BaseModel):
    tasks: list[Task] = Field(description="Lista de tareas extraídas del texto")

# 2. Configuración el modelo y el prompt
# Asegúrate de tener tu API key de Google configurada como variable de entorno
# GOOGLE_API_KEY debe estar definida como variable de entorno

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
prompt = ChatPromptTemplate.from_template("""
Extrae todas las tareas del siguiente texto y devuélvelas en formato JSON.
Para cada tarea, determina su prioridad basándote en el contexto:
- Alta: tareas urgentes, con tiempo límite específico o muy importantes
- Media: tareas importantes pero sin urgencia inmediata
- Baja: tareas rutinarias o de menor importancia

Texto: {text}

Devuelve el resultado en este formato JSON:
{{
    "tasks": [
        {{"action": "descripción de la tarea", "priority": "Alta/Media/Baja"}},
        ...
    ]
}}
""")

# 3. Crear la cadena chain

chain = prompt | model | JsonOutputParser(pydantic_object=TaskList)

# 4. Ejecutar con LangSmith tracking
texto_ejemplo = "Hola, recuerda comprar leche y enviar el informe antes de las 5pm. También necesito llamar al cliente para confirmar la reunión de mañana y revisar los emails."

# Ejecutar con metadatos para LangSmith
resultado = chain.invoke(
    {"text": texto_ejemplo},
    config={
        "metadata": {
            "user_id": "usuario_test",
            "session_id": "session_001",
            "task_type": "extraccion_tareas"
        },
        "tags": ["task_extraction", "gemini", "json_parser"]
    }
)

print("=== TAREAS EXTRAÍDAS ===")
print(f"Texto analizado: {texto_ejemplo[:50]}...")
print()

if isinstance(resultado, dict) and 'tasks' in resultado:
    print(f"Se encontraron {len(resultado['tasks'])} tareas:")
    for i, task in enumerate(resultado['tasks'], 1):
        print(f"{i}. Tarea: {task['action']}")
        print(f"   Prioridad: {task['priority']}")
        print()
else:
    print("Resultado completo:", resultado)

print("✅ Ejecución completada. Revisa LangSmith para ver el tracing en:")
print("https://smith.langchain.com")