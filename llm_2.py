# Asistente Inteligente de Gastos y Facturación Multimodal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Definir la estructura de salida deseada
class Gasto_Esquema(BaseModel):
    monto: float = Field(description="Valor numerico del gasto")
    moneda: str = Field(description="Tipo de moneda simbolo o abreviatura(USD, EUR, etc.)")
    comercio: str = Field(description="Nombre del lugar donde se hizo el gasto")
    categoria: str = Field(description="Categoria del comercio (comida, transporte, etc.)")
    articulos: list = Field(description="Lista de articulos comprados si es posible extraerlos")



