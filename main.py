from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import text

from smolagents import CodeAgent, AzureOpenAIServerModel
from models.agente_rag import *

import os, re
from datetime import datetime
import time
import numpy as np
import pandas as pd
import base64
import logging

from tenacity import retry, stop_after_attempt, retry_if_exception

from psycopg.errors import UndefinedColumn, ProgrammingError, OperationalError

from openai import AzureOpenAI

from sentence_transformers import SentenceTransformer  
from transformers.utils import logging as transfomers_logging 

import streamlit as st

from crawler.agentico.get_and_format_data import *
from crawler.clasico.get_and_format_data import *
from get_and_store_data_from_source import send_to_db
from db_utils.db_schema import CallData

from dotenv import load_dotenv
load_dotenv()
URI_TO_DB = f"postgresql+psycopg://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@db:5432/{os.getenv("POSTGRES_DB")}"
MAX_ATTEMPTS = 5
engine = create_engine(URI_TO_DB)
Session = sessionmaker(bind=engine)
# Si no existe la tabla, se crea y el vector de embeddings
CallData.init_table(engine)

# hash_funcs = {
#     list: lambda x: hash(tuple(getattr(obj, "name", obj) for obj in x))
# }

class AgenteRAG(CodeAgent):
    """Hereda de CodeAgent, intento de hacer un
    agente con reintentos en caso de fallo o error,
    no funciona por presunta incompatibilidad de smolagents
    con tenacity y streamlit.
    """
    
    def __init__(self):
        super().__init__(
        tools = [
                PerformHybridQuery(),
                PerformSQLQuery(),
                PerformSemanticQuery()
            ],
        model = AzureOpenAIServerModel(
                model_id="gpt-4o-mini",
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version="2025-01-01-preview",
            )
        )

    # @st.cache(
    #     show_spinner="Analizado la consulta, puede tardar unos segundos...",
    #     hash_funcs=hash_funcs
    # )
    @retry(
        stop=stop_after_attempt(MAX_ATTEMPTS),
        retry=retry_if_exception((UndefinedColumn, ProgrammingError, OperationalError))
    )
    def runAgent(self, query: str):
        try:
            response = self.run(query)
            
            # Validcacion
            if not response or "ERROR:" in response:
                raise ValueError("Invalid agent response")
                
            return response
            
        except Exception as e:  # tomamos con todos los errores
            if isinstance(e, (UndefinedColumn, ProgrammingError)):
                raise e("Error con la consulta SQL") # tenacity se encargas de este punto
            return f"Critical error: {str(e)}"
        
# endpoint del llm
endpoint = os.getenv("AZURE_AI_AGENT_ENDPOINT")
# api key
api_key = os.getenv("AZURE_AI_AGENT_API_KEY")
# Evitamos logging de info o debug
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
transfomers_logging.disable_progress_bar()
transfomers_logging.set_verbosity_warning()
logging.getLogger("urllib3").setLevel(logging.WARNING)

# mensajes para comunicarse con el usuario
salutes = [
    "Aquí tienes la información que he encontrado:",
    "Estas son las convocatorias que he encontrado:",
    "Aquí tienes los resultados de tu consulta:",
    "Aquí tienes los datos que he encontrado:",
    "¿Qué te parece esta información?",
    "Aquí tienes los resultados de tu búsqueda:",
    "Esto es lo que he encontrado en mi base de datos:",
]

# mensajes para despedirse
greets = [
    "No dudes en volver a preguntar",
    "¡Espero que esta información te sea útil!",
    "¡No dudes en preguntar si necesitas más información!",
    "¡Espero que encuentres lo que buscas!",
    "¡Estoy aquí para ayudarte!",
    "¡No dudes en preguntar si necesitas más ayuda!",
    "¡Espero haberte ayudado!",
    "¡Espero haberte sido de ayuda!",
    "¡Espero que esta información te sea útil!",
]   

# en caso de error
errors = [
    "Discúlpame, he tenido un error al consultar la base de datos.",
    "Lo siento, no he sido capaz de darte resultados."
    "He tenido un incoveniente al obtener la información que me pides, espero que no te moleste."
]

st.title("ACCAP")
# Create three buttons in the same row
col1, col2, col3 = st.columns(3)

with col1:
    button1 = st.button(
        "SNPSAP (agentico)",
        help="Obtiene datos de SNPSAP por medios agenticos",
        on_click=lambda : send_to_db(get_and_format_AgenticoSNPSAP_data(), uri_type="db")
    )

with col2:
    button2 = st.button(
        "AEI (agentico)",
        help="Obtiene datos de AEI por medios agenticos",
        on_click=lambda : send_to_db(get_and_format_AgenticoSNPSAP_data(), uri_type="db")
    )

# mensaje inicial
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": 
                    """
                    ¡Hola! Soy ACCAP, tu Asistente para Consultas de Convocatorias y Ayudas Públicas.
                    Tengo en mi una base de datos de varias consultas con esta información.
                    - nombre: nombre oficial de la convocatoria
                    - entidad: entidad que organiza la convocatoria
                    - fecha_publicacion: fecha de publicacion de la convocatoria
                    - fecha_inicio: fecha de inicio del período de aplicabilidad de la convocatoria
                    - fecha_final: fecha final del período de aplicabilidad de la convocatoria
                    - presupuesto: monto económico que ofrece la convocatoria
                    - localidad: localidad donde se concede la ayuda
                    - url: enlace para más informació
                    - beneficiario: tipo de beneficiario que puede solicitar la ayuda
                    - tipo: tipo de ayuda que se concede
                    - bases: enlace a las bases de la convocatoria
                    - compatibilidad: compatibilidad de la convocatoria con otras ayudas
                    - objetivo: objetivo de la convocatoria
                    - duracion: duración de la convocatoria

                    Puedes hacerme preguntas sobre la base de datos, como por ejemplo:
                    - ¿Cuántas convocatorias hay?
                    - ¿Cuántas convocatorias hay de una entidad específica?
                    - ¿Cuántas convocatorias hay en una localidad específica?
                    - ¿Cuántas convocatorias hay de un tipo específico?
                    - ¿Cuántas convocatorias hay de una fecha específica?

                    Por favor, mantente fiel a la estructura de arriba. En caso de que no sea capaz de 
                    hayar alguna, realizaré una búsqueda semántica
                    con el mensaje de tu consulta y te mostraré las que creo que más se aproximan a lo que pides.
                    """
        }
    )

# se crea el agente CodeAgent
if "agente" not in st.session_state:
    st.session_state.agente = CodeAgent(
        tools = [
            PerformHybridQuery(), # busqueda semantica + sql
            PerformSQLQuery(), # sql
            PerformSemanticQuery() # busqueda semantica
        ],
        model = AzureOpenAIServerModel(
            model_id="gpt-4o-mini",
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2025-01-01-preview",
        )
    )
    #st.session_state.agente_rag = AgenteRAG() 

# Se annaden los mensajes al historial y se escriben en
# el chat para mantenerlo visible
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Muestra el dataset de los datos
        # de anteriores consultas si existe
        st.markdown(message.get("salute", ""))
        if isinstance(message.get("content"), pd.DataFrame):
            df: pd.DataFrame = message["content"]
            # Para seleccionar columnas
            selected_cols = st.multiselect(
                "Select columns to display:",
                options=df.columns.tolist(),
                default=df.columns.tolist(),
                key=f"cols_{len(st.session_state.messages)}"
            )
            st.dataframe(df[selected_cols])
        elif isinstance(message.get("content"), (str, int)):
            st.markdown(message["content"])
        st.markdown(message.get("greet", ""))
        
# Si no existe la consulta, se crea como un chat_input
if query := st.chat_input("Haz tu consulta"):
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Se escribe la consulta en el chat
    with st.chat_message("user"):
        st.markdown(query)
    
    # Respuesta del modelo
    with st.chat_message("assistant"):
        with st.spinner("Analizando la consulta, puede tardar unos segundos..."):  # Spinner para la espera
            # Simular tiempo de procesamiento
            time.sleep(2)
            # Escogemos un greet y un mensaje aleatorio 
            greet = np.random.choice(greets)
            salute = np.random.choice(salutes)
            try:
                response = st.session_state.agente.run(query)
            except (ProgrammingError, OperationalError) as e: # La IA realiza una consulta errónea o que no se ajusta al esquema de la bbdd
                error_msg = np.random.choice(errors)
                st.session_state.messages.append({
                    "role": "assistant",
                    "salute": "",
                    "content": error_msg,
                    "greet": greet
                })
            except Exception as e: # Error desconocido, por lo general, al realizar una conexión a la bbdd
                error_msg = "Ha ocurrido un error que no puedo identificar"
                st.session_state.messages.append({
                    "role": "assistant",
                    "salute": "",
                    "content": error_msg,
                    "greet": greet
                })
            else:
                # La respuesta puede ser devuelta en varios formatos
                # no solo en la herramienta
                
                if isinstance(response, pd.DataFrame): # Si es un dataframe de pandas
                    df = response
                        
                    st.markdown(salute)
                    # filtro de columnas
                    cols = st.multiselect(
                        "Columnas a mostrar",
                        options=df.columns,
                        default=df.columns.tolist()
                    )
                    st.dataframe(df[cols])
                        
                    st.session_state.messages.append({
                        "role": "assistant",
                        "salute": salute,
                        "content": df,
                        "greet": greet
                    })

                elif isinstance(response, list): # Si es una lista, suele ser una lista de dict (records)
                    df = pd.DataFrame.from_records(response)
                        
                    st.markdown(salute)
                    # filtro de columnas
                    cols = st.multiselect(
                        "Columnas a mostrar",
                        options=df.columns,
                        default=df.columns.tolist()
                    )
                    st.dataframe(df[cols])
                    st.markdown(greet)
                        
                    st.session_state.messages.append({
                        "role": "assistant",
                        "salute": salute,
                        "content": df,
                        "greet": greet
                    })

                elif isinstance(response, dict): # Si es un dict, suele ser solo un record
                    df = pd.DataFrame().from_records([response])

                    st.markdown(salute)
                    # filtro de columnas
                    cols = st.multiselect(
                        "Columnas a mostrar",
                        options=df.columns,
                        default=df.columns.tolist()
                    )
                    st.dataframe(df[cols])
                    st.markdown(greet)
                        
                    st.session_state.messages.append({
                        "role": "assistant",
                        "salute": salute,
                        "content": df,
                        "greet": greet
                    })

                elif isinstance(response, (str, int)): # En caso cualquiera, suele ser una cadena de texto no controlable
                # Un número, entradas en texto plano, un aviso, etc.
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "salute": "",
                        "content": response,
                        "greet": greet
                    })
                    st.markdown(greet)
                else:
                    st.markdown("No he podido entender la respuesta del modelo, por favor, inténtalo de nuevo.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "salute": "",
                        "content": "No he podido entender la respuesta del modelo, por favor, inténtalo de nuevo.",
                        "greet": ""
                    })