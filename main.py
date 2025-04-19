from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import text
from pgvector.psycopg import register_vector
from sqlalchemy import event

import os
from datetime import datetime
import time
import numpy as np
import base64
import logging

from db_utils.db_schema import CallData
from crawler.get_and_format_data import *
from models.agente_rag import *

from smolagents import CodeAgent
from smolagents import HfApiModel

from openai import AzureOpenAI

from sentence_transformers import SentenceTransformer  
from transformers.utils import logging as transfomers_logging 

import streamlit as st

from dotenv import load_dotenv
load_dotenv()
URI_TO_DB = f"postgresql+psycopg://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@localhost:5432/{os.getenv("POSTGRES_DB")}"


@st.cache_resource
def send_to_db(contenido: list[dict]):
    """Función general para guardar los datos en la base de datos.
    Se conecta a la base de datos PostgreSQL y guarda los datos extraídos y formateados.
    Se utiliza SQLAlchemy para la conexión y manipulación de la base de datos.
    
    Keyword arguments:
    argument -- description
    Return: return_description
    """
    
    engine = create_engine(URI_TO_DB)
    Session = sessionmaker(bind=engine)

    @event.listens_for(engine, "connect")
    def setup_vector(dbapi_connection, _):
        # 1. Create extension FIRST
        with dbapi_connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            dbapi_connection.commit()  # Commit within the same connection

        # 2. THEN register
        register_vector(dbapi_connection)

    CallData.init_table(engine)

    with Session() as session:
        # Crear extension de vector
        session.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
        session.commit()

        # Assuming `contenido` is a list of dictionaries with the data to be inserted
        for entry in contenido:
            call_data = CallData(
                nombre=entry.get("convocatoria", None),
                entidad=entry.get("entidad", None),
                fecha_publicacion=entry.get("fecha_publicacion", None),
                fecha_inicio=entry.get("fecha_inicio", None),
                fecha_final=entry.get("fecha_final", None),
                presupuesto=entry.get("presupuesto", None),
                keywords=entry.get("keywords", None),
                localidad=entry.get("localidad", None),
                url=entry.get("url", None),
            )
            session.add(call_data)
        session.commit()
        print("Content stored")
    
    return None


@st.cache_resource
def fetch_all_data_and_store_in_db(source: str):
    match source:
        case "cienciaGob":
            contenido_cienciaGob = get_and_format_cienciaGob_data()
            send_to_db(contenido_cienciaGob)
            print("cienciaGob contenido volcado")
        case "turismoGob":
            contenido_turismoGob = get_and_format_turismoGob_data()
            send_to_db(contenido_turismoGob)
            print("turismoGob contenido volcado")
        case "SNPSAP":
            contenido_SNPSAP = get_and_format_SNPSAP_data()
            send_to_db(contenido_SNPSAP)
            print("SNPSAP contenido volcado")


# def _query_the_huggingface_rag_agent():
#     print(
#         """
#         ¡Hola! Soy ACCAP, tu Asistente para Consultas de Convocatorias y Ayudas Públicas.
#         Tengo en mi una base de datos de varias consultas con esta información.
#         - nombre: nombre oficial de la convocatoria
#         - entidad: entidad que organiza la convocatoria
#         - fecha_publicacion: fecha de publicacion de la convocatoria
#         - fecha_inicio: fecha de inicio del período de aplicabilidad de la convocatoria
#         - fecha_final: fecha final del período de aplicabilidad de la convocatoria
#         - presupuesto: monto económico que ofrece la convocatoria
#         - localidad: localidad donde se concede la ayuda
#         - url: enlace para más información
#         Por favor, mantente fiel a la estructura de arriba y emplea los nombres como
#         están redactados. En caso de que no sea capaz de hayar alguna, realizaré una búsqueda semántica
#         con el mensaje de tu consulta y te mostraré las que creo que más se aproximan a lo que pides.
#         Si quieres salir, solo escribe 'exit'
#         """
#     )
#     query = input()
#     agente = CodeAgent(
#         tools=[
#             PerformSemanticSearchQuerying(),
#             PerformStandardSQLQuerying()
#         ],
#         model=HfApiModel(
#             model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
#             token=os.getenv("HUGGING_FACE_TOKEN")
#         )
#     )
#     while query.lower() != "exit":
#         agente.run(
#             query,
#         )


endpoint = os.getenv("AZURE_AI_AGENT_ENDPOINT")
api_key = os.getenv("AZURE_AI_AGENT_API_KEY")
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
transfomers_logging.disable_progress_bar()
transfomers_logging.set_verbosity_warning()
logging.getLogger("urllib3").setLevel(logging.WARNING)

client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=api_key,  
    api_version="2025-01-01-preview",
)

st.title("ACCAP")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

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
                    - url: enlace para más información
                    Por favor, mantente fiel a la estructura de arriba y emplea los nombres como
                    están redactados. En caso de que no sea capaz de hayar alguna, realizaré una búsqueda semántica
                    con el mensaje de tu consulta y te mostraré las que creo que más se aproximan a lo que pides.
                    Si quieres salir, solo escribe 'exit'
                    """
        }
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show associated DataFrame if exists
        if message.get("df") is not None:
            df: pd.DataFrame = message["df"]
            selected_cols = st.multiselect(
                "Select columns to display:",
                options=df.columns.tolist(),
                default=df.columns.tolist(),
                key=f"cols_{len(st.session_state.messages)}"
            )
            st.dataframe(df[selected_cols])

if query := st.chat_input("Haz tu consulta"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        chat_prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": 
                            """Este es un asistente que debe, dada una consulta humana 
                            y un esquema de tabla de base de datos PostgreSQL, ser capaz de decidir si
                                1- Convertir el texto a humano a lenguaje SQL para realizar la consulta. En este caso, solo debe devolver el código solicitado como texto plano.
                                2- Obtener las palabras claves para realizar búsqueda semántica. En este caso, solo debe devolver una cadena de texto con las palabras separadas por comas.
                            El esquema de base de datos es el siguiente:
                            table_name: call_data
                                    - id: INTEGER PRIMARY KEY
                                    - nombre: VARCHAR(255) NOT NULLABLE
                                    - entidad: VARCHAR(255)
                                    - fecha_publicacion: DATETIME
                                    - fecha_inicio: DATETIME
                                    - fecha_final: DATETIME
                                    - presupuesto: FLOAT
                                    - localidad: VARCHAR(255)
                                    - url: VARCHAR(255) NOT NULLABLE
                                    - keywords: VECTOR(384)
                            Solo se debe proveer una de las opciones consideradas
                            A continuación, el usuario te ofrece su consulta:

                            """ + query
                    }
                ]
            }
        ]
        completion = client.chat.completions.create(  
            model="gpt-4o-mini",
            messages=chat_prompt,
            max_tokens=800,  
            temperature=0.7,  
            top_p=0.95,  
            frequency_penalty=0,  
            presence_penalty=0,
            stop=None,
            stream=False  
        )
        mensaje = completion.choices[0].message.content
        if 'sql' in mensaje or "SELECT" in mensaje:
            query = re.sub(r"```sql\s*|```", "", mensaje).strip()
            engine = create_engine(URI_TO_DB)
            Session = sessionmaker(bind=engine)
            with Session() as session:
                result = session.execute(text(query)).fetchall()
        else:
            model = SentenceTransformer("jaimevera1107/all-MiniLM-L6-v2-similarity-es")
            embedding: np.ndarray = np.mean(model.encode(sentences=query.split(","), show_progress_bar=False), axis=0)
            engine = create_engine(URI_TO_DB)
            Session = sessionmaker(bind=engine)
            with Session() as session:
                result = session.execute(
                    text("""
                    SELECT *, 
                        1 - (keywords <=> CAST(:embedding AS vector(384))) as similarity
                    FROM call_data
                    ORDER BY similarity DESC
                    LIMIT 5
                    """), 
                    {"embedding": embedding.tolist()}
                ).fetchall()

        df = pd.DataFrame([dict(row._mapping) for row in result])
        mensaje_respuesta =  """
            ¡Por supuesto! Aquí tienes todos los resultados que he encontrado
            que se pueden ajustar a tu solicitud. Los tienes abajo en una tabla
            """
        st.write(mensaje_respuesta)
        columns_to_show = st.multiselect(
            "Select columns",
            options=df.columns.to_list(),
            default=df.columns.to_list()
        )
        st.dataframe(df[columns_to_show])
        st.write(
            """
            ¡No dudes en volver a consultarme!
            """
        )
        st.session_state.messages.append({"role": "assistant", "content": mensaje_respuesta, "df": df})
            

