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

from openai import AzureOpenAI

from sentence_transformers import SentenceTransformer  
from transformers.utils import logging as transfomers_logging 

import streamlit as st

from dotenv import load_dotenv
load_dotenv()
URI_TO_DB = f"postgresql+psycopg://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@db:5432/{os.getenv("POSTGRES_DB")}"


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

st.title("ACCAP")

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
        if message.get("greet") is not None:
            st.markdown(message["greet"])
        

if query := st.chat_input("Haz tu consulta"):
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        agente = CodeAgent(
            tools=[
                PerformQuery()
            ],
            model=AzureOpenAIServerModel(
                model_id="gpt-4o-mini",
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version="2025-01-01-preview",
            )
        )
        
        try:

            response = agente.run(query)
            
            if isinstance(response, pd.DataFrame):
                df = response
                mensaje = "Aquí están los resultados:"
                
                st.markdown(mensaje)
                cols = st.multiselect(
                    "Columnas a mostrar",
                    options=df.columns,
                    default=df.columns.tolist()
                )
                st.dataframe(df[cols])
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": mensaje,
                    "df": df
                })

            if isinstance(response, list):
                df = pd.DataFrame.from_records(response)
                mensaje = "Aquí están los resultados:"
                
                st.markdown(mensaje)
                cols = st.multiselect(
                    "Columnas a mostrar",
                    options=df.columns,
                    default=df.columns.tolist()
                )
                st.dataframe(df[cols])
                st.markdown("¡No dudes en volver a preguntar!")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": mensaje,
                    "df": df,
                    "greet": "¡No dudes en volver a preguntar!"
                })

            else:
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "greet": "¡No dudes en volver a preguntar!"
                })
                st.markdown("¡No dudes en volver a preguntar!")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            

