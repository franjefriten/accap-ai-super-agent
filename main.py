from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import text

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
                session.execute(
                    text("""
                    CREATE OR REPLACE FUNCTION compute_similarity(keywords vector[], embeddings vector[]) 
                    RETURNS FLOAT AS $$
                    DECLARE
                        total_distance FLOAT := 0;
                        pair_count INTEGER := 0;
                        u vector;
                        v vector;
                    BEGIN
                        FOREACH u IN ARRAY keywords LOOP
                            FOREACH v IN ARRAY embeddings LOOP
                                total_distance := total_distance + (u <=> v);
                                pair_count := pair_count + 1;
                            END LOOP;
                        END LOOP;

                        IF pair_count = 0 THEN
                            RETURN 0.0;
                        END IF;

                        -- Return average similarity (1 - average distance)
                        RETURN 1 - (total_distance / pair_count);
                    END;
                    $$ LANGUAGE plpgsql;
                    """
                ))

                result = session.execute(
                    text("""      
                    WITH similarity_table AS (
                        SELECT 
                            id,
                            compute_similarity(
                                keywords, 
                                ARRAY[CAST(:embedding AS vector(384))]  -- Wrap single vector in array
                            ) AS similarity
                        FROM 
                            call_data
                    )
                    SELECT 
                        cd.*,
                        st.similarity
                    FROM 
                        call_data cd
                    JOIN 
                        similarity_table st ON cd.id = st.id
                    ORDER BY
                        st.similarity DESC  -- Higher similarity first
                    LIMIT 10;
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
            

