from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import text
from pgvector.psycopg import register_vector
from sqlalchemy import event

import os
from datetime import datetime
import time
import numpy as np

from db_utils.db_schema import CallData
from crawler.get_and_format_data import *
from models.agente_rag import *

from smolagents import tool, CodeAgent, ToolCallingAgent
from smolagents import HfApiModel

from dotenv import load_dotenv
load_dotenv()
URI_TO_DB = f"postgresql+psycopg://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@localhost:5432/{os.getenv("POSTGRES_DB")}"


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


def query_the_rag_agent():
    print(
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
    )
    query = input()
    agente = CodeAgent(
        tools=[
            PerformSemanticSearchQuerying(),
            PerformStandardSQLQuerying()
        ],
        model=HfApiModel(
            model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            token=os.getenv("HUGGING_FACE_TOKEN")
        )
    )
    while query.lower() != "exit":
        agente.run(
            query,
            additional_args={'uri': URI_TO_DB}
        )


if __name__ == "__main__":
    query_the_rag_agent()