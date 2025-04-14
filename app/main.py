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


# class QueryReasoner:
#     def __init__(self):
#         self.llm = ToolCallingAgent(model="deepseek-r1-distill-qwen-1.5B")
        
#     async def analyze_query(self, user_query: str):
#         """Decide between SQL or vector search"""
#         prompt = f"""
#         Analyze this database query and choose the best approach:
#         Query: "{user_query}"
        
#         Options:
#         1. SQL - For exact matches (dates, names, exact budgets)
#         2. VECTOR - For semantic similarity (e.g., "science grants", "education funding")
        
#         Respond with either "SQL" or "VECTOR".
#         """
        
#         decision = await self.llm.generate(prompt)
#         return decision.strip().upper()


# class HybridSearch:
#     def __init__(self):
#         self.reasoner = QueryReasoner()
#         self.Session = sessionmaker(bind=engine)
        
#     async def search(self, user_query: str, top_k: int = 5):
#         # Step 1: Reason about query type
#         search_type = await self.reasoner.analyze_query(user_query)
        
#         with self.Session() as session:
#             if search_type == "SQL":
#                 # Traditional SQL search
#                 query = self._build_sql_query(user_query)
#                 results = session.execute(query).fetchall()
#             else:
#                 # Vector similarity search
#                 query_embedding = await self._get_embedding(user_query)
#                 results = self._vector_search(session, query_embedding, top_k)
                
#         return results
    
#     async def _get_embedding(self, text: str) -> list[float]:
#         """Use your DeepSeek model to generate embeddings"""
#         # Implement your embedding generation here
#         return [0.1, 0.2, ...]  # 300-dim vector
    
#     def _vector_search(self, session, query_embedding: list[float], top_k: int):
#         """Cosine similarity search using pgvector"""
#         return session.execute(
#             text("""
#             SELECT id, nombre, entidad, 
#                   1 - (keywords <=> :embedding) as similarity
#             FROM call_data
#             ORDER BY similarity DESC
#             LIMIT :top_k
#             """), 
#             {"embedding": query_embedding, "top_k": top_k}
#         ).fetchall()
    
#     def _build_sql_query(self, natural_query: str) -> str:
#         """Convert natural language to SQL (simplified)"""
#         # Use your LLM to generate SQL here
#         return text("SELECT * FROM call_data WHERE nombre LIKE '%grant%'")

if __name__ == "__main__":
    fetch_all_data_and_store_in_db(source="turismoGob")