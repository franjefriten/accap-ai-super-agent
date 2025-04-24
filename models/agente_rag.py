from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os
from typing import Optional
from smolagents import Tool
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
import streamlit as st

# class QueryReasoner:
#     def __init__(self):
#         self.llm = ToolCallingAgent(model="deepseek-r1-distill-qwen-1.5B")
        
#     async def analyze_query(self, user_query: str):
#         """Decide entre SQL o busqueda semantica"""
#         prompt = f"""
#         Analiza esta consulta de base de datos y escoge el mejor acercamiento:
#         Consulta: "{user_query}"
        
#         Opciones:
#         1. SQL - Para consultas estándar (comparar fechas, comparar presupuesto, coincidencia exacta de entidad)
#         2. VECTOR - Para similitud semántica ("agricultura", "turismo", "comercio", "cultura", "arte", "hostelería", "ciencia")
        
#         Responde con "SQL" o "VECTOR".
#         """
        
#         decision = await self.llm.generate(prompt)
#         return decision.strip().upper()


# class HybridSearch(QueryReasoner):
#     def __init__(self, session: Session, engine: engine):
#         super().__init__()
#         self.engine = engine
#         self.Session = session
        
#     async def search(self, user_query: str, top_k: int = 5):
#         tipo_busqueda = await self.reasoner.analyze_query(user_query)
        
#         with self.Session() as session:
#             if tipo_busqueda == "SQL":
#                 # Construir la consulta SQL
#                 query = self._build_sql_query(user_query)
#                 results = session.execute(query).fetchall()
#             elif tipo_busqueda == "VECTOR":
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
#         # https://huggingface.co/docs/smolagents/en/examples/text_to_sql
#         pass


class PerformQuery(Tool):
    name = "text_to_sql"
    description = """
    This function will receive a sql query that uses the postgres dialect and pgvector and
    is generated from a user query.
    Alongside with that, it will return a set of keywords from the user query
    The schema of the database table 'call_data' to query to is this one:
        Columns:
        - id: INTEGER PRIMARY KEY
        - nombre: VARCHAR(255) NOT NULLABLE
            oficial name of the call
        - entidad: VARCHAR(255)
            entity that organizes the call
        - fecha_publicacion: DATETIME
            publication date of the call
        - fecha_inicio: DATETIME
            start of application period of the call
        - fecha_final: DATETIME
            end of application period of the call
        - presupuesto: FLOAT
            budget of the call
        - localidad: VARCHAR(255)
            place where the call is organized
        - url: VARCHAR(255) NOT NULLABLE
            url to information about the call
        - bases: VARCHAR(255)
            url to information about the legal basis of the call
        - beneficiario: VARCHAR(255)
            beneficiaries elegible to apply
        - tipo: VARCHAR(255)
            type of economic aid such as "subvencion", "concesion", etc.
        - keywords: VECTOR(384)
            a vector representing an embedding of the call's keywords
    The query should have a parameter `:embedding` (e.g., `1 - (keywords <=> :embedding)`) 
    if case cosine similarity search is used.
    """
    inputs = {
        "query": {
            "type": str,
            "description": "The sql query that will be performed"
        },
        "list_of_keywords": {
            "type": list,
            "description": "List of the keywords",
            "element_type": {"type": "str"},
            "required": False
        }
    }


    def forward(self, query: str, list_of_keywords: Optional[list[str]] = None) -> pd.DataFrame:
        params = {}
        if ":embedding" in query:
            params["embedding"] = self._get_keywords_and_generate_embedding(list_of_keywords)
        
        return self._execute_query(query, params)

    def _get_keywords_and_generate_embedding(self, list_of_keywords: list[str]) -> list:
        """Generate a 384-dim embedding if needed."""
        embedding_model = SentenceTransformer("jaimevera1107/all-MiniLM-L6-v2-similarity-es")
        return embedding_model.encode(list_of_keywords, convert_to_tensor=False).tolist()

    def _execute_query(self, query: str, params: dict) -> pd.DataFrame:
        """Run the query with parameters."""
        uri = f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/{os.getenv('POSTGRES_DB')}"
        engine = create_engine(uri)
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), params)
                return pd.DataFrame([dict(row._mapping) for row in result])
        except Exception as e:
            st.error(f"Database Error: {e}")
        finally:
            engine.dispose()