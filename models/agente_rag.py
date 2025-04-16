from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine, text
from smolagents import Tool
from sentence_transformers import SentenceTransformer
import numpy as np
import os

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


class PerformStandardSQLQuerying(Tool):
    name = "perform_standard_sql_querying"
    description = """
    This function is used to do standard sql querying (postgres dialect). Returns a string representation
    of the result. The table is named 'call_data' and has data about subsidies and public calls for economic aid. The
    description is as follows:
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
        - keywords: VECTOR(384)
            a vector representing an embedding of the call's keywords
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be correct SQL."
        }
    }
    output_type = "string"


    def forward(
        self,
        query: str,
    ):
        uri: str = f"postgresql+psycopg://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@localhost:5432/{os.getenv("POSTGRES_DB")}",
        output = "Estas son las entradas que más se pueden ajustar a tu consulta:"
        engine = create_engine(uri)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            result = session.execute(text(query))
            for row in result:
                output += "\n\t*" + str(row)
        output += "\n¡Espero que te sirva!"
        return output


class PerformSemanticSearchQuerying(Tool):
    name = "perform_semantic_search_querying"
    description = """
    This function is used to do sql querying (postgres dialect) through semantic search. Returns a string representation
    of the result. The table is named 'call_data' and has data about subsidies and public calls for economic aid. The
    description is as follows:
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
        - keywords: VECTOR(384)
            a vector representing an embedding of the call's keywords
    """
    inputs = {
        "keywords": {
            "type": "array",
            "description": "python list of strings that contains the main keywords og the user query"
        },
    }
    output_type = "string"


    def forward(
        self,
        keywords: list[str],
    ):
        uri: str = f"postgresql+psycopg://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@localhost:5432/{os.getenv("POSTGRES_DB")}",
        output = "Estas son las entradas que más se pueden ajustar a tu consulta:"
        engine = create_engine(uri)
        Session = sessionmaker(bind=engine)
        model = SentenceTransformer("jaimevera1107/all-MiniLM-L6-v2-similarity-es")
        query_embedding = np.mean(model.encode(keywords) , axis=0)
        with Session() as session:
            result = session.execute(
                text("""
                SELECT id, nombre, entidad, 
                    1 - (keywords <=> :embedding) as similarity
                FROM call_data
                ORDER BY similarity DESC
                LIMIT 10
                """), 
                {"embedding": query_embedding}
            ).fetchall()
            for row in result:
                output += "\n\t*" + str(row)
        output += "\n¡Espero que te sirva!"
        return output