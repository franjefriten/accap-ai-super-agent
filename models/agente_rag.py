from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine, text
from sqlglot import exp
import sqlglot
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os
from typing import Optional
from smolagents import Tool
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
import streamlit as st
import re

# class QueryReasoner(Tool):
#     name = "query_reasoner"
#     description = """
#     This function will receive a user query and will return the type of query that should be performed.
#     The possible types of queries are:
#     - SQL: A SQL query that uses the postgres dialect and pgvector
#     - VECTOR: A vector similarity search using pgvector
#     """
#     inputs = {
#         "user_query": {
#             "type": 'string',
#             "description": "The user query to analyze"
#         }
#     }
#     output_type = "string"

#     def forward(self, user_query: str) -> str:
#         # Implement your logic to determine the type of query
#         return self.analyze_query(user_query)

#     def analyze_query(self, user_query: str) -> str:
#         # Placeholder for actual analysis logic
#         if "SELECT" in user_query.upper():
#             return "SQL"
#         else:
#             return "VECTOR"


class PerformSemanticQuery(Tool):
    name = "perform_semantic_query"
    description = """
    This function will receive a user query and will perform a semantic search
    using the keywords extracted from the user query.
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
        - objetivo: VARCHAR(255)
            objective of the call
        - compatibilidad: VARCHAR(255)
            compatibility of the call with other calls
        - duration: VARCHAR(255)
            duration of the call
        - keywords: VECTOR(384)
            a vector representing an embedding of the call's keywords
    """
    inputs = {
        "user_query": {
            "type": 'string',
            "description": "The user query to analyze"
        }
    }
    output_type = "array"

    def forward(self, user_query: str) -> list[str]:
        # Implement your logic to extract keywords
        embedding = self._extract_keywords(user_query)
        query = """
        SELECT *
        FROM call_data
        ORDER BY 1 - (keywords <=> CAST(:embedding as VECTOR(384))) DESC
        LIMIT 5;
        """
        self._execute_query(query, {"embedding": embedding})

    def _extract_keywords(self, user_query: str) -> list[str]:
        endpoint = os.environ["AZURE_AI_LANGUAGE_ENDPOINT"]
        key = os.environ["AZURE_AI_LANGUAGE_API_KEY"]

        text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key), default_language='es')

        result = text_analytics_client.extract_key_phrases([user_query])
        doc = result[0]
        keywords = doc.key_phrases if hasattr(doc, "key_phrases") else []
        
        model = SentenceTransformer("jaimevera1107/all-MiniLM-L6-v2-similarity-es")
        embeddings = model.encode(sentences=keywords, convert_to_numpy=True, convert_to_tensor=False).mean(axis=0)
        return embeddings.tolist()

    def _execute_query(self, query: str, params: dict) -> pd.DataFrame:
        """Run the query with parameters."""
        uri = f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/{os.getenv('POSTGRES_DB')}"
        engine = create_engine(uri)
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), params)
                return pd.DataFrame([dict(row._mapping) for row in result]).to_records()
        except Exception as e:
            st.error(f"Database Error: {e}")
        finally:
            engine.dispose()


class PerformSQLQuery(Tool):
    name = "perform_sql_query"
    description = """
    This function will receive a sql query that uses the postgres dialect and
    is generated from a user query.
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
        - objetivo: VARCHAR(255)
            objective of the call
        - compatibilidad: VARCHAR(255)
            compatibility of the call with other calls
        - duration: VARCHAR(255)
            duration of the call
    """
    inputs = {
        "query": {
            "type": 'string',
            "description": "The sql query that will be performed"
        }
    }
    output_type = "array"


    def forward(self, query: str) -> pd.DataFrame:
        return self._execute_query(query, {})

    def _execute_query(self, query: str, params: dict) -> pd.DataFrame:
        """Run the query with parameters."""
        uri = f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/{os.getenv('POSTGRES_DB')}"
        engine = create_engine(uri)
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), params)
                return pd.DataFrame([dict(row._mapping) for row in result]).to_records()
        except Exception as e:
            st.error(f"Database Error: {e}")
        finally:
            engine.dispose()
    

class PerformHybridQuery(Tool):
    name = "perform_hybrid_query"
    description = """
    This function will receive a sql query that uses the postgres dialect
    Alongside with that, it will receive a set of keywords from the user query
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
        - objetivo: VARCHAR(255)
            objective of the call
        - compatibilidad: VARCHAR(255)
            compatibility of the call with other calls
        - duration: VARCHAR(255)
            duration of the call
    """
    inputs = {
        "query": {
            "type": 'string',
            "description": "The sql query that will be performed"
        },
        "list_of_keywords": {
            "type": 'array',
            "description": "List of the keywords",
            "element_type": {"type": "string"},
            "required": "false",
            "nullable": "true"
        }
    }
    output_type = "array"


    def forward(self, query: str, list_of_keywords: Optional[list[str]] = None) -> pd.DataFrame:
        # Parse SQL into an Abstract Syntax Tree (AST)
        tree = sqlglot.parse_one(query, dialect="postgres")
        
        # Check if ORDER BY exists
        order = tree.find(exp.Order)
        
        # Create the properly typed embedding expression
        embedding_expr = exp.Cast(
            this=exp.Placeholder(this="embedding"),
            to=exp.DataType(
                this=exp.DataType.Type.VECTOR,
                expressions=[exp.Literal.number(384)]
            )
        )
        
        # Build the similarity expression
        similarity_expr = exp.Ordered(
            this=exp.Sub(
                this=exp.Literal.number(1),
                expression=exp.Binary(
                    this=exp.Column(this=exp.Identifier(this="keywords")),
                    expression=embedding_expr,
                    operator="<=>",
                ),
            ),
            desc=True,
        )
        
        if order:
            # Annada la expresión de similitud al ORDER BY existente
            order.args["expressions"].insert(0, similarity_expr)
        else:
            # Annade nueva cláusula ORDER BY
            tree.order_by = similarity_expr
        
        # Reconvertir a SQL
        modified_query = tree.sql(dialect="postgres")

        return self._execute_query(modified_query, params={"embedding": self._get_keywords_and_generate_embedding(list_of_keywords)}) 

    def _get_keywords_and_generate_embedding(self, list_of_keywords: list[str]) -> list:
        """Generate a 384-dim embedding if needed."""
        embedding_model = SentenceTransformer("jaimevera1107/all-MiniLM-L6-v2-similarity-es")
        return embedding_model.encode(list_of_keywords, convert_to_tensor=False).tolist()

    def _execute_query(self, query: str, params: dict) -> pd.DataFrame:
        """Run the query with parameters."""
        uri = f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/{os.getenv('POSTGRES_DB')}"
        engine = create_engine(uri)
        if not self._validate_query(query):
            return pd.DataFrame().to_records()
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query), params)
                return pd.DataFrame([dict(row._mapping) for row in result]).to_records()
        except Exception as e:
            st.error(f"Database Error: {e}")
        finally:
            engine.dispose()