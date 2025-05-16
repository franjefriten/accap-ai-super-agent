import asyncio
import re, os, json
from datetime import datetime
from typing import List, Dict

from sqlalchemy import Null

import pandas as pd
import numpy as np

from .main import AgenticoAEI, AgenticoSNPSAP

from dotenv import load_dotenv
load_dotenv()

## AZURE IMPORTS
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient


from sentence_transformers import SentenceTransformer


def extract_key_words_azure(contenido: List[Dict]) -> List[Dict]:
    """Emplea Azure AI Services para sacar palabras claves con api y endpoint de azure.
    
    Keyword arguments:
    argument -- description
    Return: return_description
    """
    endpoint = os.environ["AZURE_AI_LANGUAGE_ENDPOINT"]
    key = os.environ["AZURE_AI_LANGUAGE_API_KEY"]

    text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key), default_language='es')
    articles = [entry["descripcion"] for entry in contenido]

    result = text_analytics_client.extract_key_phrases(articles)
    for idx, doc in enumerate(result):
        contenido[idx]["keywords"] = doc.key_phrases if hasattr(doc, "key_phrases") else []
    
    print("Palabras claves extraidas de Azure AI Services")

    return contenido


def embbed_key_words(contenido: List[Dict]):
    """Se crean embebidos con jaimevera1107/all-MiniLM-L6-v2-similarity-es"""
    model = SentenceTransformer("jaimevera1107/all-MiniLM-L6-v2-similarity-es")
    contenido = [{k: list(model.encode(sentences=v, convert_to_numpy=True).mean(axis=0)) if k == "keywords" else v for k, v in entry.items()} for entry in contenido]
    return contenido


def get_and_format_AgenticoAEI_data() -> List[Dict]:
    """
    Formatear y guardar datos de la Agencia
    Espacial de Investigación con el sistema agéntico
    """
    def format_publicacion(entry: Dict):
        """Formatear distintos formatos de fecha.
        O cuando las fechas están mal cpatadas
        
        Keyword arguments:
            entry: entrada de una convocatoria
        Return: entry 
        """
        if not hasattr(entry, "fecha_publicacion"):
            entry["fecha_publicacion"] = None
        if entry["fecha_publicacion"] == Null:
            entry["fecha_publicacion"] = None
        return entry

    def format_presupuesto(entry: Dict):
        """Formatear el presupuesto. La mayoría de veces
        vienen en formato '10.000.000,00 €' y se reconvierten
        a formato de entero
        
        Keyword arguments:
            entry: entrada de una convocatoria
        Return: entry 
        """

        if not hasattr(entry, "presupuesto"):
            entry["presupuesto"] = None  
        else:
            if entry["presupuesto"] != "":
                entry["presupuesto"] = int("".join(re.findall(r"[\d\.\,]", entry["presupuesto"])[:-3]).replace(".", "")) 
            else:
                entry["presupuesto"] = None
        return entry

    def format_compatibilidad(entry: dict):
        """Formatear el atributo de compatiblidad. Se debe introducir en la
        base de datos como un buleano y se debe formatear como tal
        
        Keyword arguments:
            entry: entrada de convocatoria con compatibilidad en forma de (NO, no, Si, si, SI, No)
        Return: entrada con compatibilidad codificada
        """
        if not hasattr(entry, "compatibilidad"):
            entry["compatibilidad"] = None
        else:
            if re.search(r"\s*no\s*", entry["compatibilidad"].lower(), re.IGNORECASE):
                entry["compatibilidad"] = False
            elif re.search(r"\s*si\s*", entry["compatibilidad"].lower(), re.IGNORECASE):
                entry["compatibilidad"] = True
            else:
                entry["compatibilidad"] = None
        return entry
    
    def parse_date(entry: Dict) -> Dict:
        """Formatear fechas para formato correcto"""
        fechas = ["fecha_inicio", "fecha_final", "fecha_publicacion"]
        for fecha in fechas:
            if entry[fecha] is None:
                entry[fecha] = None
            else:
                entry[fecha] = datetime.strptime(entry[fecha], "%d/%m/%Y")
        return entry

    # Obtenemos datos
    contenido = asyncio.run(AgenticoAEI())
    #contenido = [entry for iteracion in contenido for entry in iteracion if sum([v != '' for k, v in entry.items()]) > 3]
    contenido = list(
        map(
            format_compatibilidad,
            contenido
        )
    )
    contenido = list(
        map(
            format_publicacion,
            contenido
        )
    )
    contenido = list(
        map(
            parse_date,
            contenido
        )
    )
    contenido = list(
        map(
            format_presupuesto,
            contenido
        )
    )
    #contenido = [{**contenido, "localidad": None, "presupuesto": None} for contenido in contenido]
    # Extraer pares clave-valor
    # Azure solo permite documentos de 10 en 10.
    contenido = [extract_key_words_azure(contenido=contenido[seccion:seccion+10]) for seccion in range(0, len(contenido), 10)]
    # se aplana la lista
    contenido = [entry for iteracion in contenido for entry in iteracion]
    # El limite de la base de datos es 255 caracteres de texto
    contenido = [{k: v[:255] if k in ("beneficiario", "descripcion", "entidad", "convocatoria", "localidad", "tipo") else v for k, v in entry.items()} for entry in contenido]
    # embebimos las palabras clave
    contenido = embbed_key_words(contenido)
    return contenido


def get_and_format_AgenticoSNPSAP_data() -> List[Dict]:
    """
    Formatear y guardar datos del Sistema Nacional de Publicidad de 
    Subvenciones y Ayudas Públicas con el sistema agéntico
    """
    
    # El dataset tiene columnas:
    # Código BDNS, Mecanismo de Recuperación y Resiliencia, Administración, Departamento, Órgano, Fecha de Registro,
    # Título, Título Cooficial
    # presupuesto, fecha_inicio, fecha_final, finalidad
    def format_presupuesto(string: str):
        """Formatear el presupuesto. La mayoría de veces
        vienen en formato '10.000.000,00 €' y se reconvierten
        a formato de entero
        
        Keyword arguments:
            string: presupuesto a formatear
        Return: entry 
        """
        if string != "":
            string = int("".join(re.findall(r"[\d\.\,]", string)[:-3]).replace(".", "")) 
        else:
            string = None
        return string
    
    def format_compatibilidad(string: str):
        """Formatear el atributo de compatiblidad. Se debe introducir en la
        base de datos como un buleano y se debe formatear como tal
        
        Keyword arguments:
            string: compatibilidad en forma de (NO, no, Si, si, SI, No)
        Return: return_description
        """
        
        if re.search(r"\s*no\s*", string.lower(), re.IGNORECASE):
            return False
        elif re.search(r"\s*si\s*", string.lower(), re.IGNORECASE):
            return True
        else:
            return None

    df: pd.DataFrame = asyncio.run(AgenticoSNPSAP())
    print(df.columns)
    # renombramos las columnas
    df = df.rename(columns={
        "Departamento": "entidad",
        "Fecha de registro": "fecha_publicacion",
        "Título": "convocatoria",
        "finalidad": "descripcion"
    })
    df["presupuesto"] = df["presupuesto"].map(format_presupuesto)
    df["compatibilidad"] = df["compatibilidad"].map(format_compatibilidad)
    # Limitamos la cadena de texto a 255, que es el maximo que permite la tabla
    df["entidad"] = df["entidad"].astype('str').map(lambda x: x[:255])
    df["convocatoria"] = df["convocatoria"].astype('str').map(lambda x: x[:255])
    df["descripcion"] = df["descripcion"].astype('str').map(lambda x: x[:255])
    df["beneficiario"] = df["beneficiario"].astype('str').map(lambda x: x[:255])
    # Convertimos las fechas string a datetime
    df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"], format="%d/%m/%Y", errors="coerce")
    df["fecha_final"] = pd.to_datetime(df["fecha_final"], format="%d/%m/%Y", errors="coerce")
    df["fecha_publicacion"] = pd.to_datetime(df["fecha_publicacion"], format="%d/%m/%Y", errors="coerce")
    # Manejamos valores nulos de las fechas para que la base de datos lo maneje bien
    df[["fecha_inicio", "fecha_final", "fecha_publicacion"]] = df[["fecha_inicio", "fecha_final", "fecha_publicacion"]].map(lambda x: None if pd.isna(x) else x) #datetime.strptime("01/01/1900", "%d/%m/%Y")
    contenido = df.to_dict('records')
    # Azure solo permite documentos de 10 en 10.
    contenido = [extract_key_words_azure(contenido=contenido[seccion:seccion+10]) for seccion in range(0, len(df), 10)]
    # se aplana la lista
    contenido = [entry for iteracion in contenido for entry in iteracion]
    # embebimos las palabras clave
    contenido = embbed_key_words(contenido)
    return contenido