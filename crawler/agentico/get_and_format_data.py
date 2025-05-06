import asyncio
import re, os, json
from datetime import datetime

import pandas as pd
import numpy as np

from .main import AgenticoAEI, AgenticoSNPSAP

from dotenv import load_dotenv
load_dotenv()

## AZURE IMPORTS
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient


from sentence_transformers import SentenceTransformer


def extract_key_words_azure(contenido):
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


def embbed_key_words(contenido: list[dict]):
    model = SentenceTransformer("jaimevera1107/all-MiniLM-L6-v2-similarity-es")
    contenido = [{k: list(model.encode(sentences=v, convert_to_numpy=True).mean(axis=0)) if k == "keywords" else v for k, v in entry.items()} for entry in contenido]
    return contenido


def get_and_format_AgenticoAEI_data():
    """
    Format and store data from AEI.
    """
    def extract_dates(entry):
        pat = r"\d{1,2}\/\d{2}\/\d{2}"
        fechas = re.findall(pat, entry["plazos"])
        if len(fechas) >= 2:
            entry["fecha_inicio"] = datetime.strptime(fechas[0], "%d/%m/%y")
            entry["fecha_final"] = datetime.strptime(fechas[1], "%d/%m/%y")
            entry.pop("plazos")
            return entry
        elif len(fechas) == 1:
            entry["fecha_inicio"] = datetime.strptime(fechas[0], "%d/%m/%y")
            entry["fecha_final"] = datetime.strptime(fechas[0], "%d/%m/%y")
            entry.pop("plazos")
            return entry
        else:
            entry["fecha_inicio"] = None
            entry["fecha_final"] = None
            entry.pop("plazos")
            return entry

    def format_presupuesto(entry):
        if entry["presupuesto"] != "":
            entry["presupuesto"] = int("".join(re.findall(r"[\d\.\,]", entry["presupuesto"])[:-3]).replace(".", "")) 
        else:
            entry["presupuesto"] = None
        return entry
        
    contenido = asyncio.run(AgenticoAEI())
    #contenido = [entry for iteracion in contenido for entry in iteracion if sum([v != '' for k, v in entry.items()]) > 3]
    contenido = list(
        map(
            extract_dates,
            contenido
        )
    )
    contenido = list(
        map(
            format_presupuesto,
            contenido
        )
    )
    contenido = [{**contenido, "localidad": None, "presupuesto": None} for contenido in contenido]
    contenido = [extract_key_words_azure(contenido=contenido[seccion:seccion+10]) for seccion in range(0, len(contenido), 10)]
    contenido = [entry for iteracion in contenido for entry in iteracion]
    contenido = [{k: v[:255] if k == "beneficiario" else v for k, v in entry.items()} for entry in contenido]
    contenido = embbed_key_words(contenido)
    return contenido


def get_and_format_AgenticoSNPSAP_data():
    # El dataset tiene columnas:
    # Código BDNS, Mecanismo de Recuperación y Resiliencia, Administración, Departamento, Órgano, Fecha de Registro,
    # Título, Título Cooficial
    # presupuesto, fecha_inicio, fecha_final, finalidad
    def format_presupuesto(string):
        if string != "":
            string = int("".join(re.findall(r"[\d\.\,]", string)[:-3]).replace(".", "")) 
        else:
            string = None
        return string
    df: pd.DataFrame = asyncio.run(AgenticoSNPSAP())
    print(df.columns)
    df = df.rename(columns={
        "Departamento": "entidad",
        "Fecha de registro": "fecha_publicacion",
        "Título": "convocatoria",
        "finalidad": "descripcion"
    })
    df["presupuesto"] = df["presupuesto"].map(format_presupuesto)
    df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"], format="%d/%m/%Y", errors="coerce")
    df["fecha_final"] = pd.to_datetime(df["fecha_final"], format="%d/%m/%Y", errors="coerce")
    df["fecha_publicacion"] = pd.to_datetime(df["fecha_publicacion"], format="%d/%m/%Y", errors="coerce")
    df[["fecha_inicio", "fecha_final", "fecha_publicacion"]] = df[["fecha_inicio", "fecha_final", "fecha_publicacion"]].map(lambda x: datetime.strptime("01/01/1900", "%d/%m/%Y") if x is pd.NaT else x)
    contenido = df.to_dict('records')
    contenido = [extract_key_words_azure(contenido=contenido[seccion:seccion+10]) for seccion in range(0, len(df), 10)]
    contenido = [entry for iteracion in contenido for entry in iteracion]
    contenido = embbed_key_words(contenido)
    return contenido