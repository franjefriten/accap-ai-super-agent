import asyncio
import re, os, json
from datetime import datetime
from typing import List, Dict

import pandas as pd
import numpy as np

from .main import turismoGob, cienciaGob, SNPSAP, AEI_selenium

from dotenv import load_dotenv
load_dotenv()

## AZURE IMPORTS
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient


from sentence_transformers import SentenceTransformer



def extract_key_words_azure(contenido):
    """Emplea Azure AI Services para sacar palabras
    claves con api y endpoint de azure.
    
    Keyword arguments:
        contenido: lista de registros
    Return: lista de registros con la palabras claves codificadas
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


def embbed_key_words(contenido: List[Dict]) -> List[Dict]:
    """Codifica las palabras mediante jaimevera1107/all-MiniLM-L6-v2-similarity-es
    
    Keyword arguments:
        contenido: lista de registros
    Return: contenido con palabras clave codificadas
    """
    
    model = SentenceTransformer("jaimevera1107/all-MiniLM-L6-v2-similarity-es")
    contenido = [{k: model.encode(sentences=v).mean(0) if k == "keywords" else v for k, v in entry.items()} for entry in contenido]
    return contenido


def get_and_format_cienciaGob_data() -> List[Dict]:
    """
    Formatear y guardar datos de cienciaGob
    """
    def extract_dates(entry: Dict) -> Dict:
        """Formatear información de las fechas de los plazos
        
        Keyword arguments:
            entry: entrada representando una convocatoria
        Return: entrada formateada
        """
        
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
    
    def format_pub_date(entry: Dict):
        """Formatear la fecha de publicacion, pues viene errónea
        con un formato no legible por la máquina

        Keyword arguments:
            entry: entrada representando una convocatoria
        Return: entrada formateada
        """
        mes_a_num = {
            "enero": 1,
            "febrero": 2,
            "marzo": 3,
            "abril": 4,
            "mayo": 5,
            "junio": 6,
            "julio": 7,
            "agosto": 8,
            "septiembre": 9,
            "octubre": 10,
            "noviembre": 11,
            "diciembre": 12
        }
        if "fecha_publicacion" in entry:
            aux_date = entry["fecha_publicacion"].split(" ")
            aux_date[0] = str(mes_a_num[aux_date[0]])
            aux_date = "/".join(aux_date)
            entry["fecha_publicacion"] = datetime.strptime(aux_date, "%m/%Y")
        return entry

    contenido = asyncio.run(cienciaGob())
    contenido = list(
        map(
            extract_dates,
            contenido
        )
    )
    contenido = list(
        map(
            format_pub_date,
            contenido
        )
    )
    contenido = [{**contenido, "localidad": None, "presupuesto": None} for contenido in contenido]
    # extraer palabras clave
    contenido = extract_key_words_azure(contenido)
    # codificar palabras clave
    contenido = embbed_key_words(contenido=contenido)
    return contenido


def get_and_format_AEI_data():
    """
    Formatear y guardar datos de la Agencia Estatal de Investigación
    """
    def extract_dates(entry):
        """Formatear información de las fechas de los plazos
        
        Keyword arguments:
            entry: entrada representando una convocatoria
        Return: entrada formateada
        """
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

    def format_presupuesto(entry: Dict) -> Dict:
        """Formatear el presupuesto. La mayoría de veces
        vienen en formato '10.000.000,00 €' y se reconvierten
        a formato de entero
        
        Keyword arguments:
            entry: entrada de una convocatoria
        Return: entry 
        """
        
        if entry["presupuesto"] != "":
            entry["presupuesto"] = int("".join(re.findall(r"[\d\.\,]", entry["presupuesto"])[:-3]).replace(".", "")) 
        else:
            entry["presupuesto"] = None
        return entry
        
    contenido = AEI_selenium()
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
    # Extraer palabras clave
    contenido = [extract_key_words_azure(contenido=contenido[seccion:seccion+10]) for seccion in range(0, len(contenido), 10)]
    # aplanar la lista
    contenido = [entry for iteracion in contenido for entry in iteracion]
    contenido = [{k: v[:255] if k == "beneficiario" else v for k, v in entry.items()} for entry in contenido]
    contenido = embbed_key_words(contenido)
    return contenido


def get_and_format_turismoGob_data():
    """
    Formatear y guardar datos del Ministerio de Turismo
    """
    def extract_dates(entry: Dict) -> Dict:
        """Formatear información de las fechas de los plazos
        
        Keyword arguments:
            entry: entrada representando una convocatoria
        Return: entrada formateada
        """
        pat = r"\d{1,2}\/\d{2}\/\d{2}"
        fechas = re.findall(pat, entry["plazos"])
        entry["fecha_inicio"] = datetime.strptime(fechas[0], "%d/%m/%y")
        entry["fecha_final"] = datetime.strptime(fechas[1], "%d/%m/%y")
        entry.pop("plazos")
        return entry
        
    contenido = asyncio.run(turismoGob())
    contenido = list(
        map(
            extract_dates,
            contenido
        )
    )
    contenido = [{**entry, "localidad": None, "presupuesto": None} for entry in contenido]
    # Extraer palabras clave
    contenido = extract_key_words_azure(contenido)
    # Embeber palabras clave
    contenido = embbed_key_words(contenido=contenido)
    return contenido

def get_and_format_SNPSAP_data():
    """
    Formatear y guardar datos del Sistema Nacional de Publicidad de 
    Subvenciones y Ayudas Públicas con el sistema clásico
    """
    # El dataset tiene columnas:
    # Código BDNS, Mecanismo de Recuperación y Resiliencia, Administración, Departamento, Órgano, Fecha de Registro,
    # Título, Título Cooficial
    # presupuesto, fecha_inicio, fecha_final, finalidad
    def format_presupuesto(string: str) -> int:
        """Formatear el presupuesto. La mayoría de veces
        vienen en formato '10.000.000,00 €' y se reconvierten
        a formato de entero
        
        Keyword arguments:
            string: cadena de texto que representa el presupuesto
        Return: presupuesto en entero
        """
        if string != "":
            string = int("".join(re.findall(r"[\d\.\,]", string)[:-3]).replace(".", "")) 
        else:
            string = None
        return string
    df: pd.DataFrame = asyncio.run(SNPSAP())
    # Seleccionar columnas que precisamos
    df = df[[
        "Departamento",
        "Fecha de registro",
        "Título",
        "presupuesto",
        "fecha_inicio",
        "fecha_final",
        "finalidad",
        "url",
        "localidad",
        "bases",
        "beneficiario",
        "tipo"
    ]]
    # Renombrar columnas
    df = df.rename(columns={
        "Departamento": "entidad",
        "Fecha de registro": "fecha_publicacion",
        "Título": "convocatoria",
        "finalidad": "descripcion"
    })
    df["presupuesto"] = df["presupuesto"].map(format_presupuesto)
    # convertir las cadenas de texto de fechas a datetime
    df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"], format="%d/%m/%Y", errors="coerce")
    df["fecha_final"] = pd.to_datetime(df["fecha_final"], format="%d/%m/%Y", errors="coerce")
    df["fecha_publicacion"] = pd.to_datetime(df["fecha_publicacion"], format="%d/%m/%Y", errors="coerce")
    # Manejar valores nulos
    df[["fecha_inicio", "fecha_final", "fecha_publicacion"]] = df[["fecha_inicio", "fecha_final", "fecha_publicacion"]].map(lambda x: datetime.strptime("01/01/1900", "%d/%m/%Y") if x is pd.NaT else x)
    contenido = df.to_dict('records')
    # Obtener palabras clave, azure solo deja 10 en 10 documentos
    contenido = [extract_key_words_azure(contenido=contenido[seccion:seccion+10]) for seccion in range(0, len(df), 10)]
    # aplanar la lista por el paso anterior
    contenido = [entry for iteracion in contenido for entry in iteracion]
    # embeber palabras clave
    contenido = embbed_key_words(contenido)
    return contenido
    