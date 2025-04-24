import asyncio
import re, os, json
from datetime import datetime

import pandas as pd
import numpy as np

from crawler.main import turismoGob, cienciaGob, SNPSAP, AEI, AEI_selenium

from dotenv import load_dotenv
load_dotenv()

## AZURE IMPORTS
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient


from sentence_transformers import SentenceTransformer

# class Base(DeclarativeBase):
    # """Base class for SQLAlchemy models."""
    # pass
# 
# class CallData(Base):
    # __tablename__: str = "call_data"
# 
    # id: Mapped[int] = Column(Integer, primary_key=True)
    # nombre: Mapped[str] = Column(String(255), nullable=False)
    # entidad: Mapped[str] = Column(String(255), nullable=True)
    # localidad: Mapped[str] = Column(String(255), nullable=True)
    # fecha_publicacion: Mapped[DateTime] = Column(DateTime, nullable=True)
    # fecha_inicio: Mapped[DateTime] = Column(DateTime, nullable=True)
    # fecha_final: Mapped[DateTime] = Column(DateTime, nullable=True)
    # presupuesto: Mapped[float] = Column(Float, nullable=True)
    # url: Mapped[str] = Column(String(255), nullable=False)
    # keywords: Vector = Vector(dim=300)
    # 
# 
    # def __repr__(self):
        # return f"""
        # <CallData(id={self.id}, titulo={self.nombre}, entidad_convocante={self.entidad}, 
        # fecha_inicio={self.fecha_inicio}, fecha_final={self.fecha_final}, presupuesto={self.presupuesto}, 
        # descripcion={self.descripcion}, url={self.url})>
        # """
    # @classmethod
    # def init_table(cls, engine):
        # """Initialize the table in the database."""
        # if not inspect(engine).has_table(cls.__tablename__):
            # cls.metadata.create_all(engine)
            # print(f"Table {cls.__tablename__} created.")
        # else:
            # print(f"Table {cls.__tablename__} already exists.")


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
    contenido = [{k: [model.encode(sentences=word) for word in v] if k == "keywords" else v for k, v in entry.items()} for entry in contenido]
    return contenido


def get_and_format_cienciaGob_data():
    """
    Format and store data from cienciaGob.
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
    
    def format_pub_date(entry):
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


    # contenido = [
        # {'convocatoria': 'Distinción Ciudad de la Ciencia y la Innovación 2024', 'fecha_publicacion': 'julio 2024', 'plazos': 'Plazos de Solicitud:\n\t                    Comienzo: \n\t                    16/09/24\n\t                    -\n\t                    Fin: \n\t                    28/10/24', 'entidad': 'Secretaría General de Innovación', 'descripcion': 'La distinción «Ciudad de la Ciencia y la Innovación» se otorga a las ciudades que se distinguen en el apoyo a la innovación en sus territorios, definiendo políticas, potenciando estructuras, instituciones y empresas locales con un fuerte componente científico, tecnológico e innovador.', 'url': 'https://www.ciencia.gob.es/Convocatorias/2024/DCCI2024.html'},
        # {'convocatoria': 'Ayudas para la realización de estudios de máster en Estados Unidos de América. Curso 2025-2026', 'fecha_publicacion': 'enero 2025', 'plazos': 'Plazos de Solicitud:\n\t                    Comienzo: \n\t                    16/01/25\n\t                    -\n\t                    Fin: \n\t                    6/02/25', 'entidad': 'Subdirección General de Formación del Profesorado Universitario y Gestión de Programas de Ayuda', 'descripcion': 'Esta convocatoria tiene por finalidad la concesión de ayudas para la realización de estudios de máster en universidades e instituciones de educación superior, acreditadas para impartir dichos estudios en Estados Unidos. Y cuenta con la participación de la Comisión Fulbright, tanto en la convocatoria y selección de las personas beneficiarias como durante el período de disfrute de las ayudas, garantizando a las personas beneficiarias el apoyo de dicha institución durante la realización de sus estudios.', 'url': 'https://www.ciencia.gob.es/Convocatorias/2025/Master_EEUU_25_26.html'},
        # {'convocatoria': 'Premios Nacionales de Innovación y de Diseño 2025', 'fecha_publicacion': 'marzo 2025', 'plazos': 'Plazos de Solicitud:\n\t                    desde las 00:00 horas del día 07/03/2025  hasta las 15:00 horas del 24/04/2025 :\n\t                    Comienzo: \n\t                    7/03/25\n\t                    -\n\t                    Fin: \n\t                    24/04/25', 'entidad': 'Subdirección General de Políticas de Innovación', 'descripcion': 'Con la concesión de los Premios Nacionales de Innovación y de Diseño se pretende distinguir a aquellas personas y entidades (instituciones y organizaciones) que han hecho de la innovación un elemento indispensable en el desarrollo de su estrategia profesional y de su crecimiento empresarial. Asimismo, se trata de galardonar a los profesionales y empresas que han contribuido significativamente al incremento del prestigio del diseño español y a las entidades que, incorporándolo a su estrategia empresarial, han demostrado que el diseño es una potente palanca de la innovación y la competitividad.', 'url': 'https://www.ciencia.gob.es/Convocatorias/2025/PNID2025.html'},
        # {'convocatoria': 'Línea FID 4 2024', 'fecha_publicacion': 'febrero 2024', 'plazos': 'Plazos de Solicitud:\n\t                    El plazo de solicitudes comenzará el día 04/03/2024 a las 00:00 horas y terminará el 09/04/2024 a las 14:59:59 horas:\n\t                    Comienzo: \n\t                    4/03/24\n\t                    -\n\t                    Fin: \n\t                    9/04/24', 'entidad': 'Ministerio de Ciencia, Innovación y Universidades', 'url': 'https://www.ciencia.gob.es/Convocatorias/FEDER2021-2027/FID42024.html'},
        # {'convocatoria': 'Subvenciones para estudiantes universitarios afectados por la DANA', 'fecha_publicacion': 'noviembre 2024', 'plazos': 'Plazos de Solicitud:\n\t                    Comienzo: \n\t                    1/12/24\n\t                    -\n\t                    Fin: \n\t                    30/12/24', 'entidad': 'Subdirección General de Formación del Profesorado Universitario y Gestión de Programas de Ayuda', 'descripcion': 'De acuerdo con lo establecido en el artículo 22.2.b) de la Ley 38/2003, de 17 de noviembre, y en el marco de la situación de emergencia provocada por la DANA, se establece la concesión directa de subvenciones para estudiantes universitarios.', 'url': 'https://www.ciencia.gob.es/Convocatorias/2024/SubvencionesDANA.html'},
        # {'convocatoria': 'Ayudas para la preparación y gestión de proyectos europeos y facilitar la atracción de talento internacional 2025', 'fecha_publicacion': 'marzo 2025', 'plazos': 'Plazos de Solicitud:\n\t                    El plazo finaliza el 10 de abril a las 14:00 horas:\n\t                    Comienzo: \n\t                    20/03/25\n\t                    -\n\t                    Fin: \n\t                    10/04/25', 'entidad': 'Subdivisión de Planificación y Gestión Administrativa. Agencia Estatal de Investigación', 'descripcion': 'El objeto de la convocatoria es, por un lado, reforzar las estructuras de las instituciones solicitantes y los conocimientos necesarios para la promoción, preparación, apoyo y gestión de proyectos internacionales con el fin de mejorar sus posibilidades de participación en proyectos del Programa Horizonte Europa y, por otro, dotar a las instituciones de herramientas para la captación de personal investigador o técnico internacional, su incorporación y consolidación.', 'url': 'https://www.ciencia.gob.es/Convocatorias/2025/GPE2025.html'},
        # {'convocatoria': 'Certificado BIOPYME 2025', 'fecha_publicacion': 'febrero 2025', 'plazos': 'Plazos de Solicitud:\n\t                    Comienzo: \n\t                    1/03/25\n\t                    -\n\t                    Fin: \n\t                    31/03/25', 'descripcion': 'La Orden regula la expedición de una certificación para PYME de alta intensidad inversora en I+D+I, para acompañar a la presentación de solicitudes de aplazamiento o fraccionamiento con dispensa de garantía a la Delegación de Hacienda competente, identificando las entidades y deudas que se acogen a esta orden. El objetivo es proporcionar a las Delegaciones de Economía y Hacienda documentación que les permita analizar el carácter transitorio de las dificultades económico-financieras de las empresas y su futura viabilidad.', 'url': 'https://www.ciencia.gob.es/Convocatorias/2025/BIOPYME2025.html'},
        # {'convocatoria': 'Proyectos de Colaboración Internacional PCI 2025-1', 'fecha_publicacion': 'marzo 2025', 'plazos': 'Plazos de Solicitud:\n\t                    El plazo comienza el 19-03-2025 y termina el 01-04-2025 a las 14:00 hora peninsular española:\n\t                    Comienzo: \n\t                    19/03/25\n\t                    -\n\t                    Fin: \n\t                    1/04/25', 'entidad': 'Subdivisión de Planificación y Gestión Administrativa. Agencia Estatal de Investigación', 'descripcion': 'España, a través de la Agencia Estatal de Investigación, es miembro de consorcios transnacionales, tanto en el Espacio Europeo de Investigación como en el ámbito internacional, a través de la firma de acuerdos y memorandos de entendimiento en los que se compromete a financiar proyectos colaborativos transnacionales, bilaterales y multilaterales, de alto nivel científico-técnico y en los que la participación de equipos de investigación españoles es relevante.', 'url': 'https://www.ciencia.gob.es/Convocatorias/2025/PCI_2025-1.html'},
        # {'convocatoria': 'Agentes Locales de Innovación 2025', 'fecha_publicacion': 'abril 2025', 'plazos': 'Plazos de Solicitud:\n\t                    Comienzo: \n\t                    31/03/25\n\t                    -\n\t                    Fin: \n\t                    22/04/25', 'entidad': 'Subdirección General de Fomento de la Innovación', 'descripcion': 'Ayudas destinadas a la cofinanciación de la incorporación o mantenimiento, en su caso, de agentes locales de innovación por parte de los ayuntamientos de las ciudades miembros de la Red Innpulso.', 'url': 'https://www.ciencia.gob.es/Convocatorias/2025/ALI2025.html'},
        # {'convocatoria': 'Ayudas para contratos para la formación de doctores y doctoras en empresas y otras entidades (Doctorados Industriales) 2024-2', 'fecha_publicacion': 'diciembre 2024', 'plazos': 'Plazos de Solicitud:\n\t                    el plazo termina a las 14.00 hora peninsular española:\n\t                    Comienzo: \n\t                    21/01/25\n\t                    -\n\t                    Fin: \n\t                    6/03/25', 'entidad': 'Subdivisión de Planificación y Gestión Administrativa. Agencia Estatal de Investigación', 'descripcion': 'Ayudas de una duración de cuatro años para distintos tipos de entidades cuya finalidad es promover la realización de proyectos de investigación industrial o de desarrollo experimental, en los que se enmarque una tesis doctoral, a fin de favorecer la inserción laboral de personal investigador desde los inicios de sus carreras profesionales, contribuir a la empleabilidad de estos investigadores e investigadoras y promover la incorporación de talento en el tejido productivo para elevar la competitividad del mismo.', 'url': 'https://www.ciencia.gob.es/Convocatorias/2024/DIN2024-2.html'}
        # ]
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
    contenido = extract_key_words_azure(contenido)
    contenido = embbed_key_words(contenido=contenido)
    return contenido


def get_and_format_AEI_data():
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
    contenido = [extract_key_words_azure(contenido=contenido[seccion:seccion+10]) for seccion in range(0, len(contenido), 10)]
    contenido = [entry for iteracion in contenido for entry in iteracion]
    contenido = [{k: v[:255] if k == "beneficiario" else v for k, v in entry.items()} for entry in contenido]
    contenido = embbed_key_words(contenido)
    return contenido


def get_and_format_turismoGob_data():
    """
    Format and store data from turismoGob.
    """
    def extract_dates(entry):
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
    contenido = extract_key_words_azure(contenido)
    contenido = embbed_key_words(contenido=contenido)
    return contenido

def get_and_format_SNPSAP_data():
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
    df: pd.DataFrame = asyncio.run(SNPSAP())
    print(df.columns)
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
    