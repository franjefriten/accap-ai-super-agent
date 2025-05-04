from db_utils.db_schema import CallData
from crawler.clasico.get_and_format_data import *
from crawler.agentico.main import *

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import text

from typing import Annotated, Literal

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
                beneficiario=entry.get("beneficiario", None),
                tipo=entry.get("tipo", None),
                bases=entry.get("bases", None),
                url=entry.get("url", None),
            )
            session.add(call_data)
        session.commit()
        print("Content stored")
    
    return None


def fetch_all_data_and_store_in_db(
    source: Literal["cienciaGob", "turismoGob", "SNPSAP", "AEI"],
    tipo: Literal["agentico", "clasico"]
):
    if tipo == "clasico":
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
            case "AEI":
                contenido_AEI = get_and_format_AEI_data()
                send_to_db(contenido_AEI)
                print("AEI contenido volcado")
    elif tipo == "agentico":
        match source:
            case "cienciaGob":
                print("opcion no disponible")
            case "turismoGob":
                contenido_turismoGob = get_and_format_turismoGob_data()
                send_to_db(contenido_turismoGob)
                print("opcion no disponible")
            case "SNPSAP":
                contenido_SNPSAP = AgenticoSNPSAP()
                send_to_db(contenido_SNPSAP)
                print("SNPSAP contenido volcado")
            case "AEI":
                contenido_AEI = AgenticoAEI()
                send_to_db(contenido_AEI)
                print("AEI contenido volcado")


if __name__ == "__main__":
    source = input("Fuente (cienciaGob, turismoGob, SNPSAP, AEI): \n")
    tipo = input("Tipo (clasico, agentico): \n")
    if tipo not in ["clasico", "agentico"]:
        raise ValueError("Tipo no válido. Debe ser 'clasico' o 'agentico'.")
    if source not in ["cienciaGob", "turismoGob", "SNPSAP", "AEI"]:
        raise ValueError("Fuente no válida. Debe ser 'cienciaGob', 'turismoGob', 'SNPSAP' o 'AEI'.")
    fetch_all_data_and_store_in_db(source, tipo)
