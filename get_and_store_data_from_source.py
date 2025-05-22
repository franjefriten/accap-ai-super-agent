from db_utils.db_schema import CallData
from crawler.clasico.get_and_format_data import *
from crawler.agentico.get_and_format_data import *

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import text

from typing import Annotated, Literal

from dotenv import load_dotenv
load_dotenv()


def send_to_db(contenido: list[dict], uri_type: Literal["localhost", "db"]):
    """Función general para guardar los datos en la base de datos.
    Se conecta a la base de datos PostgreSQL y guarda los datos extraídos y formateados.
    Se utiliza SQLAlchemy para la conexión y manipulación de la base de datos.
    
    Keyword arguments:

    contenido: list[dict]
        lista de entradas de convocatorias ya procesadas para ser guardadas
    uri_type: "localhost" o "db"
        se refiere a si se envía desde fuera de docker (localhost) o desde
        dentro del contenedor docker (db)
   
    Return: None
    """
    if uri_type.lower() == "localhost":
        URI_TO_DB = f"postgresql+psycopg://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@localhost:5432/{os.getenv("POSTGRES_DB")}"
    elif uri_type.lower() == "db":
        URI_TO_DB = f"postgresql+psycopg://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@db:5432/{os.getenv("POSTGRES_DB")}"
    else:
        raise Exception
        
    
    engine = create_engine(URI_TO_DB)
    Session = sessionmaker(bind=engine)
    # Si no existe la tabla, se crea y el vector de embeddings
    CallData.init_table(engine)

    with Session() as session:
        # Crear extension de vector
        session.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
        session.commit()

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
                compatibilidad=entry.get("compatibilidad", None),
                objetivo=entry.get("objetivo", None),
                duracion=entry.get("duracion", None),
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
    """Esta función está pensada para ser ejecutada por línea de comandos.
    Realiza la captación y procesado de datos según que fuente se escoja por terminal.
    Llama a una función para volcar datos en la base de datos
    
    Keyword arguments:
        source: fuente, solo se permiten las dejadas en la pista de dato
        tipo: emplear el agente agentico o clasico, agentico obtiene más datos pero suele tardar más
    
    Return: None
    """
    
    if tipo == "clasico":
        match source:
            case "cienciaGob":
                contenido_cienciaGob = get_and_format_cienciaGob_data()
                send_to_db(contenido_cienciaGob, uri_type='localhost')
                print("cienciaGob contenido volcado")
            case "turismoGob":
                contenido_turismoGob = get_and_format_turismoGob_data()
                send_to_db(contenido_turismoGob, uri_type='localhost')
                print("turismoGob contenido volcado")
            case "SNPSAP":
                contenido_SNPSAP = get_and_format_SNPSAP_data()
                send_to_db(contenido_SNPSAP, uri_type='localhost')
                print("SNPSAP contenido volcado")
            case "AEI":
                contenido_AEI = get_and_format_AEI_data()
                send_to_db(contenido_AEI, uri_type='localhost')
                print("AEI contenido volcado")
    elif tipo == "agentico":
        match source:
            case "cienciaGob":
                print("opcion no disponible")
            case "turismoGob":
                print("opcion no disponible")
            case "SNPSAP":
                contenido_SNPSAP = get_and_format_AgenticoSNPSAP_data()
                send_to_db(contenido_SNPSAP, uri_type='localhost')
                print("SNPSAP contenido volcado")
            case "AEI":
                contenido_AEI = get_and_format_AgenticoAEI_data()
                send_to_db(contenido_AEI, uri_type='localhost')
                print("AEI contenido volcado")


if __name__ == "__main__":
    if not os.path.exists("./downloads/pdf"):
        os.makedirs(name="./downloads/pdf")
    source = input("Fuente (cienciaGob, turismoGob, SNPSAP, AEI): \n")
    tipo = input("Tipo (clasico, agentico): \n")
    if tipo not in ["clasico", "agentico"]:
        raise ValueError("Tipo no válido. Debe ser 'clasico' o 'agentico'.")
    if source not in ["cienciaGob", "turismoGob", "SNPSAP", "AEI"]:
        raise ValueError("Fuente no válida. Debe ser 'cienciaGob', 'turismoGob', 'SNPSAP' o 'AEI'.")
    fetch_all_data_and_store_in_db(source, tipo)
