from db_utils.db_schema import CallData
from crawler.get_and_format_data import *

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import text
from pgvector.psycopg import register_vector
from sqlalchemy import event

from dotenv import load_dotenv
load_dotenv()
URI_TO_DB = f"postgresql+psycopg://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@db:5432/{os.getenv("POSTGRES_DB")}"


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


if __name__ == "__main__":
    source = input("Fuente (cienciaGob, turismoGob, SNPSAP): \n")
    fetch_all_data_and_store_in_db(source)