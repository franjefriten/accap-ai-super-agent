from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os
from db_utils.db_schema import CallData
from crawler.get_and_format_data import *


def send_to_db(contenido):
    """Función general para guardar los datos en la base de datos.
    Se conecta a la base de datos PostgreSQL y guarda los datos extraídos y formateados.
    Se utiliza SQLAlchemy para la conexión y manipulación de la base de datos.
    
    Keyword arguments:
    argument -- description
    Return: return_description
    """
    
    engine = create_engine(
        f"postgresql+psycopg://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@localhost:5432/{os.getenv("POSTGRES_DB")}"
    )
    Session = sessionmaker(bind=engine)
    CallData.init_table(engine)
    with Session() as session:
        # Assuming `contenido` is a list of dictionaries with the data to be inserted
        for entry in contenido:
            call_data = CallData(
                nombre=entry["convocatoria"],
                entidad=entry["entidad"],
                fecha_publicaciom=entry["fecha_publicacion"],
                fecha_inicio=entry["fecha_inicio"],
                fecha_final=entry["fecha_final"],
                presupuesto=entry["presupuesto"],
                keywords=entry["keywords"],
                localidad=entry["localidad"],
                url=entry["url"],
            )
            session.add(call_data)
        session.commit()
        print("Content stored")
    
    return None


def fetch_all_data_and_store_in_db():
    contenido_cienciaGob = get_and_format_cienciaGob_data()
    contenido_turismoGob = get_and_format_turismoGob_data()
    contenido_SNPSAP = get_and_format_SNPSAP_data()
    contenido = {**contenido_cienciaGob, **contenido_turismoGob, **contenido_SNPSAP}
    send_to_db(contenido=contenido)