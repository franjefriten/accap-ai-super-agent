from sqlalchemy import Column, Integer, String, DateTime, Float, inspect, Null, ARRAY, Boolean
from datetime import datetime
from sqlalchemy.orm import Mapped, DeclarativeBase
from pgvector.sqlalchemy import Vector
from pgvector.psycopg import register_vector
from sqlalchemy import event

class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass

class CallData(Base):
    __tablename__ = "call_data"

    id: Mapped[int] = Column(Integer, primary_key=True)
    objetivo: Mapped[str] = Column(String(255), nullable=True, default=Null)
    duracion: Mapped[str] = Column(String(255), nullable=True, default=Null)
    nombre: Mapped[str] = Column(String(255), nullable=False, default="Desconocido")
    entidad: Mapped[str] = Column(String(255), nullable=True)
    compatibilidad: Mapped[str] = Column(Boolean, nullable=True)
    fecha_publicacion: Mapped[DateTime] = Column(DateTime, nullable=True, default=Null)
    fecha_inicio: Mapped[DateTime] = Column(DateTime, nullable=True, default=Null)
    fecha_final: Mapped[DateTime] = Column(DateTime, nullable=True, default=Null)
    presupuesto: Mapped[float] = Column(Float, nullable=True)
    localidad: Mapped[str] = Column(String(255), nullable=True)
    beneficiario: Mapped[str] = Column(String(255), nullable=True)
    tipo: Mapped[str] = Column(String(255), nullable=True)
    bases: Mapped[str] = Column(String(255), nullable=True)
    keywords: Mapped[list[float]] = Column(Vector(dim=384))
    url: Mapped[str] = Column(String(255), nullable=False, default="https://example.com")

    def __repr__(self):
        return f"<CallData(id={self.id}, titulo={self.nombre}, entidad={self.entidad}, ...)>"

    @classmethod
    def init_table(cls, engine):
        """Inicializa la tabla de la base de datos.
        Además, crea la extensión de vector de pgvector para
        guardar los embeddings, y una función procedural de postgres
        para comparar listas de pgvectores (no empleada por coste computacional)-
        Pensada para ser ejecutada como método de clase

        Keyword arguments:

        contenido: engine
            motor para las consultas
        """

        @event.listens_for(engine, "connect")
        def setup_vector(dbapi_connection, _):
            # Se crea la extensión con un contextualizador de psycopg
            with dbapi_connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    """
                    CREATE OR REPLACE FUNCTION compute_similarity(keywords vector[], embeddings vector[]) 
                    RETURNS FLOAT AS $$
                    DECLARE
                        total_distance FLOAT := 0;
                        pair_count INTEGER := 0;
                        u vector;
                        v vector;
                    BEGIN
                        FOREACH u IN ARRAY keywords LOOP
                            FOREACH v IN ARRAY embeddings LOOP
                                total_distance := total_distance + (u <=> v);
                                pair_count := pair_count + 1;
                            END LOOP;
                        END LOOP;
                        
                        -- Return average similarity (1 - average distance)
                        RETURN 1 - (total_distance / pair_count);
                    END;
                    $$ LANGUAGE plpgsql;
                    """
                )
                dbapi_connection.commit()  # Commit los cambios en la conexión

            # Registrar el vector
            register_vector(dbapi_connection)

        # Si no existe la tabla en la bbdd
        if not inspect(engine).has_table(cls.__tablename__):
            cls.metadata.create_all(engine)
            print(f"Table {cls.__tablename__} created.")
        # Si existe
        else:
            print(f"Table {cls.__tablename__} already exists.")