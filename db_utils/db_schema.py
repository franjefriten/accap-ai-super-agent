from sqlalchemy import Column, Integer, String, DateTime, Float, inspect, Null, ARRAY
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
    nombre: Mapped[str] = Column(String(255), nullable=False, default=Null)
    entidad: Mapped[str] = Column(String(255), nullable=True, default=Null)
    objetivo: Mapped[str] = Column(String(255), nullable=True, default=Null)
    duracion: Mapped[str] = Column(String(255), nullable=True, default=Null)
    nombre: Mapped[str] = Column(String(255), nullable=False, default="Desconocido")
    entidad: Mapped[str] = Column(String(255), nullable=True)
    compatibilidad: Mapped[str] = Column(String(255), nullable=True)
    duracion: Mapped[str] = Column(String(255), nullable=True)
    fecha_publicacion: Mapped[DateTime] = Column(DateTime, nullable=True, default=Null)
    fecha_inicio: Mapped[DateTime] = Column(DateTime, nullable=True, default=Null)
    fecha_final: Mapped[DateTime] = Column(DateTime, nullable=True, default=Null)
    presupuesto: Mapped[float] = Column(Float, nullable=True)
    localidad: Mapped[str] = Column(String(255), nullable=True)
    beneficiario: Mapped[str] = Column(String(255), nullable=True)
    tipo: Mapped[str] = Column(String(255), nullable=True)
    bases: Mapped[str] = Column(String(255), nullable=True)
    keywords: Mapped[list[list[float]]] = Column(ARRAY(Vector(dim=384)))
    url: Mapped[str] = Column(String(255), nullable=False, default="https://example.com")

    def __repr__(self):
        return f"<CallData(id={self.id}, titulo={self.nombre}, entidad={self.entidad}, ...)>"

    @classmethod
    def init_table(cls, engine):
        """Initialize the table in the database."""

        @event.listens_for(engine, "connect")
        def setup_vector(dbapi_connection, _):
            # 1. Create extension FIRST
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
                dbapi_connection.commit()  # Commit within the same connection

            # 2. THEN register
            register_vector(dbapi_connection)

        if not inspect(engine).has_table(cls.__tablename__):
            cls.metadata.create_all(engine)
            print(f"Table {cls.__tablename__} created.")
        else:
            print(f"Table {cls.__tablename__} already exists.")