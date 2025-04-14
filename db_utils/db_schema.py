from sqlalchemy import Column, Integer, String, DateTime, Float, inspect
from sqlalchemy.orm import Mapped, DeclarativeBase, sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY

class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass

class CallData(Base):
    __tablename__ = "call_data"

    id: Mapped[int] = Column(Integer, primary_key=True)
    nombre: Mapped[str] = Column(String(255), nullable=False, default=None)
    entidad: Mapped[str] = Column(String(255), nullable=True, default=None)
    fecha_publicacion: Mapped[DateTime] = Column(DateTime, nullable=True, default=None)
    fecha_inicio: Mapped[DateTime] = Column(DateTime, nullable=True, default=None)
    fecha_final: Mapped[DateTime] = Column(DateTime, nullable=True, default=None)
    presupuesto: Mapped[float] = Column(Float, nullable=True, default=None)
    localidad: Mapped[str] = Column(String(255), nullable=True, default=None)
    keywords: Mapped[list[float]] = Column(Vector(dim=384))
    url: Mapped[str] = Column(String(255), nullable=False)

    def __repr__(self):
        return f"<CallData(id={self.id}, titulo={self.nombre}, entidad={self.entidad}, ...)>"

    @classmethod
    def init_table(cls, engine):
        """Initialize the table in the database."""
        if not inspect(engine).has_table(cls.__tablename__):
            cls.metadata.create_all(engine)
            print(f"Table {cls.__tablename__} created.")
        else:
            print(f"Table {cls.__tablename__} already exists.")