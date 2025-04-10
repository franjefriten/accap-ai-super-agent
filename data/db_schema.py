from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy import inspect
from sqlalchemy.orm import Mapped, DeclarativeBase
from pgvector.sqlalchemy import Vector

class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass

class CallData(Base):
    __tablename__: str = "call_data"

    id: Mapped[int] = Column(Integer, primary_key=True)
    nombre: Mapped[str] = Column(String(255), nullable=False)
    entidad: Mapped[str] = Column(String(255), nullable=True)
    fecha_publicacion: Mapped[DateTime] = Column(DateTime, nullable=True)
    fecha_inicio: Mapped[DateTime] = Column(DateTime, nullable=True)
    fecha_final: Mapped[DateTime] = Column(DateTime, nullable=True)
    presupuesto: Mapped[float] = Column(Float, nullable=True)
    keywords: Vector = Vector(dim=300, nullable=True)
    url: Mapped[str] = Column(String(255), nullable=False)

    def __repr__(self):
        return f"""
        <CallData(id={self.id}, titulo={self.nombre}, entidad_convocante={self.entidad}, 
        fecha_inicio={self.fecha_inicio}, fecha_final={self.fecha_final}, presupuesto={self.presupuesto}, 
        descripcion={self.descripcion}, url={self.url})>
        """
    @classmethod
    def init_table(cls, engine):
        """Initialize the table in the database."""
        if not inspect(engine).has_table(cls.__tablename__):
            cls.metadata.create_all(engine)
            print(f"Table {cls.__tablename__} created.")
        else:
            print(f"Table {cls.__tablename__} already exists.")