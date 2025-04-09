from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy import inspect
from sqlalchemy.orm import Mapped, DeclarativeBase
from pgvector.sqlalchemy import Vector

class CallData(DeclarativeBase):
    __tablename__ = "call_data"

    id: Mapped[int] = Column(Integer, primary_key=True)
    nombre: Mapped[str] = Column(String(255), nullable=False)
    entidad: Mapped[str] = Column(String(255), nullable=False)
    fecha_publicaciom: Mapped[DateTime] = Column(DateTime, nullable=False)
    fecha_inicio: Mapped[DateTime] = Column(DateTime, nullable=False)
    fecha_final: Mapped[DateTime] = Column(DateTime, nullable=False)
    presupuesto: Mapped[float] = Column(Float, nullable=False)
    descripcion = Vector(dim=300, nullable=False)

    def __repr__(self):
        return f"""
        <CallData(id={self.id}, titulo={self.nombre}, entidad_convocante={self.entidad}, 
        fecha_inicio={self.fecha_inicio}, fecha_final={self.fecha_final}, presupuesto={self.presupuesto}, 
        descripcion={self.descripcion})>
        """
    
    def init_table(self, engine):
        """Initialize the table in the database."""
        if not inspect(engine).has_table(self.__tablename__):
            self.metadata.create_all(engine)
            print(f"Table {self.__tablename__} created.")
        else:
            print(f"Table {self.__tablename__} already exists.")