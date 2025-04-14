from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import create_engine, inspect
from sqlalchemy import text
from pgvector.psycopg import register_vector
from sqlalchemy import event
from sqlalchemy.orm import Mapped
from sqlalchemy import DateTime, Column, Integer, String, ARRAY, Float
from pgvector.sqlalchemy import Vector

import os
from datetime import datetime

from crawler.get_and_format_data import *

if __name__ == "__main__":
    print(get_and_format_cienciaGob_data())