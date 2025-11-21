from sqlalchemy import Column, Integer, String, DateTime
from .database import Base
import datetime

class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=datetime.datetime.utcnow)
    reps = Column(Integer, default=0)
