from sqlalchemy import Column, Integer, String, Text, JSON, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True)
    url = Column(String, unique=True, index=True, nullable=False)
    title = Column(String)
    date = Column(String)
    text = Column(Text, nullable=False)
    media = Column(JSON, default=dict)
    scraped_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String)

    def to_dict(self):
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "date": self.date,
            "text": self.text,
            "media": self.media,
            "scraped_at": self.scraped_at.isoformat() if self.scraped_at else None,
            "source": self.source
        }