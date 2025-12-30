from sqlalchemy import create_engine, Column, String, Float, DateTime, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import enum

Base = declarative_base()

class CovenantStatusEnum(enum.Enum):
    COMPLIANT = "compliant"
    WARNING = "warning"
    BREACH = "breach"

class FinancialMetricDB(Base):
    __tablename__ = "financial_metrics"
    
    id = Column(String, primary_key=True)
    borrower_id = Column(String, index=True)
    date = Column(DateTime, index=True)
    ebitda = Column(Float)
    total_debt = Column(Float)
    cash_flow = Column(Float)
    interest_expense = Column(Float)
    capex = Column(Float)
    revenue = Column(Float)

class CovenantTestDB(Base):
    __tablename__ = "covenant_tests"
    
    test_id = Column(String, primary_key=True)
    covenant_id = Column(String, index=True)
    borrower_id = Column(String, index=True)
    test_date = Column(DateTime, index=True)
    actual_value = Column(Float)
    threshold_value = Column(Float)
    status = Column(SQLEnum(CovenantStatusEnum))
    breach_margin = Column(Float, nullable=True)

class LegalDocumentDB(Base):
    __tablename__ = "legal_documents"
    
    document_id = Column(String, primary_key=True)
    borrower_id = Column(String, index=True)
    document_type = Column(String)
    breach_details = Column(String)
    generated_text = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Database setup
DATABASE_URL = "sqlite:///./lma_pulse.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
