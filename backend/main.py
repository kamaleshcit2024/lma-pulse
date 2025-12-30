from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import uuid
from datetime import datetime

from models import *
from database import get_db, init_db
from covenant_engine import CovenantEngine
from ai_engine import LMALegalEngine
from mock_erp import MockERPConnector
from forecasting import forecaster

app = FastAPI(title="LMA Pulse API", version="1.0.0")

# CORS for Electron app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()
    # Train forecasting model on mock data
    try:
        print("Training PerforatedAI Forecasting Model...")
        history = MockERPConnector.generate_historical_data(24)
        forecaster.train_mock_model(history)
        print("Forecasting Model Trained.")
    except Exception as e:
        print(f"WARNING: PerforatedAI Forecasting failed to train: {e}")
        print("Application will continue without forecasting features.")


# ==================== ENDPOINTS ====================

@app.get("/")
def root():
    return {"message": "LMA Pulse API - Covenant Monitoring System", "version": "1.0.0"}

@app.get("/borrower/profile", response_model=BorrowerProfile)
def get_borrower_profile():
    """Get the borrower profile for Solaris Energy"""
    profile = MockERPConnector.BORROWER_PROFILE
    return BorrowerProfile(**profile)

@app.get("/financial/latest", response_model=FinancialMetric)
def get_latest_financials():
    """Get the latest financial metrics from the ERP system"""
    return MockERPConnector.get_latest_metrics()

@app.get("/financial/history", response_model=List[FinancialMetric])
def get_financial_history(months: int = 12):
    """Get historical financial data"""
    return MockERPConnector.generate_historical_data(months)

@app.get("/covenants/list", response_model=List[Covenant])
def get_covenants():
    """Get all covenants defined in the facility agreement"""
    return CovenantEngine.STANDARD_COVENANTS

@app.post("/covenants/test", response_model=List[CovenantTest])
def test_covenants():
    """
    Test all covenants against latest financial data.
    This is the core monitoring function.
    """
    latest_metrics = MockERPConnector.get_latest_metrics()
    results = CovenantEngine.test_all_covenants(latest_metrics)
    return results

@app.post("/legal/generate-reservation", response_model=LegalDocument)
def generate_reservation_of_rights(breach: CovenantTest):
    """
    Generate a Reservation of Rights letter for a covenant breach.
    This is the AI magic.
    """
    borrower = MockERPConnector.BORROWER_PROFILE
    
    # Find the covenant details
    covenant = next(
        (c for c in CovenantEngine.STANDARD_COVENANTS if c.covenant_id == breach.covenant_id),
        None
    )
    
    if not covenant:
        raise HTTPException(status_code=404, detail="Covenant not found")
    
    # Generate the letter using GPT-4
    letter_text = LMALegalEngine.generate_reservation_of_rights(
        borrower_name=borrower["name"],
        facility_amount=borrower["facility_amount"],
        breach=breach,
        covenant_clause=covenant.clause_reference
    )
    
    return LegalDocument(
        document_type="reservation_of_rights",
        breach_details=f"{covenant.clause_reference} - {breach.status}",
        generated_text=letter_text,
        clause_references=[covenant.clause_reference, "Clause 23.1 - Events of Default"],
        created_at=datetime.now()
    )

@app.post("/legal/generate-waiver-template", response_model=LegalDocument)
def generate_waiver_template(breach: CovenantTest):
    """Generate a waiver request template for the borrower"""
    borrower = MockERPConnector.BORROWER_PROFILE
    
    covenant = next(
        (c for c in CovenantEngine.STANDARD_COVENANTS if c.covenant_id == breach.covenant_id),
        None
    )
    
    if not covenant:
        raise HTTPException(status_code=404, detail="Covenant not found")
    
    waiver_text = LMALegalEngine.generate_waiver_request_template(
        borrower_name=borrower["name"],
        breach=breach,
        covenant_clause=covenant.clause_reference
    )
    
    return LegalDocument(
        document_type="waiver_request_template",
        breach_details=f"{covenant.clause_reference} - Template for Borrower",
        generated_text=waiver_text,
        clause_references=[covenant.clause_reference],
        created_at=datetime.now()
    )

@app.get("/alerts/active")
def get_active_alerts():
    """Get all active covenant breach alerts"""
    latest_metrics = MockERPConnector.get_latest_metrics()
    results = CovenantEngine.test_all_covenants(latest_metrics)
    
    breaches = [r for r in results if r.status == CovenantStatus.BREACH]
    warnings = [r for r in results if r.status == CovenantStatus.WARNING]
    
    return {
        "breaches": breaches,
        "warnings": warnings,
        "total_alerts": len(breaches) + len(warnings),
        "critical": len(breaches) > 0
    }

@app.get("/financial/forecast")
def get_financial_forecast(months: int = 3):
    """Get financial forecast using PerforatedAI augmented model"""
    latest = MockERPConnector.get_latest_metrics()
    return forecaster.predict_next_months(latest, months)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
