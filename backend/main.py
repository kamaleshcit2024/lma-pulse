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
from neural_engine import CovenantMonitor, train_model
import os

app = FastAPI(title="LMA Pulse API", version="1.0.0")

# Global Monitor Instance
covenant_monitor = None

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
    global covenant_monitor
    init_db()
    
    # 1. Train/Load Dendritic Covenant Model
    if not os.path.exists("covenant_breach_model.pth"):
        print("Training Dendritic Covenant Model...")
        try:
            train_model()
            print("Dendritic Model Trained Successfully.")
        except Exception as e:
            print(f"Failed to train Dendritic Model: {e}")
            
    covenant_monitor = CovenantMonitor("covenant_breach_model.pth")
    
    # 2. Train Forecasting Model
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

@app.get("/borrowers/search")
def search_borrowers(q: str = ""):
    """Search for borrowers by name or ID"""
    return MockERPConnector.search_borrowers(q)

@app.get("/borrower/profile", response_model=BorrowerProfile)
def get_borrower_profile(borrower_id: str = "BOR_SOLARIS_001"):
    """Get the borrower profile"""
    profile = MockERPConnector.get_profile(borrower_id)
    return BorrowerProfile(**profile)

@app.get("/financial/latest", response_model=FinancialMetric)
def get_latest_financials(borrower_id: str = "BOR_SOLARIS_001"):
    """Get the latest financial metrics from the ERP system"""
    return MockERPConnector.get_latest_metrics(borrower_id)

@app.get("/financial/history", response_model=List[FinancialMetric])
def get_financial_history(months: int = 12, borrower_id: str = "BOR_SOLARIS_001"):
    """Get historical financial data"""
    return MockERPConnector.generate_historical_data(months, borrower_id)

@app.get("/covenants/list", response_model=List[Covenant])
def get_covenants():
    """Get all covenants defined in the facility agreement"""
    return CovenantEngine.STANDARD_COVENANTS

@app.post("/covenants/test", response_model=List[CovenantTest])
def test_covenants(borrower_id: str = "BOR_SOLARIS_001"):
    """
    Test all covenants against latest financial data.
    This is the core monitoring function.
    """
    latest_metrics = MockERPConnector.get_latest_metrics(borrower_id)
    results = CovenantEngine.test_all_covenants(latest_metrics)
    return results

@app.post("/legal/generate-reservation", response_model=LegalDocument)
def generate_reservation_of_rights(breach: CovenantTest, borrower_id: str = "BOR_SOLARIS_001"):
    """
    Generate a Reservation of Rights letter for a covenant breach.
    This is the AI magic.
    """
    borrower = MockERPConnector.get_profile(borrower_id)
    
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
def generate_waiver_template(breach: CovenantTest, borrower_id: str = "BOR_SOLARIS_001"):
    """Generate a waiver request template for the borrower"""
    borrower = MockERPConnector.get_profile(borrower_id)
    
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
def get_active_alerts(borrower_id: str = "BOR_SOLARIS_001"):
    """Get all active covenant breach alerts"""
    latest_metrics = MockERPConnector.get_latest_metrics(borrower_id)
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

@app.get("/covenants/predict-risk")
def predict_breach_risk(borrower_id: str = "BOR_SOLARIS_001"):
    """
    Predict covenant breach risk using the Dendritic Neural Network.
    Returns probabilities for 1/7/30/90 days.
    """
    if not covenant_monitor:
        raise HTTPException(status_code=503, detail="AI Model not initialized")
        
    latest = MockERPConnector.get_latest_metrics(borrower_id)
    profile = MockERPConnector.get_profile(borrower_id)
    
    # Convert financial metric to model input format
    # The model expects raw floats, MockERP returns FinancialMetric object
    # We map fields. Note: Model input vector is flexible in our wrapper.
    
    risk_assessment = covenant_monitor.predict_breach_risk(
        borrower_name=profile["name"],
        ebitda=latest.ebitda / 1_000_000, # Normalize to Millions for model stability
        revenue=latest.revenue / 1_000_000,
        total_debt=latest.total_debt / 1_000_000,
        cash_flow=latest.cash_flow / 1_000_000,
        current_assets=(latest.total_debt * 0.4) / 1_000_000, # Mock derived
        current_liabilities=(latest.total_debt * 0.3) / 1_000_000, # Mock derived
        interest_expense=latest.interest_expense / 1_000_000,
        capex=latest.capex / 1_000_000,
        industry="Energy", # In production, this would come from profile
        loan_type="Term"
    )
    
    return risk_assessment

@app.post("/financial/simulate", response_model=SimulationResult)
def simulate_financials(params: SimulationParams):
    """Simulate financial stress scenarios"""
    # 1. Get Base Data
    base = MockERPConnector.get_latest_metrics()
    
    # Ensure base is a dict if it isn't already (MockERP usually returns dicts)
    if not isinstance(base, dict):
        base = base.dict()
        
    # 2. Calculate Stressed Metrics
    # Revenue Change
    rev_change_factor = 1 + (params.revenue_change_pct / 100.0)
    new_revenue = base['revenue'] * rev_change_factor
    
    # Maintain EBITDA Margin
    margin = base['ebitda'] / base['revenue'] if base['revenue'] else 0
    new_ebitda = new_revenue * margin
    
    # Interest Rate Shock
    # Calculate implied current rate
    current_debt = base['total_debt']
    current_interest_annual = base['interest_expense'] * 12 # Interest is usually monthly in these metrics? 
    # Let's check MockERP. usually metrics are LTM or Annualized?
    # Covenant tests usually use LTM. Let's assume the metric provided is the relevant period value.
    # If interest_expense is satisfying Interest Cover (EBITDA/Int ~ 4x), then they are same period.
    # Simple logic: New Interest = Old Interest + (Debt * DeltaRate)
    # Rate is Annual. Metric period is unknown. 
    # Modification: New Interest = Old Interest + (Debt * (bps/10000) * (PeriodAdjustment))
    # Let's assume metrics are Annualized for simplicity in simulation.
    
    interest_increase = current_debt * (params.interest_rate_change_bps / 10000.0)
    new_interest = base['interest_expense'] + interest_increase
    
    # Create Stressed Object
    stressed_data = base.copy()
    stressed_data.update({
        "revenue": new_revenue,
        "ebitda": new_ebitda,
        "interest_expense": new_interest,
        # Simplify cash flow impact
        "cash_flow": base['cash_flow'] * rev_change_factor - (interest_increase) 
    })
    
    stressed_metric = FinancialMetric(**stressed_data)
    base_metric = FinancialMetric(**base)
    
    # 3. Test Covenants on Stressed Data
    results = CovenantEngine.test_all_covenants(stressed_metric)
    
    return SimulationResult(
        base_metrics=base_metric,
        stressed_metrics=stressed_metric,
        covenant_results=results
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
