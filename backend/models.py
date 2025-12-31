from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from enum import Enum

class CovenantType(str, Enum):
    LEVERAGE_RATIO = "leverage_ratio"
    INTEREST_COVER = "interest_cover"
    DEBT_SERVICE_COVER = "debt_service_cover"
    CAPEX_LIMIT = "capex_limit"

class CovenantStatus(str, Enum):
    COMPLIANT = "compliant"
    WARNING = "warning"
    BREACH = "breach"

class FinancialMetric(BaseModel):
    date: datetime
    ebitda: float
    total_debt: float
    cash_flow: float
    interest_expense: float
    capex: float
    revenue: float

class Covenant(BaseModel):
    covenant_id: str
    covenant_type: CovenantType
    threshold: float
    clause_reference: str
    description: str
    test_frequency: str  # quarterly, annual

class CovenantTest(BaseModel):
    test_id: str
    covenant_id: str
    test_date: datetime
    actual_value: float
    threshold_value: float
    status: CovenantStatus
    breach_margin: Optional[float] = None

class LegalDocument(BaseModel):
    document_type: str  # "reservation_of_rights", "waiver_request"
    breach_details: str
    generated_text: str
    clause_references: List[str]
    created_at: datetime

class BorrowerProfile(BaseModel):
    borrower_id: str
    name: str
    facility_amount: float
    facility_currency: str
    agent_bank: str
    origination_date: datetime

class SimulationParams(BaseModel):
    revenue_change_pct: float = 0.0
    interest_rate_change_bps: float = 0.0

class SimulationResult(BaseModel):
    base_metrics: FinancialMetric
    stressed_metrics: FinancialMetric
    covenant_results: List[CovenantTest]
