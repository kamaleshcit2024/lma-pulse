from typing import Dict, List
from models import CovenantTest, CovenantStatus, FinancialMetric, Covenant
from datetime import datetime
import uuid

class CovenantEngine:
    """
    The covenant calculation engine - this is the core logic that 
    evaluates whether a borrower is in compliance with LMA covenants.
    """
    
    # Standard LMA Clause 21 Financial Covenants for our demo
    STANDARD_COVENANTS = [
        Covenant(
            covenant_id="cov_leverage",
            covenant_type="leverage_ratio",
            threshold=4.0,  # Max 4.0x
            clause_reference="Clause 21.1 - Leverage Ratio",
            description="Total Net Debt / EBITDA must not exceed 4.00:1",
            test_frequency="quarterly"
        ),
        Covenant(
            covenant_id="cov_interest_cover",
            covenant_type="interest_cover",
            threshold=3.0,  # Min 3.0x
            clause_reference="Clause 21.2 - Interest Cover",
            description="EBITDA / Interest Expense must be at least 3.00:1",
            test_frequency="quarterly"
        ),
        Covenant(
            covenant_id="cov_debt_service",
            covenant_type="debt_service_cover",
            threshold=1.25,  # Min 1.25x
            clause_reference="Clause 21.3 - Debt Service Cover",
            description="Cash Flow / Debt Service must be at least 1.25:1",
            test_frequency="quarterly"
        )
    ]
    
    @staticmethod
    def calculate_leverage_ratio(metrics: FinancialMetric) -> float:
        """Total Debt / EBITDA"""
        if metrics.ebitda <= 0:
            return float('inf')
        return metrics.total_debt / metrics.ebitda
    
    @staticmethod
    def calculate_interest_cover(metrics: FinancialMetric) -> float:
        """EBITDA / Interest Expense"""
        if metrics.interest_expense <= 0:
            return float('inf')
        return metrics.ebitda / metrics.interest_expense
    
    @staticmethod
    def calculate_debt_service_cover(metrics: FinancialMetric) -> float:
        """Cash Flow / (Interest + Principal) - simplified"""
        debt_service = metrics.interest_expense * 1.5  # Simplified assumption
        if debt_service <= 0:
            return float('inf')
        return metrics.cash_flow / debt_service
    
    @classmethod
    def test_covenant(cls, covenant: Covenant, metrics: FinancialMetric) -> CovenantTest:
        """
        Test a single covenant against current financial metrics.
        Returns a CovenantTest with status (compliant, warning, breach).
        """
        
        # Calculate actual value based on covenant type
        if covenant.covenant_type == "leverage_ratio":
            actual = cls.calculate_leverage_ratio(metrics)
            is_breach = actual > covenant.threshold
            breach_margin = actual - covenant.threshold if is_breach else None
            # Warning if within 10% of threshold
            is_warning = (actual > covenant.threshold * 0.9) and not is_breach
            
        elif covenant.covenant_type == "interest_cover":
            actual = cls.calculate_interest_cover(metrics)
            is_breach = actual < covenant.threshold
            breach_margin = covenant.threshold - actual if is_breach else None
            is_warning = (actual < covenant.threshold * 1.1) and not is_breach
            
        elif covenant.covenant_type == "debt_service_cover":
            actual = cls.calculate_debt_service_cover(metrics)
            is_breach = actual < covenant.threshold
            breach_margin = covenant.threshold - actual if is_breach else None
            is_warning = (actual < covenant.threshold * 1.1) and not is_breach
        
        else:
            raise ValueError(f"Unknown covenant type: {covenant.covenant_type}")
        
        # Determine status
        if is_breach:
            status = CovenantStatus.BREACH
        elif is_warning:
            status = CovenantStatus.WARNING
        else:
            status = CovenantStatus.COMPLIANT
        
        return CovenantTest(
            test_id=str(uuid.uuid4()),
            covenant_id=covenant.covenant_id,
            test_date=metrics.date,
            actual_value=round(actual, 2),
            threshold_value=covenant.threshold,
            status=status,
            breach_margin=round(breach_margin, 2) if breach_margin else None
        )
    
    @classmethod
    def test_all_covenants(cls, metrics: FinancialMetric) -> List[CovenantTest]:
        """Test all standard covenants"""
        return [cls.test_covenant(cov, metrics) for cov in cls.STANDARD_COVENANTS]
