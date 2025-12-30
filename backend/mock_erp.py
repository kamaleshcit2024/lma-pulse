from models import FinancialMetric
from datetime import datetime, timedelta
from typing import List
import random

class MockERPConnector:
    """
    Simulates a real-time connection to a borrower's ERP system (SAP, Xero, Oracle).
    In production, this would be replaced with actual API connectors.
    """
    
    BORROWER_PROFILE = {
        "borrower_id": "BOR_SOLARIS_001",
        "name": "Solaris Energy Ltd",
        "facility_amount": 500_000_000,  # $500M facility
        "facility_currency": "USD",
        "agent_bank": "Barclays Bank PLC",
        "origination_date": datetime(2022, 1, 15),
        "maturity_date": datetime(2027, 1, 15)
    }
    
    @classmethod
    def generate_historical_data(cls, months: int = 12) -> List[FinancialMetric]:
        """
        Generate historical financial data for Solaris Energy.
        We'll simulate a company that starts healthy but deteriorates.
        """
        metrics = []
        base_date = datetime.now() - timedelta(days=30 * months)
        
        # Start with healthy metrics
        base_ebitda = 120_000_000  # $120M annual EBITDA
        base_debt = 400_000_000     # $400M debt (3.33x leverage - healthy)
        base_revenue = 600_000_000  # $600M revenue
        
        for i in range(months):
            # Simulate deterioration over time (commodity price crash scenario)
            deterioration_factor = 1 - (i * 0.05)  # 5% decline per month
            
            # Add some randomness
            random_factor = random.uniform(0.95, 1.05)
            
            ebitda = max(base_ebitda * deterioration_factor * random_factor, 50_000_000)
            debt = base_debt + (i * 5_000_000)  # Debt creeping up
            revenue = base_revenue * deterioration_factor * random_factor
            
            metric = FinancialMetric(
                date=base_date + timedelta(days=30 * i),
                ebitda=round(ebitda, 2),
                total_debt=round(debt, 2),
                cash_flow=round(ebitda * 0.8, 2),  # 80% conversion
                interest_expense=round(debt * 0.05 / 12, 2),  # 5% annual rate
                capex=round(revenue * 0.15, 2),  # 15% of revenue
                revenue=round(revenue, 2)
            )
            
            metrics.append(metric)
        
        return metrics
    
    @classmethod
    def get_latest_metrics(cls) -> FinancialMetric:
        """Get the most recent financial data (simulates real-time feed)"""
        # Current month - designed to trigger a breach
        return FinancialMetric(
            date=datetime.now(),
            ebitda=85_000_000,      # EBITDA collapsed
            total_debt=450_000_000,  # Debt increased
            cash_flow=68_000_000,    # Cash flow under pressure
            interest_expense=1_875_000,  # Monthly interest
            capex=45_000_000,
            revenue=420_000_000
        )
        # This gives us: 450M / 85M = 5.29x leverage ratio
        # BREACH! (threshold is 4.0x)
