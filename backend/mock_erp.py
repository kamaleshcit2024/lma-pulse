from models import FinancialMetric
from datetime import datetime, timedelta
from typing import List
import random

class MockERPConnector:
    """
    Simulates a real-time connection to a borrower's ERP system (SAP, Xero, Oracle).
    In production, this would be replaced with actual API connectors.
    """
    
    PROFILES = {
        # ENERGY SECTOR
        "BOR_SOLARIS_001": {
            "borrower_id": "BOR_SOLARIS_001", "name": "Solaris Energy Ltd",
            "facility_amount": 500_000_000, "facility_currency": "USD", "agent_bank": "Barclays Bank PLC",
            "origination_date": datetime(2022, 1, 15), "maturity_date": datetime(2027, 1, 15), "profile_type": "deteriorating"
        },
        "BOR_NEBULA_010": {
            "borrower_id": "BOR_NEBULA_010", "name": "Nebula Power Gen",
            "facility_amount": 850_000_000, "facility_currency": "EUR", "agent_bank": "BNP Paribas",
            "origination_date": datetime(2020, 8, 1), "maturity_date": datetime(2030, 8, 1), "profile_type": "healthy"
        },
        
        # INDUSTRIAL / MANUFACTURING
        "BOR_APEX_002": {
            "borrower_id": "BOR_APEX_002", "name": "Apex Industrials Inc",
            "facility_amount": 750_000_000, "facility_currency": "EUR", "agent_bank": "Deutsche Bank",
            "origination_date": datetime(2021, 6, 10), "maturity_date": datetime(2026, 6, 10), "profile_type": "healthy"
        },
        "BOR_TITAN_011": {
            "borrower_id": "BOR_TITAN_011", "name": "Titan Steelworks",
            "facility_amount": 1_200_000_000, "facility_currency": "USD", "agent_bank": "JPMorgan Chase",
            "origination_date": datetime(2019, 4, 12), "maturity_date": datetime(2029, 4, 12), "profile_type": "volatile"
        },
        "BOR_NORDIC_012": {
            "borrower_id": "BOR_NORDIC_012", "name": "Nordic Paper Mills",
            "facility_amount": 300_000_000, "facility_currency": "SEK", "agent_bank": "Nordea",
            "origination_date": datetime(2023, 1, 5), "maturity_date": datetime(2026, 1, 5), "profile_type": "stable"
        },

        # RETAIL & CONSUMER
        "BOR_ZENITH_003": {
            "borrower_id": "BOR_ZENITH_003", "name": "Zenith Retail Group",
            "facility_amount": 200_000_000, "facility_currency": "GBP", "agent_bank": "HSBC",
            "origination_date": datetime(2023, 3, 20), "maturity_date": datetime(2028, 3, 20), "profile_type": "volatile"
        },
        "BOR_OMNI_004": {
            "borrower_id": "BOR_OMNI_004", "name": "Omni Global Markets",
            "facility_amount": 1_500_000_000, "facility_currency": "USD", "agent_bank": "Citi",
            "origination_date": datetime(2021, 11, 30), "maturity_date": datetime(2026, 11, 30), "profile_type": "healthy"
        },
        "BOR_LUXE_013": {
            "borrower_id": "BOR_LUXE_013", "name": "Luxe Fashion House",
            "facility_amount": 450_000_000, "facility_currency": "EUR", "agent_bank": "Credit Agricole",
            "origination_date": datetime(2022, 5, 20), "maturity_date": datetime(2027, 5, 20), "profile_type": "high_growth"
        },

        # TECH & TELECOM
        "BOR_VERTEX_005": {
            "borrower_id": "BOR_VERTEX_005", "name": "Vertex Systems",
            "facility_amount": 300_000_000, "facility_currency": "USD", "agent_bank": "Silicon Valley Bridge Bank",
            "origination_date": datetime(2022, 2, 14), "maturity_date": datetime(2027, 2, 14), "profile_type": "high_growth"
        },
        "BOR_QUANTUM_006": {
            "borrower_id": "BOR_QUANTUM_006", "name": "Quantum Networks",
            "facility_amount": 900_000_000, "facility_currency": "USD", "agent_bank": "Goldman Sachs",
            "origination_date": datetime(2020, 9, 15), "maturity_date": datetime(2025, 9, 15), "profile_type": "deteriorating"
        },
        "BOR_CYBER_014": {
            "borrower_id": "BOR_CYBER_014", "name": "CyberDyne Systems",
            "facility_amount": 2_000_000_000, "facility_currency": "USD", "agent_bank": "Bank of America",
            "origination_date": datetime(2024, 1, 1), "maturity_date": datetime(2029, 12, 31), "profile_type": "high_growth"
        },

        # LOGISTICS & TRANSPORT
        "BOR_PACIFIC_007": {
            "borrower_id": "BOR_PACIFIC_007", "name": "Pacific Logistics",
            "facility_amount": 400_000_000, "facility_currency": "SGD", "agent_bank": "DBS",
            "origination_date": datetime(2021, 7, 7), "maturity_date": datetime(2026, 7, 7), "profile_type": "steady"
        },
        "BOR_AERO_008": {
            "borrower_id": "BOR_AERO_008", "name": "AeroDynamics Inc",
            "facility_amount": 1_100_000_000, "facility_currency": "USD", "agent_bank": "Morgan Stanley",
            "origination_date": datetime(2019, 12, 1), "maturity_date": datetime(2029, 12, 1), "profile_type": "distressed"
        },

        # HEALTHCARE
        "BOR_VITA_009": {
            "borrower_id": "BOR_VITA_009", "name": "VitaLife Pharma",
            "facility_amount": 600_000_000, "facility_currency": "CHF", "agent_bank": "UBS",
            "origination_date": datetime(2023, 6, 15), "maturity_date": datetime(2028, 6, 15), "profile_type": "healthy"
        },
        "BOR_MEDIX_015": {
            "borrower_id": "BOR_MEDIX_015", "name": "Medix Healthcare Grp",
            "facility_amount": 350_000_000, "facility_currency": "GBP", "agent_bank": "NatWest",
            "origination_date": datetime(2021, 3, 10), "maturity_date": datetime(2026, 3, 10), "profile_type": "stable"
        }
    }
    
    @classmethod
    def get_profile(cls, borrower_id: str):
        return cls.PROFILES.get(borrower_id, cls.PROFILES["BOR_SOLARIS_001"])

    @classmethod
    def search_borrowers(cls, query: str):
        query = query.lower()
        results = []
        for pid, profile in cls.PROFILES.items():
            if query in profile["name"].lower() or query in pid.lower():
                results.append(profile)
        return results
    
    @classmethod
    def generate_historical_data(cls, months: int = 12, borrower_id: str = "BOR_SOLARIS_001") -> List[FinancialMetric]:
        metrics = []
        base_date = datetime.now() - timedelta(days=30 * months)
        profile = cls.get_profile(borrower_id)
        ptype = profile.get("profile_type", "deteriorating")
        
        # Base stats based on company size (facility amount as proxy)
        size_factor = profile["facility_amount"] / 500_000_000
        
        base_ebitda = 120_000_000 * size_factor
        base_debt = 400_000_000 * size_factor
        base_revenue = 600_000_000 * size_factor
        
        for i in range(months):
            # Calculate deterioration/growth factors based on profile type
            if ptype == "deteriorating":
                trend_factor = 1 - (i * 0.05)
            elif ptype == "distressed":
                trend_factor = 1 - (i * 0.08) # Rapid decline
            elif ptype in ["healthy", "stable", "steady"]:
                trend_factor = 1 + (i * 0.01) # Slow steady growth
            elif ptype == "high_growth":
                trend_factor = 1 + (i * 0.04) # Fast growth
            else: # volatile
                trend_factor = 1 + (random.uniform(-0.15, 0.15))

            random_factor = random.uniform(0.95, 1.05)
            
            ebitda = max(base_ebitda * trend_factor * random_factor, base_ebitda * 0.1)
            
            # Debt logic
            if ptype in ["deteriorating", "distressed"]:
                debt = base_debt + (i * 5_000_000 * size_factor)
            elif ptype == "high_growth":
                 debt = base_debt * 1.2 # Taking on more debt for growth
            elif ptype == "healthy":
                debt = base_debt * 0.95 # Paying down debt
            else:
                debt = base_debt * random.uniform(0.95, 1.05)

                
            revenue = base_revenue * trend_factor * random_factor
            
            metrics.append(FinancialMetric(
                date=base_date + timedelta(days=30 * i),
                ebitda=round(ebitda, 2),
                total_debt=round(debt, 2),
                cash_flow=round(ebitda * 0.8, 2),
                interest_expense=round(debt * 0.05 / 12, 2),
                capex=round(revenue * 0.15, 2),
                revenue=round(revenue, 2)
            ))
        
        return metrics
    
    @classmethod
    def get_latest_metrics(cls, borrower_id: str = "BOR_SOLARIS_001") -> FinancialMetric:
        # Generate history and take the last/simulated next step
        # For simplicity, we can reuse logic or defined specific "latest" states
        
        profile = cls.get_profile(borrower_id)
        ptype = profile.get("profile_type", "deteriorating")
        
        # Generate 1 month ahead based on trend logic
        history = cls.generate_historical_data(months=1, borrower_id=borrower_id) 
        # But generate_historical generates past up to now-ish. 
        # Actually generate_historical logic starts from (now - months). So index 'i' goes up to months-1.
        # Let's just generate a fresh "current" metric based on profile type.
        
        size_factor = profile["facility_amount"] / 500_000_000
        
        if ptype in ["deteriorating", "distressed"]:
            # Breached state
            return FinancialMetric(
                date=datetime.now(),
                ebitda=85_000_000 * size_factor * (0.8 if ptype=='distressed' else 1.0),
                total_debt=450_000_000 * size_factor,
                cash_flow=68_000_000 * size_factor,
                interest_expense=1_875_000 * size_factor,
                capex=45_000_000 * size_factor,
                revenue=420_000_000 * size_factor
            )
        elif ptype in ["healthy", "stable", "steady"]:
            # Strong state
            return FinancialMetric(
                date=datetime.now(),
                ebitda=140_000_000 * size_factor,
                total_debt=380_000_000 * size_factor,
                cash_flow=110_000_000 * size_factor,
                interest_expense=1_500_000 * size_factor,
                capex=60_000_000 * size_factor,
                revenue=700_000_000 * size_factor
            )
        elif ptype == "high_growth":
             # High EBITDA but High Debt
             return FinancialMetric(
                date=datetime.now(),
                ebitda=180_000_000 * size_factor,
                total_debt=600_000_000 * size_factor,
                cash_flow=90_000_000 * size_factor, # Lower CF due to heavy reinvestment
                interest_expense=2_500_000 * size_factor,
                capex=120_000_000 * size_factor, 
                revenue=900_000_000 * size_factor
            )
        else: # Volatile
             return FinancialMetric(
                date=datetime.now(),
                ebitda=100_000_000 * size_factor, # On the edge
                total_debt=410_000_000 * size_factor,
                cash_flow=80_000_000 * size_factor,
                interest_expense=1_700_000 * size_factor,
                capex=50_000_000 * size_factor,
                revenue=550_000_000 * size_factor
            )
