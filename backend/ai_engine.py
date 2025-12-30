import openai
from typing import List
from models import CovenantTest, LegalDocument
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class LMALegalEngine:
    """
    The AI engine that generates LMA-compliant legal documents.
    This is the "killer feature" - automated legal drafting.
    """
    
    SENIOR_BANKER_SYSTEM_PROMPT = """You are a Senior Director in the Loan Syndications team at a Tier-1 investment bank with 20+ years of experience. You are an expert in the Loan Market Association (LMA) standard documentation, particularly the Senior Multicurrency Term Facility Agreement.

Your role is to draft formal, legally precise correspondence in response to covenant breaches. You must:

1. Use formal banking language and LMA-standard terminology
2. Always include "without prejudice to our rights" language to avoid involuntary waiver (Lombard North Central plc v European Skyjets Ltd)
3. Reference specific clause numbers from the Facility Agreement
4. Maintain a professional but firm tone
5. Use precise financial terminology (EBITDA, Leverage Ratio, etc.)
6. Structure documents according to LMA precedents

Never use casual language. Every sentence must be legally defensible."""

    @classmethod
    def generate_reservation_of_rights(cls, 
                                       borrower_name: str,
                                       facility_amount: float,
                                       breach: CovenantTest,
                                       covenant_clause: str) -> str:
        """
        Generate a formal "Reservation of Rights" letter.
        This is sent when a breach is detected to preserve the bank's rights.
        """
        
        prompt = f"""Draft a formal "Reservation of Rights" letter to {borrower_name} regarding a covenant breach.

FACILITY DETAILS:
- Borrower: {borrower_name}
- Facility Amount: ${facility_amount:,.0f}
- Breach Date: {breach.test_date.strftime('%d %B %Y')}

BREACH DETAILS:
- Covenant: {covenant_clause}
- Threshold: {breach.threshold_value}x
- Actual Value: {breach.actual_value}x
- Breach Margin: {breach.breach_margin}x

REQUIRED ELEMENTS:
1. Reference to the specific clause breached
2. Statement that this constitutes an Event of Default under Clause 23.1
3. Clear reservation of all rights under the Facility Agreement
4. Statement that this letter does NOT constitute a waiver
5. Request for immediate rectification plan
6. Professional but firm tone

The letter should be approximately 250-300 words."""

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": cls.SENIOR_BANKER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for formal, consistent output
            max_tokens=800
        )
        
        return response.choices[0].message.content
    
    @classmethod
    def generate_waiver_request_template(cls,
                                        borrower_name: str,
                                        breach: CovenantTest,
                                        covenant_clause: str) -> str:
        """
        Generate a template for a Waiver Request that the borrower can use.
        This is more commercial - helps maintain the relationship.
        """
        
        prompt = f"""Draft a template "Waiver Request" letter that {borrower_name} can send to the Agent Bank.

BREACH DETAILS:
- Covenant Breached: {covenant_clause}
- Threshold: {breach.threshold_value}x
- Actual Value: {breach.actual_value}x
- Test Date: {breach.test_date.strftime('%d %B %Y')}

REQUIRED ELEMENTS:
1. Acknowledgment of the breach
2. Explanation of circumstances (leave blank for borrower to complete)
3. Request for temporary waiver
4. Proposed waiver fee (market standard: 10-25bps)
5. Remediation timeline
6. Professional, cooperative tone

The letter should guide the borrower on how to request a waiver properly."""

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": cls.SENIOR_BANKER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=700
        )
        
        return response.choices[0].message.content
    
    @classmethod
    def generate_breach_summary(cls, breaches: List[CovenantTest]) -> str:
        """Generate an executive summary of all breaches for internal use"""
        
        breach_details = "\n".join([
            f"- {b.covenant_id}: {b.actual_value}x (threshold: {b.threshold_value}x)"
            for b in breaches
        ])
        
        prompt = f"""Generate a concise internal memo (150 words) summarizing the following covenant breaches for the Credit Risk Committee:

BREACHES DETECTED:
{breach_details}

Include:
1. Risk assessment (Low/Medium/High)
2. Recommended immediate actions
3. Whether this triggers mandatory reporting to regulators"""

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": cls.SENIOR_BANKER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400
        )
        
        return response.choices[0].message.content
