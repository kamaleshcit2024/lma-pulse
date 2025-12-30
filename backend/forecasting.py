import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
from perforatedai import utils_perforatedai as UPA
from perforatedai import globals_perforatedai as GPA
from models import FinancialMetric
import numpy as np
from datetime import datetime, timedelta

class FinancialForecaster(nn.Module):
    """
    A simple forecasting model for financial metrics.
    It takes a sequence of past EBITDA values and predicts the next one.
    """
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(FinancialForecaster, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

class ForecastingEngine:
    """
    Wraps the PerforatedAI Augmented Model to provide forecasts.
    """
    def __init__(self):
        self.device = torch.device("cpu") # Keep it simple for demo
        self.model = FinancialForecaster().to(self.device)
        
        # Initialize with PerforatedAI
        # maximizing_score=False because we want to minimize Loss (MSE)
        self.model = UPA.initialize_pai(self.model, maximizing_score=False)
        self.model = self.model.to(self.device)
        
        # Setup optimizer via PerforatedAI tracker
        GPA.pai_tracker.set_optimizer(optim.Adam)
        # We won't strictly use a scheduler for this simple mock training, but PAI needs one set
        GPA.pai_tracker.set_scheduler(optim.lr_scheduler.StepLR)
        
        # Hack to prevent PAI from exiting on dimension mismatch during mock training
        GPA.pc.set_debugging_output_dimensions(1)
        
        # Use unique save directory to avoid Windows file lock errors
        import time
        GPA.pc.set_save_name(f"forecast_run_{int(time.time())}")

        self.optimArgs = {'params': self.model.parameters(), 'lr': 0.01}
        self.schedArgs = {'step_size': 100, 'gamma': 0.1}
        
        self.optimizer, self.scheduler = GPA.pai_tracker.setup_optimizer(
            self.model, self.optimArgs, self.schedArgs
        )
        
        self.is_trained = False

    def train_mock_model(self, metrics: List[FinancialMetric]):
        """
        Simulates a training loop using PerforatedAI logic.
        Since we have very little data, we'll just overfit to the historical data 
        to demonstrate the integration.
        """
        # Prepare data (Extract EBITDA)
        ebitda_values = [m.ebitda / 1_000_000 for m in metrics] # Scale down for stability
        data = torch.tensor(ebitda_values, dtype=torch.float32).view(-1, 1)
        
        if len(data) < 2:
            return # Not enough data
        
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(10): # Short training for demo
            epoch_loss = 0
            for i in range(len(data) - 1):
                x = data[i]
                y_true = data[i+1]
                
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = criterion(y_pred, y_true)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            # Validation step (simulated) 
            # In a real scenario, we'd use a separate validation set.
            # Here we just feed the loss to PAI to trigger potential restructuring.
            avg_loss = epoch_loss / (len(data) - 1)
            
            # CRITICAL: PerforatedAI hook
            # "score" here is the loss we want to minimize
            self.model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
                avg_loss, self.model
            )
            
            if restructured:
                print(f"[PerforatedAI] Model restructured at epoch {epoch}")
                self.optimizer, self.scheduler = GPA.pai_tracker.setup_optimizer(
                    self.model, self.optimArgs, self.schedArgs
                )
                self.model = self.model.to(self.device)

        self.is_trained = True

    def predict_next_months(self, current_metrics: FinancialMetric, months=3) -> List[Dict]:
        """
        Predict future metrics.
        """
        if not self.is_trained:
            # Fallback if not trained
            return []

        predictions = []
        
        # Start from current
        current_ebitda = current_metrics.ebitda / 1_000_000
        last_input = torch.tensor([current_ebitda], dtype=torch.float32).view(1)
        
        self.model.eval()
        with torch.no_grad():
            for i in range(months):
                next_ebitda_scaled = self.model(last_input).item()
                
                # Construct a predicted metric dict
                # Simplification: assume other metrics scale roughly with EBITDA or stay constant for the demo
                next_ebitda_real = next_ebitda_scaled * 1_000_000
                
                predictions.append({
                    "month_offset": i + 1,
                    "predicted_ebitda": round(next_ebitda_real, 2),
                    "note": "Predicted by PAI-Augmented Model"
                })
                
                # Update input for next step
                last_input = torch.tensor([next_ebitda_scaled], dtype=torch.float32).view(1)

        return predictions

# Singleton instance
forecaster = ForecastingEngine()
