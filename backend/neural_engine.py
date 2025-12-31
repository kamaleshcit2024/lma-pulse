import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
import json
import os
from datetime import datetime, timedelta

# Handle imports for PerforatedAI whether installed or local
try:
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
except ImportError:
    try:
        from PerforatedAI.perforatedai import globals_perforatedai as GPA
        from PerforatedAI.perforatedai import utils_perforatedai as UPA
    except ImportError:
        print("WARNING: PerforatedAI not found. Dendritic features will be disabled or mocked.")
        # Minimal mocks to allow import without crashing
        class MockGPA:
            class MockPC:
                def append_module_ids_to_track(self, *args): pass
                def set_weight_decay_accepted(self, *args): pass
                def set_testing_dendrite_capacity(self, *args): pass
                def set_initial_correlation_batches(self, *args): pass
            pc = MockPC()
            class MockTracker:
                def set_optimizer_instance(self, *args): pass
                def add_extra_score(self, *args): pass
                def add_validation_score(self, score, model): return model, False, False
            pai_tracker = MockTracker()
            class MockSequential(nn.Sequential): pass
            PAISequential = MockSequential

        class MockUPA:
            def initialize_pai(self, model, **kwargs): return model
        
        GPA = MockGPA()
        UPA = MockUPA()


class DendriticSegment(nn.Module):
    def __init__(self, input_dim: int, context_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim

        self.dendritic_weights = nn.Linear(input_dim, input_dim)
        self.context_gate = nn.Linear(context_dim, input_dim)
        self.threshold = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        dendritic_output = self.dendritic_weights(x)
        gate = torch.sigmoid(self.context_gate(context))
        gated_output = dendritic_output * gate
        activated = torch.tanh(gated_output - self.threshold)
        return activated

class DendriticLayer(nn.Module):
    def __init__(self, input_dim: int, context_dim: int, num_segments: int):
        super().__init__()
        self.num_segments = num_segments
        self.segments = nn.ModuleList(
            [DendriticSegment(input_dim, context_dim) for _ in range(num_segments)]
        )
        self.combination_layer = nn.Linear(input_dim * num_segments, input_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        segment_outputs = []
        for segment in self.segments:
            segment_outputs.append(segment(x, context))

        combined_output = torch.cat(segment_outputs, dim=-1)
        output = self.combination_layer(combined_output)
        return output

# This is just a linear layer that also accepts context for the dendrites to use
class DendriticLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, context_dim: int, num_segments: int):
        super().__init__()
        self.context_dim = context_dim
        self.num_segments = num_segments
        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        output = self.output_layer(x)
        return output

class CovenantBreachPredictor(nn.Module):
    def __init__(
        self,
        financial_metrics_dim: int = 12,
        context_dim: int = 8,
        num_dendritic_segments: int = 8,
        hidden_dim: int = 128
    ):
        super().__init__()

        self.financial_encoder = GPA.PAISequential(
            [nn.Linear(financial_metrics_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()]
        )

        self.context_encoder = GPA.PAISequential(
            [nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()]
        )

        self.dendritic_layer1 = DendriticLinear(
            hidden_dim, hidden_dim, context_dim, num_dendritic_segments
        )

        self.dendritic_layer2 = DendriticLinear(
            hidden_dim, hidden_dim, context_dim, num_dendritic_segments
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

        self.covenant_classifier = nn.Linear(hidden_dim, 5)

    def forward(
        self,
        financial_metrics: torch.Tensor,
        context: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        financial_features = self.financial_encoder(financial_metrics)
        context_features = self.context_encoder(context)

        dendrite_out1 = self.dendritic_layer1(financial_features, context_features)
        dendrite_out1 = dendrite_out1 + financial_features

        dendrite_out2 = self.dendritic_layer2(dendrite_out1, context_features)
        dendrite_out2 = dendrite_out2 + dendrite_out1


        breach_probs = torch.sigmoid(self.prediction_head(dendrite_out2))
        covenant_types = torch.softmax(self.covenant_classifier(dendrite_out2), dim=-1)

        return {
            'breach_probabilities': breach_probs,
            'covenant_types': covenant_types,
            'features': dendrite_out2
        }


class CovenantDataset(Dataset):
    def __init__(self, num_samples: int = 1000, sequence_length: int = 90):
        self.num_samples = num_samples
        self.sequence_length = sequence_length

        self.data = self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> List[Dict]:
        data = []

        for i in range(self.num_samples):
            ebitda = np.random.randn() * 0.2 + 0.5
            revenue = np.random.randn() * 0.15 + 1.0
            total_debt = np.random.randn() * 0.3 + 2.0
            cash_flow = np.random.randn() * 0.25 + 0.4
            current_assets = np.random.randn() * 0.2 + 1.5
            current_liabilities = np.random.randn() * 0.2 + 1.0
            interest_expense = np.random.randn() * 0.1 + 0.2
            capex = np.random.randn() * 0.15 + 0.3
            working_capital = current_assets - current_liabilities
            net_income = np.random.randn() * 0.2 + 0.3
            depreciation = np.random.randn() * 0.1 + 0.15
            amortization = np.random.randn() * 0.05 + 0.05

            financial_metrics = np.array([
                ebitda, revenue, total_debt, cash_flow,
                current_assets, current_liabilities,
                interest_expense, capex, working_capital,
                net_income, depreciation, amortization
            ], dtype=np.float32)

            industry = np.random.randint(0, 5)
            loan_type = np.random.randint(0, 3)
            credit_rating = np.random.randn() * 0.3 + 0.5
            company_age = np.random.rand()
            loan_size = np.random.rand()
            geography = np.random.randint(0, 4)

            context = np.array([
                industry / 5.0, loan_type / 3.0, credit_rating,
                company_age, loan_size, geography / 4.0,
                np.random.rand(), np.random.rand()
            ], dtype=np.float32)

            leverage_ratio = total_debt / max(ebitda, 0.1)
            dscr = (ebitda - capex) / max(interest_expense, 0.01)
            current_ratio = current_assets / max(current_liabilities, 0.1)

            breach_1d = 1.0 if leverage_ratio > 4.0 else 0.0
            breach_7d = 1.0 if leverage_ratio > 3.8 else 0.0
            breach_30d = 1.0 if leverage_ratio > 3.5 else 0.0
            breach_90d = 1.0 if leverage_ratio > 3.2 else 0.0

            breach_labels = np.array([
                breach_1d, breach_7d, breach_30d, breach_90d
            ], dtype=np.float32)

            covenant_risk = np.zeros(5, dtype=np.float32)
            if leverage_ratio > 3.5:
                covenant_risk[0] = 1.0
            elif dscr < 1.5:
                covenant_risk[1] = 1.0
            elif current_ratio < 1.2:
                covenant_risk[2] = 1.0
            else:
                covenant_risk[4] = 1.0

            data.append({
                'financial_metrics': financial_metrics,
                'context': context,
                'breach_labels': breach_labels,
                'covenant_types': covenant_risk,
                'metadata': {
                    'leverage_ratio': leverage_ratio,
                    'dscr': dscr,
                    'current_ratio': current_ratio
                }
            })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'financial_metrics': torch.FloatTensor(sample['financial_metrics']),
            'context': torch.FloatTensor(sample['context']),
            'breach_labels': torch.FloatTensor(sample['breach_labels']),
            'covenant_types': torch.FloatTensor(sample['covenant_types'])
        }


class CovenantTrainer:
    def __init__(
        self,
        model: CovenantBreachPredictor,
        learning_rate: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)

        #tracked modules do not get dendrites
        GPA.pc.append_module_ids_to_track([".context_encoder", ".financial_encoder",
                                        ".prediction_head", ".covenant_classifier"])
        GPA.pc.set_weight_decay_accepted(True)    
        GPA.pc.set_testing_dendrite_capacity(False)
        GPA.pc.set_initial_correlation_batches(50)
        
        # Initialize PAI
        try:
             model = UPA.initialize_pai(model, maximizing_score=False)
        except Exception:
             # Fallback if PAI fails
             pass

        self.device = device
        self.learning_rate = learning_rate
        self.optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        try:
            GPA.pai_tracker.set_optimizer_instance(self.optimizer)
        except Exception: 
            pass

        self.breach_criterion = nn.BCELoss()
        self.covenant_criterion = nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            financial_metrics = batch['financial_metrics'].to(self.device)
            context = batch['context'].to(self.device)
            breach_labels = batch['breach_labels'].to(self.device)
            covenant_types = batch['covenant_types'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(financial_metrics, context)

            breach_loss = self.breach_criterion(
                outputs['breach_probabilities'],
                breach_labels
            )

            covenant_loss = self.covenant_criterion(
                outputs['covenant_types'],
                covenant_types.argmax(dim=-1)
            )

            loss = breach_loss + 0.5 * covenant_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                financial_metrics = batch['financial_metrics'].to(self.device)
                context = batch['context'].to(self.device)
                breach_labels = batch['breach_labels'].to(self.device)
                covenant_types = batch['covenant_types'].to(self.device)

                outputs = self.model(financial_metrics, context)

                breach_loss = self.breach_criterion(
                    outputs['breach_probabilities'],
                    breach_labels
                )

                covenant_loss = self.covenant_criterion(
                    outputs['covenant_types'],
                    covenant_types.argmax(dim=-1)
                )

                loss = breach_loss + 0.5 * covenant_loss
                total_loss += loss.item()

                predictions = (outputs['breach_probabilities'] > 0.5).float()
                correct_predictions += (predictions == breach_labels).sum().item()
                total_predictions += breach_labels.numel()

        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions

        self.val_losses.append(avg_loss)

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50):
        print("Starting training with Dendritic Optimization...")
        print(f"Device: {self.device}")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()

            try:
                GPA.pai_tracker.add_extra_score(train_loss, 'Train')
                model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_metrics['loss'], self.model)
                self.model = model.to(self.device)
                
                if training_complete:
                    break
                elif restructured:
                    self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
                    self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
                    GPA.pai_tracker.set_optimizer_instance(self.optimizer)
            except Exception:
                 # Fallback if PAI tracking fails
                 pass

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")

        print("Training complete!")
        return self.model


class CovenantMonitor:
    def __init__(self, model_path: str = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CovenantBreachPredictor().to(self.device)

        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
        else:
            print("No model found, using untrained model (or will train on startup).")

        self.model.eval()

    def predict_breach_risk(
        self,
        borrower_name: str,
        ebitda: float,
        revenue: float,
        total_debt: float,
        cash_flow: float,
        current_assets: float,
        current_liabilities: float,
        interest_expense: float,
        capex: float,
        industry: str = "Energy",
        loan_type: str = "Term"
    ) -> Dict:
        working_capital = current_assets - current_liabilities
        net_income = ebitda - interest_expense

        # Ensure minimal values to avoid division by zero
        ebitda = max(ebitda, 0.1)
        interest_expense = max(interest_expense, 0.01)
        current_liabilities = max(current_liabilities, 0.1)

        input_metrics = np.array([
            ebitda, revenue, total_debt, cash_flow,
            current_assets, current_liabilities,
            interest_expense, capex, working_capital,
            net_income, 0.15, 0.05
        ], dtype=np.float32)
        
        financial_metrics = torch.FloatTensor(input_metrics).unsqueeze(0).to(self.device)

        industry_map = {"Energy": 0, "Tech": 1, "Retail": 2, "Manufacturing": 3, "Healthcare": 4}
        loan_type_map = {"Term": 0, "Revolving": 1, "Bridge": 2}

        # Handle unknown keys gracefully
        ind_val = industry_map.get(industry, 0)
        loan_val = loan_type_map.get(loan_type, 0)

        context = torch.FloatTensor([
            ind_val / 5.0,
            loan_val / 3.0,
            0.5, 0.7, 0.6, 0.5, 0.5, 0.5
        ]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(financial_metrics, context)

        breach_probs = outputs['breach_probabilities'][0].cpu().numpy()
        covenant_types = outputs['covenant_types'][0].cpu().numpy()

        leverage_ratio = total_debt / ebitda
        dscr = (ebitda - capex) / interest_expense
        current_ratio = current_assets / current_liabilities

        max_prob = float(breach_probs.max())
        if max_prob > 0.8:
            risk_level = "CRITICAL"
            action = "IMMEDIATE_ESCALATION"
        elif max_prob > 0.6:
            risk_level = "HIGH"
            action = "PREPARE_WAIVER"
        elif max_prob > 0.4:
            risk_level = "MEDIUM"
            action = "MONITOR_CLOSELY"
        else:
            risk_level = "LOW"
            action = "ROUTINE_MONITORING"

        covenant_names = ["Leverage", "DSCR", "Current Ratio", "Interest Cover", "None"]
        most_at_risk = covenant_names[int(covenant_types.argmax())]

        return {
            "borrower": borrower_name,
            "risk_level": risk_level,
            "recommended_action": action,
            "breach_probabilities": {
                "1_day": float(breach_probs[0]),
                "7_day": float(breach_probs[1]),
                "30_day": float(breach_probs[2]),
                "90_day": float(breach_probs[3])
            },
            "covenant_at_risk": most_at_risk,
            "current_ratios": {
                "leverage_ratio": float(leverage_ratio),
                "dscr": float(dscr),
                "current_ratio": float(current_ratio)
            },
            "thresholds": {
                "leverage_ratio": 4.0,
                "dscr": 1.5,
                "current_ratio": 1.2
            }
        }


def train_model():
    print("=" * 70)
    print("LMA PULSE - Dendritic Covenant Breach Predictor Training")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    train_dataset = CovenantDataset(num_samples=1000)
    val_dataset = CovenantDataset(num_samples=200)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = CovenantBreachPredictor(
        financial_metrics_dim=12,
        context_dim=8,
        num_dendritic_segments=8,
        hidden_dim=128
    )

    trainer = CovenantTrainer(model, learning_rate=0.001)
    trained_model = trainer.fit(train_loader, val_loader, epochs=10) # Reduced epochs for startup speed

    msg = "covenant_breach_model.pth"
    torch.save(trained_model.state_dict(), msg)
    print(f"Model saved to: {msg}")

if __name__ == "__main__":
    train_model()
