
import torch
import torch.nn as nn

class DendriticSegments(nn.Module):
    def __init__(self, input_dim, output_dim, context_dim):
        super(DendriticSegments, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # Context gate modulates the output based on loan metadata
        self.context_gate = nn.Sequential(
            nn.Linear(context_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x, context):
        # Ensure 'x' is the financial tensor and 'context' is the loan metadata
        # Strategy 1 requires the output of this module to be a single tensor 
        # that the library can then weight and sum at the 'neuron' level.
        
        # x shape: (batch_size, input_dim)
        # context shape: (batch_size, context_dim)
        
        base_output = self.linear(x)
        gate = self.context_gate(context)
        
        gated_output = gate * base_output
        return gated_output
