import torch
import torch.nn as nn
from torch.profiler import record_function

TIME_STATS = {}

class PerformanceProbe(nn.Module):
    def __init__(self, module, name):
        super().__init__()
        self.module = module
        self.name = name
        
        if name not in TIME_STATS:
            TIME_STATS[name] = {"fwd": [], "bwd": []}
            
        self.register_full_backward_pre_hook(self._backward_pre_hook)
        self.register_full_backward_hook(self._backward_end_hook)

    def forward(self, x):
        # --- Forward Start ---
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        
        # torch.cuda.synchronize() 
        start_evt.record()
        
        with record_function(f"Fwd_Recorded_{self.name}"):
            output = self.module(x)
        
        end_evt.record()
        # --- Forward End ---
        self.last_fwd_events = (start_evt, end_evt)
        
        # --- Hook for Backward Start ---
        if output.requires_grad:
            output.register_hook(self._create_backward_start_hook())
            
        return output

    def _create_backward_start_hook(self):
        def hook(grad):
            start_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            self.bwd_start_evt = start_evt
            return grad
        return hook
    
    def _backward_pre_hook(self, module, grad_output):
        # --- Backward Start ---
        start_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        self.bwd_start_evt = start_evt
        
        self.bwd_ctx = record_function(f"Bwd_Recorded_{self.name}")
        self.bwd_ctx.__enter__()

    def _backward_end_hook(self, module, grad_input, grad_output):
        end_evt = torch.cuda.Event(enable_timing=True)
        end_evt.record()
        if self.bwd_ctx:
            self.bwd_ctx.__exit__(None, None, None)
            self.bwd_ctx = None
        
        if hasattr(self, 'bwd_start_evt'):
            TIME_STATS[self.name]['bwd'].append((self.bwd_start_evt, end_evt))
        
        if hasattr(self, 'last_fwd_events'):
            TIME_STATS[self.name]['fwd'].append(self.last_fwd_events)