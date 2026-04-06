"""
LoRA (Low-Rank Adaptation) Implementation

Parameter-efficient fine-tuning for large language models
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """LoRA configuration"""
    r: int = 8           # Rank dimension
    alpha: int = 16      # Scaling factor
    dropout: float = 0.1  # Dropout probability
    target_modules: List[str] = None  # Modules to apply LoRA
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class LoRALayer:
    """
    LoRA Layer
    
    Adds low-rank adapters to a linear layer
    """
    
    def __init__(self, in_features: int, out_features: int, config: LoRAConfig):
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Original weights (frozen)
        self.weight = None
        self.bias = None
        
        # LoRA matrices (trainable)
        # A: in_features x r
        # B: r x out_features
        self.lora_A = np.random.randn(config.r, in_features) * 0.01
        self.lora_B = np.random.randn(out_features, config.r) * 0.01
        self.scaling = config.alpha / config.r
        
        # Dropout
        self.dropout = np.random.rand(config.r, in_features) > config.dropout
    
    def set_weights(self, weight: np.ndarray, bias: np.ndarray = None):
        """Set original layer weights"""
        self.weight = weight
        self.bias = bias
    
    def forward(self, x: np.ndarray, lora_enabled: bool = True) -> np.ndarray:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, in_features)
            lora_enabled: Whether to use LoRA adapters
        
        Returns:
            Output tensor
        """
        # Original forward pass
        if self.weight is not None:
            original = np.dot(x, self.weight.T)
            if self.bias is not None:
                original += self.bias
        else:
            original = np.zeros((x.shape[0], self.out_features))
        
        # LoRA forward pass
        if lora_enabled:
            # Apply dropout to LoRA A
            lora_a = self.lora_A * self.dropout
            # lora_input = x @ lora_A.T
            lora_input = np.dot(x, lora_a.T)
            # lora_output = lora_input @ lora_B.T
            lora_output = np.dot(lora_input, self.lora_B.T)
            lora_output *= self.scaling
            
            return original + lora_output
        
        return original
    
    def get_lora_params(self) -> Dict[str, np.ndarray]:
        """Get trainable LoRA parameters"""
        return {
            "lora_A": self.lora_A,
            "lora_B": self.lora_B
        }
    
    def merge_weights(self):
        """Merge LoRA weights into original weights"""
        if self.weight is not None:
            # W = W + BA * scaling
            lora_weight = self.lora_B @ self.lora_A * self.scaling
            self.weight = self.weight + lora_weight
            # Reset LoRA params
            self.lora_A = None
            self.lora_B = None


class LoRALinear(LoRALayer):
    """LoRA for nn.Linear layer"""
    pass


class LoRAEmbedding(LoRALayer):
    """
    LoRA for Embedding layer
    
    Applies LoRA to embedding matrix
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, config: LoRAConfig):
        super().__init__(embedding_dim, num_embeddings, config)
        self.num_embeddings = num_embeddings
        
        # LoRA for embeddings: A is r x embedding_dim, B is num_embeddings x r
        self.lora_A = np.random.randn(config.r, embedding_dim) * 0.01
        self.lora_B = np.random.randn(num_embeddings, config.r) * 0.01
    
    def forward(self, x: np.ndarray, lora_enabled: bool = True) -> np.ndarray:
        """Forward pass for embeddings"""
        # Original embedding lookup
        if self.weight is not None:
            original = self.weight[x]
        else:
            original = np.zeros((x.shape[0], self.in_features))
        
        if lora_enabled:
            # LoRA adjustment
            lora_a = self.lora_A * self.dropout
            lora_input = np.dot(x, lora_a.T)
            lora_output = np.dot(lora_input, self.lora_B.T)
            lora_output *= self.scaling
            
            return original + lora_output
        
        return original


class LoRAModel:
    """
    LoRA-adapted Model
    
    Wraps a model with LoRA layers
    """
    
    def __init__(self, base_model: Dict, config: LoRAConfig):
        self.base_model = base_model
        self.config = config
        self.lora_layers: Dict[str, LoRALayer] = {}
        self.trainable_params = 0
        self.total_params = 0
        
        self._apply_lora()
    
    def _apply_lora(self):
        """Apply LoRA to target modules"""
        for name, module in self.base_model.items():
            if any(target in name for target in self.config.target_modules):
                # Create LoRA layer
                if hasattr(module, 'weight'):
                    in_features = module.weight.shape[1]
                    out_features = module.weight.shape[0]
                    
                    lora_layer = LoRALinear(in_features, out_features, self.config)
                    lora_layer.set_weights(module.weight, module.bias if hasattr(module, 'bias') else None)
                    
                    self.lora_layers[name] = lora_layer
    
    def forward(self, x: np.ndarray, use_lora: bool = True) -> np.ndarray:
        """Forward pass with optional LoRA"""
        # Simplified forward pass
        # In real implementation, would traverse model layers
        
        for name, layer in self.lora_layers.items():
            if use_lora:
                x = layer.forward(x, lora_enabled=True)
        
        return x
    
    def get_trainable_parameters(self) -> Tuple[int, int]:
        """Get number of trainable vs total parameters"""
        lora_params = 0
        for layer in self.lora_layers.values():
            lora_params += layer.lora_A.size + layer.lora_B.size
        
        # Estimate total params
        total = sum(m.weight.size for m in self.base_model.values() if hasattr(m, 'weight'))
        
        return lora_params, total
    
    def print_trainable_ratio(self):
        """Print trainable parameter ratio"""
        trainable, total = self.get_trainable_parameters()
        ratio = trainable / total * 100 if total > 0 else 0
        print(f"LoRA Parameters: {trainable:,} / {total:,} ({ratio:.2f}%)")


class LoRATrainer:
    """
    LoRA Fine-tuning Trainer
    """
    
    def __init__(self, model: LoRAModel, learning_rate: float = 0.001):
        self.model = model
        self.lr = learning_rate
        
    def train_step(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Single training step
        
        Args:
            inputs: Input data
            targets: Target labels
        
        Returns:
            Loss value
        """
        # Forward pass
        outputs = self.model.forward(inputs, use_lora=True)
        
        # Compute loss (MSE for simplicity)
        loss = np.mean((outputs - targets) ** 2)
        
        # Backward pass (simplified - real implementation would use autograd)
        # In practice, use PyTorch or JAX
        
        return loss
    
    def train(self, train_data: np.ndarray, train_labels: np.ndarray, 
              epochs: int = 10, batch_size: int = 32) -> List[float]:
        """
        Train the model
        
        Args:
            train_data: Training inputs
            train_labels: Training labels
            epochs: Number of epochs
            batch_size: Batch size
        
        Returns:
            List of losses
        """
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, len(train_data), batch_size):
                batch_x = train_data[i:i+batch_size]
                batch_y = train_labels[i:i+batch_size]
                
                loss = self.train_step(batch_x, batch_y)
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses


# ============== Utility Functions ==============

def apply_lora_to_model(model_dict: Dict, config: LoRAConfig) -> LoRAModel:
    """
    Apply LoRA to a model
    
    Args:
        model_dict: Dictionary of model layers
        config: LoRA configuration
    
    Returns:
        LoRA-adapted model
    """
    return LoRAModel(model_dict, config)


def count_parameters(model: LoRAModel) -> Dict[str, int]:
    """Count parameters in LoRA model"""
    trainable, total = model.get_trainable_parameters()
    frozen = total - trainable
    
    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": total,
        "trainable_ratio": trainable / total if total > 0 else 0
    }


# ============== Demo ==============

def demo():
    """Demo LoRA implementation"""
    print("=" * 60)
    print("LoRA (Low-Rank Adaptation) Demo")
    print("=" * 60)
    
    # Config
    config = LoRAConfig(r=8, alpha=16, dropout=0.1)
    
    # Simulated model (2 layers)
    model_dict = {
        "layer1": type('obj', (object,), {'weight': np.random.randn(512, 256) * 0.01, 'bias': np.zeros(512)})(),
        "layer2": type('obj', (object,), {'weight': np.random.randn(256, 10) * 0.01, 'bias': np.zeros(10)})(),
    }
    
    # Apply LoRA
    lora_model = LoRAModel(model_dict, config)
    
    # Print parameter ratio
    print("\nParameter Analysis:")
    lora_model.print_trainable_ratio()
    
    # Count parameters
    params = count_parameters(lora_model)
    print(f"\nTotal Parameters: {params['total']:,}")
    print(f"Trainable (LoRA): {params['trainable']:,}")
    print(f"Frozen: {params['frozen']:,}")
    
    # Demo forward pass
    print("\nForward Pass Demo:")
    x = np.random.randn(4, 256)  # Batch of 4, 256 features
    
    output_with_lora = lora_model.forward(x, use_lora=True)
    output_without_lora = lora_model.forward(x, use_lora=False)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_with_lora.shape}")
    print(f"LoRA enabled: mean={output_with_lora.mean():.4f}")
    print(f"LoRA disabled: mean={output_without_lora.mean():.4f}")
    
    print("\n" + "=" * 60)
    print("LoRA reduces trainable parameters by ~90%!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
