#!/usr/bin/env python3
# models.py - SHARED COMPATIBLE CNN MODEL FOR ALL COMPONENTS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle




class FashionMNISTCNN(nn.Module):
    """
    COMPATIBLE CNN model for Fashion-MNIST.
    Used by ALL components: Cloud, Edge, and Device.
    MUST be exactly the same architecture everywhere.
    """
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layers (EXACTLY as in cloud.py)
        self.dropout1 = nn.Dropout2d(0.25)  # Spatial dropout
        self.dropout2 = nn.Dropout(0.5)     # Regular dropout
        
        # Fully connected layers (EXACTLY as in cloud.py)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        MUST be exactly the same as in cloud.py and edge.py
        """
        # Layer 1: Conv -> ReLU -> Pool
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        # Layer 2: Conv -> ReLU -> Pool -> Dropout
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def get_parameter_names(self):
        """Get list of parameter names (for debugging)"""
        return [name for name, _ in self.named_parameters()]
    
    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




def get_model_parameters(model, as_numpy=True):
    """
    Extract model parameters.
    
    Args:
        model: FashionMNISTCNN instance
        as_numpy: If True, convert to numpy arrays (for serialization)
    
    Returns:
        List of parameters (as numpy arrays or torch tensors)
    """
    params = []
    for param in model.parameters():
        if as_numpy and torch.is_tensor(param.data):
            params.append(param.data.cpu().numpy().copy())
        else:
            params.append(param.data.clone())
    
    return params




def set_model_parameters(model, params, strict=True):
    """
    Set model parameters from list.
    
    Args:
        model: FashionMNISTCNN instance
        params: List of parameters (numpy arrays or torch tensors)
        strict: If True, require exact parameter count match
    
    Returns:
        bool: Success status
    """
    try:
        model_params = list(model.parameters())
        
        # Check parameter count
        if len(model_params) != len(params):
            if strict:
                raise ValueError(f"Parameter count mismatch: model has {len(model_params)}, "
                               f"given {len(params)}")
            else:
                print(f"⚠️ Parameter count mismatch: model has {len(model_params)}, "
                      f"given {len(params)}. Attempting to load available parameters...")
        
        # Load parameters
        for i, (model_param, new_param) in enumerate(zip(model_params, params)):
            if i >= len(params):
                break  # No more parameters to load
                
            # Convert numpy to torch tensor if needed
            if isinstance(new_param, np.ndarray):
                new_param = torch.from_numpy(new_param)
            
            # Check shape compatibility
            if model_param.shape != new_param.shape:
                print(f"⚠️ Shape mismatch at parameter {i}: "
                      f"model expects {model_param.shape}, got {new_param.shape}")
                continue
            
            # Copy parameter
            model_param.data.copy_(new_param)
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting model parameters: {e}")
        return False




def create_model_from_weights(weights, device='cpu'):
    """
    Create and initialize model from weights.
    
    Args:
        weights: List of parameters (from get_model_parameters)
        device: Device to load model on
    
    Returns:
        FashionMNISTCNN model with loaded weights
    """
    model = FashionMNISTCNN().to(device)
    success = set_model_parameters(model, weights)
    if not success:
        print("⚠️ Failed to load all weights, using default initialization")
    return model




def serialize_model(model):
    """Serialize model to bytes for network transfer"""
    params = get_model_parameters(model, as_numpy=True)
    return pickle.dumps(params)




def deserialize_model(model_bytes, device='cpu'):
    """Deserialize model from bytes"""
    params = pickle.loads(model_bytes)
    model = create_model_from_weights(params, device)
    return model




def test_model_compatibility():
    """Test that model is compatible across components"""
    print("🧪 Testing model compatibility...")
    
    # Create two models
    model1 = FashionMNISTCNN()
    model2 = FashionMNISTCNN()
    
    # Get parameters
    params1 = get_model_parameters(model1)
    params2 = get_model_parameters(model2)
    
    # Check parameter count
    if len(params1) != len(params2):
        print("❌ Parameter count mismatch!")
        return False
    
    print(f"✅ Parameter count: {len(params1)}")
    print(f"✅ Total parameters: {model1.count_parameters():,}")
    
    # Check parameter names
    param_names = model1.get_parameter_names()
    print(f"✅ Parameter names: {param_names}")
    
    # Test serialization/deserialization
    try:
        # Serialize
        model_bytes = serialize_model(model1)
        
        # Deserialize
        model3 = deserialize_model(model_bytes)
        
        # Test forward pass
        test_input = torch.randn(1, 1, 28, 28)
        output1 = model1(test_input)
        output3 = model3(test_input)
        
        if torch.allclose(output1, output3, rtol=1e-3):
            print("✅ Serialization/deserialization test passed")
            return True
        else:
            print("❌ Output mismatch after serialization")
            return False
            
    except Exception as e:
        print(f"❌ Serialization test failed: {e}")
        return False




def get_model_summary():
    """Print model summary"""
    model = FashionMNISTCNN()
    print("📋 Model Architecture Summary:")
    print("=" * 50)
    print(model)
    print("=" * 50)
    print(f"Total parameters: {model.count_parameters():,}")
    print("Parameter breakdown:")
    
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            print(f"  {name:30s}: {param_count:8,d} ({param.shape})")
    
    print(f"  {'Total':30s}: {total_params:8,d}")
    
    # Expected parameter count for verification
    expected_params = [
        (32 * 1 * 3 * 3) + 32,      # conv1: 320
        (64 * 32 * 3 * 3) + 64,     # conv2: 18,496
        (256 * 3136) + 256,         # fc1: 803,072
        (128 * 256) + 128,          # fc2: 32,896
        (10 * 128) + 10,            # fc3: 1,290
    ]
    print(f"Expected total: {sum(expected_params):,}")
    return model




if __name__ == "__main__":
    # Run compatibility tests
    print("🚀 Fashion-MNIST CNN Model Compatibility Check")
    print("=" * 60)
    
    # Print model summary
    model = get_model_summary()
    
    # Run compatibility tests
    print("\n🔧 Running compatibility tests...")
    if test_model_compatibility():
        print("\n✅ ALL TESTS PASSED - Model is compatible!")
        print("   This model can be used by Cloud, Edge, and Device components.")
    else:
        print("\n❌ Compatibility tests failed!")
        print("   Fix the model before deploying.")




