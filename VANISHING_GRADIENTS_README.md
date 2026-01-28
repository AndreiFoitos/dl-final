# Vanishing Gradients Exploration

This directory contains files for exploring vanishing gradients in RNNs and their mitigation techniques.

## Files

### `RNN_vanishing.py`
Contains model implementations with various mitigation strategies:
- **VanillaRNN**: Standard RNN (prone to vanishing gradients with deep networks and tanh activation)
- **ResidualRNN**: RNN with residual connections (mitigation)

Initialization methods:
- **Xavier/Glorot initialization**: Good default for RNNs
- **He initialization**: Better for ReLU activations

### `training_vanishing.py`
Training script that:
- Trains different model architectures
- Tracks gradient norms per layer
- Detects vanishing gradients (very small gradient norms)
- Generates comprehensive visualizations:
  - Training loss over epochs
  - Overall and minimum layer gradient norms
  - Gradient norms per layer (bar chart)
  - Gradient evolution heatmap across layers
  - Last 10 batch gradients

## Usage

### Basic Usage

```python
python training_vanishing.py
```

This will run multiple experiments comparing:
1. Vanilla RNN with tanh (no mitigation) - shows vanishing gradients
2. Vanilla RNN with Xavier initialization - mitigation via better init
3. Residual RNN with ReLU - residual connections help

### Custom Experiments

Modify the `experiments` list in `training_vanishing.py` to:
- Change model architectures
- Adjust hyperparameters (learning rate, layers, hidden size)
- Test different initialization methods
- Add custom mitigation techniques

### Key Parameters for Vanishing Gradients

**To induce vanishing gradients:**
- Deep networks (many layers, e.g., 20+)
- Tanh activation (saturates, causing vanishing)
- Small learning rates
- Long sequences

**Mitigation techniques:**
- Better initialization (Xavier/He)
- Residual connections
- ReLU activation (less prone to vanishing than tanh)
- Gradient clipping with minimum threshold

## Output

Plots are saved in the `plots/vanishing_gradient/` directory with naming convention:
- `training_loss_{model_type}_{mitigation}.png`
- `grad_norms_{model_type}_{mitigation}.png`
- `grad_per_layer_{model_type}_{mitigation}.png`
- `grad_heatmap_{model_type}_{mitigation}.png`
- `grad_norms_last_10_{model_type}_{mitigation}.png`

## Interpreting Results

**Vanishing gradients indicators:**
- Gradient norms decrease exponentially over epochs
- Minimum layer gradient norms become very small (< 1e-6)
- Training loss plateaus early or decreases very slowly
- Earlier layers have much smaller gradients than later layers

**Successful mitigation:**
- Gradient norms remain stable across epochs
- All layers maintain reasonable gradient magnitudes
- Training loss decreases steadily
- Similar gradient magnitudes across layers
