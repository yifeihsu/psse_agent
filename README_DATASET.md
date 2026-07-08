# Power System Diagnostic Dataset - Quick Start Guide

This guide will help you understand, preprocess, and analyze your power system diagnostic dataset for LLM fine-tuning.

## 📁 Files Overview

### Documentation
- **[DATASET_DOCUMENTATION.md](./DATASET_DOCUMENTATION.md)** - Complete reference guide (800+ lines)
  - Dataset structure and schema
  - Statistical analysis
  - Example data samples
  - Complete preprocessing pipeline explanation
  - Code snippets library
  - Best practices and troubleshooting

### Scripts
- **[preprocess_dataset.py](./preprocess_dataset.py)** - Preprocessing pipeline
  - Load and validate data
  - Balance classes
  - Create stratified train/val/test splits
  - Generate statistics

- **[analyze_dataset.py](./analyze_dataset.py)** - Analysis and visualization
  - Class distribution plots
  - Measurement value distributions
  - Residual analysis
  - Tool call patterns
  - Confidence score analysis

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Basic requirements
pip install numpy scikit-learn tqdm

# For visualization (optional)
pip install matplotlib seaborn pandas
```

### 2. Run Preprocessing

```bash
# Preprocess dataset with default settings
python preprocess_dataset.py

# This will create:
# - data/processed/train.jsonl
# - data/processed/val.jsonl
# - data/processed/test.jsonl
# - data/processed/statistics.json
```

**Output**:
```
Step 1: Loading dataset from data/sft_with_tools.jsonl...
  ✓ Loaded 1978 samples

Step 2: Running quality checks...
  ✓ No quality issues found

Step 3: Balancing classes (target: 500 per class)...
  measurement_error: 500 → 500
  no_error: 500 → 500
  parameter_error: 500 → 500
  topology_error: 478 → 500 (oversampled)
  ✓ Balanced dataset size: 2000

Step 4: Creating stratified splits...
  ✓ Train: 1400 samples
  ✓ Val: 300 samples
  ✓ Test: 300 samples

Step 5: Saving processed datasets...
  ✓ Saved to data/processed/

✓ Ready for fine-tuning!
```

### 3. Analyze Dataset (Optional)

```bash
# Generate analysis report and visualizations
python analyze_dataset.py --file data/sft_with_tools.jsonl

# Save plots instead of displaying
python analyze_dataset.py --file data/sft_with_tools.jsonl --no-show --output-dir data/analysis
```

This generates:
- Summary statistics (printed to console)
- 5 visualization plots (shown or saved)

## 📊 Dataset Structure

### File Formats

Each `.jsonl` file contains one JSON object per line:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "{\"case\": \"case14\", \"z_obs\": [...], ...}"},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "content": "..."},
    {"role": "assistant", "content": "{\"has_error\": ..., ...}"}
  ]
}
```

### Data Files

| File | Samples | Purpose | Balanced? |
|------|---------|---------|-----------|
| `sft_final.jsonl` | 2,487 | Main dataset | ❌ No |
| `sft_with_tools.jsonl` | 1,978 | Extended dataset | ✅ Yes |
| `sft_test.jsonl` | 14 | Test set | ❌ No (topology only) |

**Recommended**: Use `sft_with_tools.jsonl` for training.

### Error Classes

1. **no_error** (20-25%) - All measurements valid
2. **measurement_error** (25-32%) - Bad sensor reading
3. **parameter_error** (25-40%) - Incorrect network parameters
4. **topology_error** (7-24%) - Circuit breaker state mismatch

## 🔧 Customization

### Modify Preprocessing Config

Edit `preprocess_dataset.py`:

```python
CONFIG = {
    'input_file': 'data/sft_with_tools.jsonl',  # Change source file
    'output_dir': 'data/processed',             # Change output directory
    'balance_classes': True,                    # Enable/disable balancing
    'target_per_class': 500,                    # Samples per class
    'train_ratio': 0.70,                        # 70% for training
    'val_ratio': 0.15,                          # 15% for validation
    'test_ratio': 0.15,                         # 15% for testing
    'random_seed': 42                           # For reproducibility
}
```

### Use in Your Training Script

```python
from datasets import load_dataset

# Load processed dataset
dataset = load_dataset('json', data_files={
    'train': 'data/processed/train.jsonl',
    'validation': 'data/processed/val.jsonl',
    'test': 'data/processed/test.jsonl'
})

# Use with HuggingFace Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    # ... other args
)
```

## 📖 Learning Path

1. **Start here**: Read the [Quick Start](#-quick-start) section
2. **Understand the data**: Review [DATASET_DOCUMENTATION.md](./DATASET_DOCUMENTATION.md) sections 1-5
3. **Run preprocessing**: Execute `python preprocess_dataset.py`
4. **Analyze results**: Run `python analyze_dataset.py`
5. **Deep dive**: Read full documentation sections 6-12
6. **Fine-tune**: Use processed data with your notebook

## 🎯 Common Tasks

### Task 1: Create Balanced Training Set

```bash
python preprocess_dataset.py
# Uses sft_with_tools.jsonl (already balanced)
# Outputs to data/processed/
```

### Task 2: Understand Class Distribution

```python
import json

with open('data/processed/statistics.json', 'r') as f:
    stats = json.load(f)

print("Training set class distribution:")
for cls, count in stats['splits']['train']['class_distribution'].items():
    print(f"  {cls}: {count}")
```

### Task 3: Extract Single Sample for Inspection

```python
import json

# Load first sample from training set
with open('data/processed/train.jsonl', 'r') as f:
    sample = json.loads(f.readline())

# Pretty print
print(json.dumps(sample, indent=2))
```

### Task 4: Get Measurement Statistics

```python
from preprocess_dataset import extract_features
import json

with open('data/processed/train.jsonl', 'r') as f:
    sample = json.loads(f.readline())

features = extract_features(sample)
print(f"Measurement vector shape: {features['z_obs'].shape}")
print(f"Voltage range: [{features['z_obs'][0:14].min():.3f}, {features['z_obs'][0:14].max():.3f}]")
print(f"Error family: {features['error_family']}")
```

### Task 5: Visualize Data Distribution

```bash
# Interactive plots
python analyze_dataset.py

# Save plots to file
python analyze_dataset.py --no-show --output-dir data/plots
```

## 🔍 Troubleshooting

### Error: File not found

**Problem**: `FileNotFoundError: data/sft_with_tools.jsonl`

**Solution**:
```bash
# Check if file exists
ls -lh data/

# Update CONFIG['input_file'] in preprocess_dataset.py
# Or specify full path
```

### Warning: Class imbalance detected

**Problem**: Dataset has unbalanced error families

**Solution**: Already handled! The preprocessing script balances classes automatically.

### Low memory during processing

**Problem**: System runs out of RAM

**Solution**:
```python
# Edit preprocess_dataset.py
# Process in smaller batches or reduce sample count

# In quality_check() and balance_dataset():
# Add batch processing
```

## 📈 Next Steps After Preprocessing

1. **Fine-tune your model** with `submit_sft_gemma4.sh` or `gpt_oss_power_sft_revised_v3.py`
2. **Evaluate performance** on the test set
3. **Analyze errors** by error family
4. **Iterate**: Adjust preprocessing parameters based on results

## 🛠️ Advanced Usage

### Custom Feature Engineering

```python
from preprocess_dataset import extract_features, load_dataset
import numpy as np

def add_statistical_features(sample):
    """Add summary statistics as metadata."""
    features = extract_features(sample)

    if features['z_obs'] is not None:
        # Calculate statistics
        voltage_mean = features['z_obs'][0:14].mean()
        voltage_std = features['z_obs'][0:14].std()

        # Add to sample
        sample['metadata'] = {
            'voltage_mean': float(voltage_mean),
            'voltage_std': float(voltage_std)
        }

    return sample

# Apply to dataset
samples = load_dataset('data/sft_with_tools.jsonl')[0]
enhanced_samples = [add_statistical_features(s) for s in samples]
```

### Data Augmentation

```python
import random
import numpy as np

def augment_sample(sample, noise_std=0.01):
    """Add Gaussian noise to measurements."""
    features = extract_features(sample)

    if features['z_obs'] is not None:
        # Add noise
        noise = np.random.normal(0, noise_std, features['z_obs'].shape)
        z_augmented = features['z_obs'] + noise

        # Update sample
        for msg in sample['messages']:
            if msg['role'] == 'user':
                user_data = json.loads(msg['content'])
                user_data['z_obs'] = z_augmented.tolist()
                msg['content'] = json.dumps(user_data)
                break

    return sample

# Create augmented version
augmented = augment_sample(sample.copy(), noise_std=0.02)
```

## 📚 Additional Resources

- **Full Documentation**: [DATASET_DOCUMENTATION.md](./DATASET_DOCUMENTATION.md)
- **PyPower**: Power system simulation library
- **Unsloth**: Efficient LLM fine-tuning
- **HuggingFace TRL**: Training utilities

## 💡 Tips

1. **Always use balanced dataset** (`sft_with_tools.jsonl`) for training
2. **Check statistics.json** after preprocessing to verify class balance
3. **Use visualization** to understand data distributions
4. **Start with default config** then iterate based on results
5. **Track experiments** with wandb or tensorboard

## ❓ FAQ

**Q: Which dataset file should I use?**
A: Use `sft_with_tools.jsonl` - it's balanced across all error families.

**Q: How do I know if preprocessing worked?**
A: Check `data/processed/statistics.json` - all classes should have ~equal counts.

**Q: Can I change the train/val/test split?**
A: Yes, modify `CONFIG` in `preprocess_dataset.py`.

**Q: How do I add more data?**
A: Combine multiple `.jsonl` files before preprocessing, or use the `artifacts/measurements/out_sft_measurements/` directory.

**Q: What if I get memory errors?**
A: Process in batches or reduce the number of samples used for statistics.

---

**Happy Fine-Tuning! 🚀**

For detailed information, refer to [DATASET_DOCUMENTATION.md](./DATASET_DOCUMENTATION.md)
