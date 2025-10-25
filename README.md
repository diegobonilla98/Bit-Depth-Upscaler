# BitDepthUpscaler

**BitDepthUpscaler** is a deep learning framework for *safe perceptual dequantization*: reconstructing **16-bit per channel** images from standard **8-bit inputs** while preserving quantization consistency and reducing banding artifacts under tone edits.

Traditional 8-bit imagery (256 levels per channel) suffers from *false contouring* when subjected to gamma correction, contrast adjustments, or heavy post-processing. Professional workflows often rely on 16-bit intermediates, but such data may not be available in consumer archives.  
BitDepthUpscaler addresses this gap by predicting **bounded residuals** inside each quantization bin, ensuring safety and fidelity while restoring smooth gradients.

---

## Features

- **Residual dequantization with LSB bounds**: network outputs are strictly constrained within quantization bins.
- **Multiple architectures**:
  - Baseline U-Net (safe, bin-adherent)
  - Augmented U-Net (domain augmentations, relaxed bounds)
  - ConvNeXt-Attention U-Net (higher capacity, stronger banding suppression)
- **Sliding-window inference** with overlap-blend stitching for arbitrarily large images.
- **Comprehensive evaluation suite**:
  - Fidelity: PSNR, SSIM (sRGB + linear)
  - Perceptual: Oklab ΔE, LPIPS (optional)
  - Quantization safety: bin-adherence, re-quant match rate, idempotence drift
  - Banding stress: zero-run statistics after γ edits (0.6, 1.6)

---

## Project Structure

```
ImageBitDepthSR/
├── models.py                    # Neural network architectures (U-Net variants)
├── inference.py                 # Main inference script for evaluation
├── inference_utils.py           # Utility functions for image I/O and processing
├── bit_depth_color_info.py      # Color space conversion and bit depth utilities
│
├── Training Scripts:
├── first_train.py              # Initial training stage
├── second_train.py             # Second training stage (with augmentations)
├── third_train.py              # Final training stage (ConvNeXt-Attention)
│
├── Data Management:
├── download_raw_images.py      # Download RAW images from external sources
├── download_aws.py             # AWS S3 download utilities
├── upload_aws.py               # AWS S3 upload utilities
├── create_fake.py              # Generate synthetic test images
│
├── Model Checkpoints:
├── first_step_000002000.pt     # First stage trained model
├── second_step_0002000.pt      # Second stage trained model
├── third_step_0001750.pt       # Final stage trained model
│
├── Evaluation:
├── eval_results.json           # Evaluation metrics and results
├── eval_out/                   # Output directory for processed images
│   └── r00d161b7t_dequant.tiff
├── val_files.txt               # List of validation files
│
└── __pycache__/                # Python bytecode cache
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Dependencies
```bash
pip install torch torchvision torchaudio
pip install accelerate
pip install rawpy
pip install pillow
pip install tifffile
pip install imageio
pip install tqdm
pip install boto3
pip install numpy
```

## Usage

### Training

The project uses a three-stage training approach:

1. **First Stage** (`first_train.py`):
   ```bash
   python first_train.py
   ```
   - Basic U-Net architecture
   - Safe residual prediction with LSB bounds
   - Conservative learning rate

2. **Second Stage** (`second_train.py`):
   ```bash
   python second_train.py
   ```
   - Enhanced U-Net with domain augmentations
   - Relaxed bounds for better reconstruction
   - Tone curve jitter for robustness

3. **Third Stage** (`third_train.py`):
   ```bash
   python third_train.py
   ```
   - ConvNeXt-Attention U-Net
   - Highest capacity model
   - Strongest banding suppression

### Configuration

Each training script contains user-configurable parameters at the top:

```python
RAW_GLOB = "./NEF/*.NEF"        # Path to RAW image dataset
PATCH = 128                      # Training patch size
BATCH = 16                       # Batch size
EPOCHS = 20                      # Number of epochs
LR = 1e-3                        # Learning rate
WIDTH = 32                       # U-Net base channels
USE_DITHER = True                # Enable dithering
USE_TONE_JITTER = True           # Enable tone curve jitter
```

### Inference

Run inference on your images:

```bash
python inference.py
```

Configure the inference parameters in the script:
- `CHECKPOINT_PATH`: Path to model checkpoint
- `INPUT_DIRS`: Input directories or files
- `INPUT_BASENAMES_FILE`: Optional file list
- `OUTPUT_JSON`: Results output file
- `SAVE_OUTPUTS`: Whether to save processed images

### Data Preparation

1. **Download RAW images**:
   ```bash
   python download_raw_images.py
   ```

2. **Create synthetic test data**:
   ```bash
   python create_fake.py
   ```

3. **AWS S3 integration** (optional):
   ```bash
   python download_aws.py    # Download from S3
   python upload_aws.py      # Upload to S3
   ```

## Model Architectures

### FirstDequantUNet
- Basic U-Net with residual prediction
- Zero-initialized head for identity start
- Strict LSB bounds for safety

### SecondDequantUNet
- Enhanced U-Net with domain augmentations
- Relaxed bounds for better reconstruction
- Tone curve jitter integration

### ThirdDequantUNet
- ConvNeXt-Attention U-Net
- Higher capacity with attention mechanisms
- Strongest banding artifact suppression

## Evaluation Metrics

The framework provides comprehensive evaluation:

- **Fidelity Metrics**:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - Both sRGB and linear color spaces

- **Perceptual Metrics**:
  - Oklab ΔE (color difference)
  - LPIPS (Learned Perceptual Image Patch Similarity)

- **Quantization Safety**:
  - Bin-adherence rate
  - Re-quantization match rate
  - Idempotence drift

- **Banding Stress Tests**:
  - Zero-run statistics after gamma edits (0.6, 1.6)
  - Tone curve robustness

## File Formats

### Input
- **RAW formats**: NEF, DNG, CR2, CR3, ARW, RW2, ORF, RAF, SRW, PEF, NRW
- **Standard formats**: PNG, JPEG, TIFF (8-bit)

### Output
- **16-bit TIFF**: Primary output format
- **Evaluation JSON**: Comprehensive metrics
- **Processed images**: High bit-depth reconstructions

## Advanced Features

### Sliding Window Inference
- Handles arbitrarily large images
- Overlap-blend stitching for seamless results
- Memory-efficient processing

### Color Space Support
- sRGB to linear conversion
- Oklab perceptual color space
- Multiple bit depth handling (8, 10, 12, 16-bit)

### Training Augmentations
- Stochastic dithering
- Tone curve jitter
- Domain-specific augmentations
- Gradient-based losses

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size
   - Use smaller patch sizes
   - Enable gradient checkpointing

2. **RAW file reading errors**:
   - Ensure rawpy is properly installed
   - Check file format compatibility
   - Verify file permissions

3. **Training instability**:
   - Reduce learning rate
   - Enable gradient clipping
   - Check data preprocessing

### Performance Tips

- Use SSD storage for faster I/O
- Enable mixed precision training
- Monitor GPU memory usage
- Use appropriate batch sizes for your hardware

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{bitdepthupscaler2024,
  title={Safe Perceptual Dequantization: Reconstructing 16-bit Images from 8-bit Inputs},
  author={Diego Bonilla Salvador},
  url = {https://github.com/diegobonilla98/Bit-Depth-Upscaler},
  year={2024}
}
```

## Acknowledgments

- RawPy library for RAW image processing
- PyTorch team for the deep learning framework
- Accelerate library for training optimization
- The open-source computer vision community


