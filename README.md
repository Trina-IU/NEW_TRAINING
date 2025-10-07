# Handwriting OCR Training Pipeline

This repository now contains a reusable training script that turns the images in `Handwriting_Dataset/` into a TensorFlow Lite bundle ready for Android integration. Each image filename encodes the exact label to learn—suffixes such as ` (2)` are ignored automatically.

## 🚀 Quickstart Checklist

1. **Install Python 3.10+ (64-bit)** if you don't already have it.
2. **Create and activate a virtual environment** at the project root:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. **Install training dependencies**:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Run a quick smoke test** to make sure everything is wired correctly:
   ```powershell
   python scripts/train_ocr.py --dataset-dir Handwriting_Dataset --output-dir models/smoke --epochs 1 --max-samples 60 --max-steps-per-epoch 4 --cache
   ```
   This uses a subset of images so it finishes in a couple of minutes. Confirm that `models/smoke/ocr_model.tflite` and `models/smoke/labels.txt` are produced.
5. **Launch the full training run** (adjust hyperparameters if you have a GPU or more time):
   ```powershell
   python scripts/train_ocr.py --dataset-dir Handwriting_Dataset --output-dir models/ocr --epochs 45 --batch-size 32 --cache --fine-tune-fraction 0.2 --mixed-precision
   ```
   Expect ~30–45 minutes on a modern CPU. Watch the printed validation accuracy—>90% is a good target. The best model checkpoint is automatically restored before export.

## 🧠 How the script interprets labels

- The label comes from the image filename (e.g., `Amoxicillin.jpg`, `Take 1 tsp every 6 hours (2).png`).
- Any trailing ` (number)` is stripped, underscores become spaces, and multiple spaces collapse into one.
- Unicode characters such as `°` are preserved (so `for fever above 38°c` remains intact).

## 📦 Training Outputs

After each run you'll find the following inside the chosen `--output-dir`:

- `ocr_model.tflite` – quantized TensorFlow Lite classification model (ready for Android Studio).
- `labels.txt` – one label per line matching the TFLite output indices.
- `saved_model/` – standard TensorFlow SavedModel (handy if you need to re-export or serve on desktop).
- `metrics.json` – evaluation metrics on validation/test splits.
- `history.json` & `training_curves.png` – per-epoch training curves for documentation.

## 📲 Android Studio Integration Steps

1. **Copy the TFLite bundle** (`ocr_model.tflite` and `labels.txt`) into your Android module, typically under `app/src/main/ml/`.
2. **Add TensorFlow Lite dependencies** in `app/build.gradle`:
   ```groovy
   implementation "org.tensorflow:tensorflow-lite-task-vision:0.4.4"
   ```
3. **Load and run inference** using the TensorFlow Lite Task Library:
   ```kotlin
   import org.tensorflow.lite.task.vision.classifier.ImageClassifier
   import org.tensorflow.lite.task.vision.classifier.Classifications
   import org.tensorflow.lite.task.vision.classifier.ImageClassifier.ImageClassifierOptions
   import org.tensorflow.lite.task.core.BaseOptions
   import org.tensorflow.lite.support.image.TensorImage

   private fun createClassifier(context: Context): ImageClassifier {
       val baseOptions = BaseOptions.builder()
           .setNumThreads(4)
           .build()
       val options = ImageClassifierOptions.builder()
           .setBaseOptions(baseOptions)
           .setMaxResults(3)
           .setScoreThreshold(0.35f)
           .build()
       return ImageClassifier.createFromFileAndOptions(
           context,
           "ocr_model.tflite",
           options
       )
   }

   private fun runInference(bitmap: Bitmap, classifier: ImageClassifier): List<Classifications.Category> {
       val image = TensorImage.fromBitmap(bitmap)
       val results = classifier.classify(image)
       return results.firstOrNull()?.categories ?: emptyList()
   }
   ```
4. **Map predictions to friendly text**: each `Category` already contains the label string from `labels.txt`. Display the top prediction in your UI and optionally expose the top-3 results with confidence bars.
5. **Bundle labels**: place `labels.txt` next to the model or package it under `app/src/main/assets/` if you prefer manual loading.
6. **Test on-device**: deploy to an emulator/physical device, draw or photograph sample handwriting, and ensure the predictions align with expectations.

## 🧪 Recommended Validation Before Defense Day

- **Hold-out evaluation**: Review `metrics.json` to confirm you reach the desired accuracy (ideally >90%).
- **Manual spot checks**: Use a small Android utility screen that lets you load gallery images and prints the classifier output—great for your live defense.
- **Confusion patterns**: If accuracy is below expectations, inspect low-confidence labels and consider adding more samples or applying fine-tuning (`--fine-tune-fraction 0.4`) for a few extra epochs.

## 🛠️ Helpful Script Flags

| Flag | Purpose |
|------|---------|
| `--max-samples` | Train on a subset (great for experiments/debugging). |
| `--fine-tune-fraction` | Unfreeze a portion of MobileNetV2 for extra accuracy. |
| `--cache` | Keep datasets in memory once loaded (faster training on repeated runs). |
| `--disable-quantization` | Export a float model if you hit accuracy drops from quantization. |
| `--max-steps-per-epoch` | Cap training iterations (smoke tests / CI). |
| `--num-threads` | Pin TensorFlow threads for reproducibility. |
| `--mixed-precision` | Enable float16 mixed-precision when a GPU is detected (faster & memory-friendly). |
| `--force-cpu` | Ignore GPUs and stick to CPU (useful for debugging). |

## ⚡ GPU acceleration tips

- TensorFlow automatically switches to GPU when one is available. The script now prints whether GPU or CPU is active at startup and safely falls back to CPU if configuration fails.
- Install the matching CUDA Toolkit and cuDNN versions for your TensorFlow build (for 2.20, CUDA 12.4 + cuDNN 9.1). Verify with `nvidia-smi` before launching training.
- Pass `--mixed-precision` to squeeze extra throughput on RTX/RTX-series cards; on unsupported hardware the flag is ignored.
- If you hit GPU driver issues, retry with `--force-cpu` while you diagnose the setup.

## 📈 Next Steps & Tips

- Run the full training as soon as possible so you can iterate if accuracy is low.
- Capture screenshots of `training_curves.png` and validation metrics for your presentation slides.
- Keep at least one backup of the trained `ocr_model.tflite` in cloud storage and on a USB drive for defense day.
- If you plan to accept live handwriting input, normalize camera images to square 224×224 bitmaps before inference (center-crop + resize).

You're now ready to train and integrate the OCR model into your Android app. Good luck with your defense! 🎯
