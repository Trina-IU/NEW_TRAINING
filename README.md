# Handwriting OCR Training Pipeline

This repository provides a full pipeline that turns the handwritten images in `Handwriting_Dataset/` into a TensorFlow Lite bundle ready for Android deployment. The workflow now has two clear phases:

1. **Dataset preparation** – clean filenames, normalize every image to 128×128 grayscale, create ten augmented variants per source, and split into train/val/test.
2. **Model training** – train a custom CNN on the prepared splits, track metrics, and export SavedModel + TFLite artifacts.

## 🚀 Quickstart Checklist

1. **Install Python 3.10+ (64-bit)**.
2. **Create & activate a virtual environment** (recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. **Install dependencies**:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Prepare the dataset** (writes an augmented copy with train/val/test splits):
   ```powershell
   python scripts/prepare_dataset.py `
       --dataset-dir Handwriting_Dataset `
       --output-dir Handwriting_Dataset_Augmented `
       --augmentations-per-image 10 `
       --image-size 128 `
       --force
   ```
   The output folder will contain `train/`, `val/`, `test/`, plus a `class_labels.json` lookup table.
5. **Run a quick smoke test** (limits each split so it finishes fast):
   ```powershell
   python scripts/train_ocr.py `
       --dataset-dir Handwriting_Dataset_Augmented `
       --output-dir models/smoke `
       --epochs 2 `
       --batch-size 32 `
       --train-sample-limit 300 `
       --val-sample-limit 60 `
       --test-sample-limit 60 `
       --cache
   ```
   Confirm the command produces `models/smoke/ocr_model.tflite`, `labels.txt`, and the CSV/JSON logs.
6. **Launch the full training run** (tweak for your hardware):
   ```powershell
   python scripts/train_ocr.py `
       --dataset-dir Handwriting_Dataset_Augmented `
       --output-dir models/ocr `
       --epochs 60 `
       --batch-size 64 `
       --cache `
       --mixed-precision
   ```
   Expect roughly 30–40 minutes on a modern GPU-enabled laptop; CPU-only runs will take longer. The best checkpoint is restored automatically before exporting.

## 🧠 Label semantics

- Labels come from the original filenames (e.g., `Ambroxol.jpg`, `Take 1 tsp every 6 hours (2).png`).
- Trailing counts like ` (2)` are stripped, underscores turn into spaces, and whitespace collapses.
- Unicode such as `°` is preserved thanks to NFKC normalization, so labels stay human-readable.

## 🧳 Dataset preparation script

`scripts/prepare_dataset.py` performs every preprocessing step needed for robust training:

- Renames Windows-reserved paths (PRN, AUX, etc.) to avoid copy issues.
- Drops empty/corrupt files, converts images to 128×128 grayscale, and stores them as PNG.
- Generates configurable augmentation variants via Keras (default ×10).
- Splits into train/val/test at 80/10/10 while keeping folder structure intact.
- Emits `class_labels.json`, mapping each safe slug (e.g., `prn`) back to the original label (`PRN`).

Run `python scripts/prepare_dataset.py --help` to view all options.

## 📦 Training outputs

Every call to `scripts/train_ocr.py` produces these artifacts inside `--output-dir`:

- `ocr_model.tflite` – quantized TensorFlow Lite classifier for Android.
- `saved_model/` – TensorFlow SavedModel (handy for troubleshooting or re-export).
- `best_model.keras` – best-performing Keras checkpoint (restored before export).
- `labels.txt` – label order matching the model output indices.
- `metrics.json` – evaluation metrics, sample counts, and runtime metadata.
- `history.json` + `training_curves.png` – accuracy/loss curves per epoch.
- `training_log.csv` – CSV logger ready for spreadsheets or MLflow import.

## 🛠️ Key training flags

| Flag | Purpose |
|------|---------|
| `--train-sample-limit / --val-sample-limit / --test-sample-limit` | Randomly subsample each split (ideal for smoke tests). |
| `--cache` | Cache datasets in memory for faster subsequent epochs. |
| `--early-stopping-patience` | Stop training when validation accuracy plateaus (set to `0` to disable). |
| `--reduce-lr-patience` | Reduce learning rate when validation loss stalls. |
| `--disable-quantization` | Export a float32 TFLite model instead of dynamic-range quantized. |
| `--num-threads` | Control TensorFlow intra/inter-op threads for reproducibility. |
| `--mixed-precision` | Enable float16 training when a compatible GPU is detected. |
| `--force-cpu` | Ignore GPUs and run everything on CPU (helpful for debugging drivers). |

Run `python scripts/train_ocr.py --help` for the full list.

## ⚡ GPU acceleration tips

- TensorFlow automatically uses available GPUs. The script prints the detected device at startup and falls back gracefully if configuration fails.
- For TensorFlow 2.20, install CUDA 12.4 + cuDNN 9.1 (or the versions recommended by `pip install tensorflow`).
- Pass `--mixed-precision` on RTX/RTX-series GPUs to unlock float16 speed-ups. The flag is ignored safely on unsupported hardware.
- Use `--force-cpu` when diagnosing GPU driver issues to keep working while you troubleshoot.

## � Android Studio integration

1. Copy `ocr_model.tflite` and `labels.txt` into `app/src/main/ml/` (or your preferred assets folder).
2. Add the Task Library dependency:
   ```groovy
   implementation "org.tensorflow:tensorflow-lite-task-vision:0.4.4"
   ```
3. Create an `ImageClassifier`:
   ```kotlin
   private fun createClassifier(context: Context): ImageClassifier {
       val options = ImageClassifier.ImageClassifierOptions.builder()
           .setBaseOptions(BaseOptions.builder().setNumThreads(4).build())
           .setMaxResults(3)
           .setScoreThreshold(0.35f)
           .build()
       return ImageClassifier.createFromFileAndOptions(context, "ocr_model.tflite", options)
   }
   ```
4. Convert camera/gallery bitmaps to 128×128 grayscale (or let TensorFlow handle RGB—it works either way) before classification.
5. Display the top prediction and optionally show the top-3 confidence scores for better UX.

## 🧪 Validation before demo day

- Inspect `metrics.json` to confirm validation/test accuracy is where you expect (>90% is a solid target).
- Review `training_curves.png` to spot overfitting or stalled learning early.
- Run the model inside your Android UI with a handful of handwritten samples to catch any preprocessing mismatches.
- Keep a backup of the exported `ocr_model.tflite` and `labels.txt` in cloud storage plus a USB drive.

You're now ready to prep your data, train the CNN, and wire the TFLite model into Android Studio. Happy training! 🎯
