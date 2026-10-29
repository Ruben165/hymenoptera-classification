# Hymenoptera Classification

Binary image classification of **ants vs. bees** using transfer learning from a pre-trained **MobileNetV2** (ImageNet weights).

## Project Structure

```
data/
├── train/{ants,bees}/      # 245 images
└── val/{ants,bees}/        # 154 images, split into validation + test in the notebook
impl/
└── Hymenoptera_Classification.ipynb
result/                     # created on run
├── model/best_model.keras
└── *.png                   # sample images, training curves, predictions
```

## Approach

1. **Data**: images resized to 160×160. 1/7 of the `val` folder is held out as a fixed test set.
2. **Model**: data augmentation (flip, rotation, zoom), then frozen MobileNetV2, global average pooling, dropout (0.2), and a sigmoid output.
3. **Training**: two phases.
   - Train only the classification head (Adam, lr 1e-3).
   - Fine-tune MobileNetV2 from layer 100 onward (Adam, lr 1e-6).
4. **Callbacks**: `ModelCheckpoint` keeps the model with the lowest validation loss across both phases. `EarlyStopping` (patience 5) stops a phase once validation loss stops improving.

## Usage

Install the dependencies:

```bash
pip install tensorflow matplotlib
```

Open `impl/Hymenoptera_Classification.ipynb` and run all cells from the `impl/` folder. The best model is saved to `result/model/best_model.keras`.

### Inference

```python
import tensorflow as tf

model = tf.keras.models.load_model("../result/model/best_model.keras")
img = tf.keras.utils.load_img("path/to/image.jpg", target_size=(160, 160))
prob = float(model.predict(tf.expand_dims(tf.keras.utils.img_to_array(img), 0))[0][0])
print("bees" if prob >= 0.5 else "ants", f"({max(prob, 1 - prob):.1%})")
```

The model rescales pixels itself, so it takes raw RGB images (0–255) at 160×160.
