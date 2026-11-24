# Car Damage Classifier

A binary image classifier that decides whether a car in a photo is **damaged** or **whole** (undamaged).

## Dataset

- Images are read from a `data/` folder with this layout (the same layout the notebook expects):
  - `data/training/00-damage`, `data/training/01-whole`
  - `data/validation/00-damage`, `data/validation/01-whole`
- The repository ships the images as `data.rar`. Extract it so that the `data/` folder sits next to the notebook.
- The test cells load images from a Kaggle path named `car-damage-detection`, which suggests the data was taken from a public Kaggle dataset with that name. The original author is not recorded in the code.
- Images are rescaled to 0-1 and resized to 150 x 150, batch size 20. No data augmentation is used.

## Model

Transfer learning with Keras / TensorFlow:

- `DenseNet121` pretrained on ImageNet (without its top layers) as the base
- `Flatten` -> `Dense(512, relu)` -> `Dense(1, sigmoid)`
- Loss: binary cross-entropy, optimizer: RMSprop (learning rate 1e-4)
- Metrics tracked: accuracy, precision, recall
- Trained for 15 epochs; the model is saved as `classifier.h5`

## Results

The notebook plots training and validation curves, and has two single-image prediction examples ("The car is damaged" / "The car is not damaged"). **The saved notebook has no recorded output, so no accuracy numbers are reported here.** Run the notebook to get them.

## How to run

1. Extract `data.rar` so the `data/` folder is next to the notebook.
2. Install the requirements: `pip install tensorflow numpy pandas matplotlib opencv-python`
3. Open `car-damage-classifier.ipynb` and run the cells in order.
4. The prediction cells at the end use paths from the original Kaggle environment. Change them to your own image paths.

## Files

- `car-damage-classifier.ipynb`: the full code
- `data.rar`: the dataset
