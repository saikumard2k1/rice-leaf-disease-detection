# Rice Leaf Disease Detection using Transfer Learning (CNN)

A deep learning project that classifies rice leaf images into three major diseases: **Bacterial Leaf Blight**, **Brown Spot** and **Leaf Smut**. It uses MobileNetV2 transfer learning with data augmentation, is compared against a CNN trained from scratch, and is deployed as a Streamlit web app.

## Problem Statement
Rice diseases reduce crop yield, and identifying them early matters. The goal is to build a model that classifies a rice leaf image into one of three diseases, to analyse techniques such as data augmentation, and to recommend the best model for production.

## Dataset
- 120 images of diseased rice leaves, 40 per class (119 after removing 1 duplicate)
- Classes: Bacterial Leaf Blight, Brown Spot, Leaf Smut
- Source: [PRCP-1001-RiceLeaf dataset](https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1001-RiceLeaf.zip)
- The dataset is not included in this repository. Download and extract it, then set `DATA_DIR` in the notebook.

## Project Workflow
1. Exploratory data analysis (class balance, image sizes, brightness, sample images)
2. Duplicate removal using file hashing
3. Stratified train, validation and test split (70/15/15)
4. Data augmentation on training data only
5. Model building: MobileNetV2 transfer learning and a baseline CNN
6. Evaluation with classification reports and confusion matrices
7. Augmentation experiment (with vs without)
8. Deployment as a Streamlit web app

## Key Observations from EDA
- **Balanced classes:** about 40 images per class, so no resampling or class weights were needed.
- **1 duplicate image** (Leaf smut) was found and removed to prevent leakage between train and test sets.
- **Lab-style images:** every photo has a plain black background with a centred leaf. Brightness is consistent across classes.
- **Visual similarity:** Brown Spot and Leaf Smut lesions look alike, so these two classes are the most likely to be confused.
- Images are resized to 224x224 for MobileNetV2.

## Data Split
| Set | Images |
|---|---|
| Train | 83 |
| Validation | 18 |
| Test | 18 |

The split is stratified, so validation and test have 6 images per class. The test set is never used for training or early stopping.

## Preprocessing and Augmentation
Augmentation is applied to **training data only**, so validation and test metrics reflect real, unmodified images.

| Technique | Setting | Reason |
|---|---|---|
| Rotation | ±20° | Leaves have no fixed orientation |
| Horizontal and vertical flip | on | A mirrored leaf is still a valid leaf |
| Zoom | 0.15 | Camera distance varies |
| Width and height shift | 0.1 | Small, so lesions are not cropped out |
| Brightness | 0.8 to 1.2 | Simulates sunlight and shadow |

Pixels are scaled with MobileNetV2's own `preprocess_input` (range -1 to 1), matching how the pre-trained weights were trained.

## Models
| Model | Description |
|---|---|
| **MobileNetV2** (transfer learning) | Frozen ImageNet base, then GlobalAveragePooling, Dense(64), Dropout(0.5), Softmax(3). About 82K trainable parameters |
| **Baseline CNN** | 3 Conv + MaxPool blocks, GlobalAveragePooling, Dense(64), Dropout(0.5), Softmax(3). Trained from scratch |

Training uses Adam, categorical crossentropy, EarlyStopping (patience 5, restores best weights) and ReduceLROnPlateau.

## Results
| Model | Train Acc | Val Acc | Test Acc | Test Macro F1 | Parameters |
|---|---|---|---|---|---|
| Baseline CNN | 0.602 | 0.556 | 0.333 | 0.297 | 101,699 |
| **MobileNetV2** | **0.940** | **0.833** | **0.889** | **0.889** | 2,340,163 (about 82K trainable) |

- MobileNetV2 classified **16 of 18** test images correctly, with F1 of 0.83 or higher for every class.
- Its errors were one Brown Spot predicted as Bacterial Leaf Blight, and one Leaf Smut predicted as Brown Spot.
- The baseline CNN performed at chance level (33%) and never identified Leaf Smut correctly. It **underfits**, because 83 images are too few to learn features from scratch.

**Recommendation for production: MobileNetV2.** It is far more accurate, balanced across all classes, and small enough for web or mobile deployment.

### Augmentation Experiment
| Setting | Val Acc | Test Acc |
|---|---|---|
| No augmentation | 0.944 | 0.944 |
| With augmentation | 0.889 | 0.889 |

The difference is a single image out of 18 (5.6%), so it is within noise and does not show that augmentation hurts. A likely reason it did not help is that all images are lab-style photos, so the test images closely resemble the training images. Augmentation is kept for robustness, since it should matter more for field photos with varied angles and lighting.

## Challenges Faced
| Challenge | Technique used | Reason |
|---|---|---|
| Very small dataset | Transfer learning | Reuses ImageNet features |
| Overfitting risk | Frozen base, Dropout, EarlyStopping | Fewer trainable weights, regularisation |
| Duplicate image | File hashing | Prevents train/test leakage |
| Unreliable evaluation | Stratified split with separate test set | Equal classes, unbiased test estimate |
| Augmentation affecting evaluation | Augment training data only | Validation and test must reflect real images |
| Pre-trained input scaling | `preprocess_input` instead of `rescale=1/255` | Matches MobileNetV2 training |

## Limitations
- Only 18 test images, so one wrong prediction changes accuracy by 5.6% and the estimate is uncertain.
- All images have plain black backgrounds, so performance on real field photos is untested.
- No fine-tuning or hyperparameter search was done.

## Future Work
- Collect field images with varied backgrounds and lighting
- Use cross-validation for a more reliable estimate
- Fine-tune the top MobileNetV2 layers and tune dropout, dense units and learning rate
- Add Grad-CAM to show which leaf regions drive each prediction
- Deploy on mobile with TensorFlow Lite

## Web App
A Streamlit app lets a user upload a leaf image and see the predicted disease with confidence scores for all three classes.

<img width="1916" height="902" alt="image" src="https://github.com/user-attachments/assets/8f247e05-e20b-4e80-a910-58e87cd90a4a" />
<img width="1917" height="907" alt="image" src="https://github.com/user-attachments/assets/15837d7f-c64a-4a2f-b274-181e52ccd77d" />


### Run the app
```
pip install -r requirements.txt
python -m streamlit run app.py
```
Tested with Python 3.14.7.

## Repository Structure
```
├── Rice_Leaf_Disease_Prediction.ipynb   # EDA, training, evaluation, reports
├── app.py                               # Streamlit web app
├── rice_leaf_model.keras                # Trained MobileNetV2 model
├── class_names.json                     # Class index mapping
├── requirements.txt
└── README.md
```

## Tech Stack
Python, TensorFlow/Keras, MobileNetV2, scikit-learn, pandas, NumPy, Matplotlib, Pillow, Streamlit

Live demo: https://rice-leaf-disease-detection-v72d.onrender.com/

> Hosted on a free instance, so it sleeps after 15 minutes of inactivity. The first load can take a few minutes to wake up.
