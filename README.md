# 🐟 FishVision AI — Fish Classification using Deep Learning

## 📌 Project Overview
This project uses transfer learning models (MobileNetV2, EfficientNetB0, VGG16) to classify fish species from images.  
It is deployed as a **Streamlit web app** allowing users to upload a fish image and get:
- The predicted species
- Confidence scores
- A short description of the fish

## 📊 Dataset
- **Source:** [https://drive.google.com/drive/folders/1iKdOs4slf3XvNWkeSfsszhPRggfJ2qEd?usp=sharing]
- **Classes:** 11 fish categories
- **Preprocessing:**
  - Image resizing to 224×224
  - Normalization (pixel values scaled between 0 and 1)
  - Data augmentation (rotation, flipping, zooming)

## 🧠 Models Used
| Model         | Accuracy | Remarks |
|---------------|----------|---------|
| **MobileNetV2** |  99%    | Lightweight, fastest inference time |
| **EfficientNetB0** | 09%      | underperformed significantly in this dataset and are not recommended for use |
| **VGG16**       | 99%      | High accuracy but slower inference |
|**CNN Classification**|   90%     |  performed reasonably but lacked the accuracy of transfer learning models |
|**Resnet50**   |      29%      | underperformed significantly in this dataset and are not recommended for use |
|**InceptionV3**|    99%         | also performed very well but is larger in size, which could slow down inference slightly in resource-constrained environments | 
## ⚙️ Methodology
1. **Data Preprocessing**
   - Train-validation split
   - ImageDataGenerator for augmentation
2. **Model Training**
   - Transfer learning with pretrained ImageNet weights
   - Fine-tuning last layers
   - Early stopping & learning rate scheduling
3. **Evaluation**
   - Accuracy, confusion matrix, classification report
4. **Deployment**
   - Streamlit app with fish descriptions

## 📈 Evaluation
- **Accuracy Graphs**
- **Loss Graphs**
- **Confusion Matrices**
- **Precision, Recall, F1-scores**

## 🚀 Deployment Instructions
1. **Clone Repository**
```bash
git clone https://github.com/username/FishVision_AI.git
cd FishVision_AI
