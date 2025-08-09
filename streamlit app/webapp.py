import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------
# 1️⃣ Page config
# -----------------------------------
st.set_page_config(page_title="🐟 Fish Classifier", layout="centered")

# -----------------------------------
# 2️⃣ Load the trained model (cached)
# -----------------------------------
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model("mobilenet_best_model.h5")  # Change to your model

model = load_model()

# -----------------------------------
# 3️⃣ Class labels
# -----------------------------------
CLASS_LABELS = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

# Custom descriptions for each class
CLASS_DESCRIPTIONS = {
    'animal fish': "A general fish category. Could be any aquatic vertebrate.",
    'animal fish bass': "Bass is a common name for various species of fish known for their firm texture and mild flavor.",
    'fish sea_food black_sea_sprat': "Black Sea sprat is a small pelagic fish found in the Black Sea, often eaten grilled or smoked.",
    'fish sea_food gilt_head_bream': "Gilt-head bream is a prized Mediterranean fish with a delicate taste.",
    'fish sea_food hourse_mackerel': "Horse mackerel is a fast swimmer, known for its oily flesh and rich flavor.",
    'fish sea_food red_mullet': "Red mullet has a sweet, delicate flavor, popular in Mediterranean cuisine.",
    'fish sea_food red_sea_bream': "Red sea bream is a premium fish in Japanese cuisine, valued for sushi and sashimi.",
    'fish sea_food sea_bass': "Sea bass is a mild-flavored fish popular in fine dining.",
    'fish sea_food shrimp': "Shrimp are small crustaceans enjoyed worldwide, versatile in many dishes.",
    'fish sea_food striped_red_mullet': "Striped red mullet is known for its tender texture and subtle flavor.",
    'fish sea_food trout': "Trout is a freshwater fish known for its delicate taste and high omega-3 content."
}

IMG_SIZE = (224, 224)

# -----------------------------------
# 4️⃣ Prediction Function
# -----------------------------------
def predict_fish(uploaded_image):
    img = Image.open(uploaded_image).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]
    top_index = np.argmax(predictions)
    predicted_label = CLASS_LABELS[top_index]
    confidence_scores = dict(zip(CLASS_LABELS, predictions * 100))
    return predicted_label, confidence_scores

# -----------------------------------
# 5️⃣ Streamlit UI
# -----------------------------------
st.title("🐟 Fish Image Classification App")
st.write("Upload an image of a fish and get the predicted category, confidence scores, and extra details.")

uploaded_file = st.file_uploader("Upload a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=300)
    with st.spinner("Predicting..."):
        predicted_label, confidence_scores = predict_fish(uploaded_file)

    st.subheader(f"✅ Predicted Category: {predicted_label}")
    st.write(f"**ℹ️ About this fish:** {CLASS_DESCRIPTIONS.get(predicted_label, 'No description available.')}")
    
    st.write("### 📊 Top 3 Confidence Scores:")
    for label, score in sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
        st.write(f"{label}: **{score:.2f}%**")
else:
    st.info("Please upload an image file to classify.")
