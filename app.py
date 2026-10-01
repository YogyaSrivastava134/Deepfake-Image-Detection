
import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image


# =========================
# Configuration
# =========================

MODEL_PATH = "model/best_efficientnet_b0.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# Image Preprocessing
# =========================

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# Load Model
# =========================

@st.cache_resource
def load_model():

    model = efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model


model = load_model()


# =========================
# Prediction
# =========================

def predict_image(image):

    image = image.convert("RGB")

    image_tensor = inference_transform(image)

    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        if device.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                outputs = model(image_tensor)

        else:
            outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    real_probability = probabilities[0, 0].item()
    fake_probability = probabilities[0, 1].item()

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    prediction = (
        "Fake"
        if predicted_class == 1
        else "Real"
    )

    confidence = (
        fake_probability
        if prediction == "Fake"
        else real_probability
    )

    return (
        prediction,
        confidence,
        real_probability,
        fake_probability
    )


# =========================
# Streamlit Interface
# =========================

st.title("Deepfake Image Detector")

st.write(
    "Upload a face image to classify it as "
    "Real or Fake."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Analyze Image"):

        prediction, confidence, real_probability, fake_probability = predict_image(
            image
        )

        st.subheader(
            f"Prediction: {prediction.upper()}"
        )

        st.write(
            f"Confidence: {confidence:.2%}"
        )

        st.write(
            f"Real Probability: {real_probability:.2%}"
        )

        st.write(
            f"Fake Probability: {fake_probability:.2%}"
        )
