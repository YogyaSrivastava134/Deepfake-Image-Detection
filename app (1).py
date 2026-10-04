
import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Deepfake Image Detector",
    page_icon="🔍",
    layout="centered"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #777777;
        margin-bottom: 30px;
    }

    .prediction {
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin-top: 20px;
    }

    .confidence {
        text-align: center;
        font-size: 20px;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🔍 Deepfake Image Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">EfficientNet-B0 based deepfake image classification</div>',
    unsafe_allow_html=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_REPO = "Yogya134/deepfake-efficientnet-b0"
MODEL_FILENAME = "best_efficientnet_b0.pth"


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    # Download model from Hugging Face
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME
    )

    # Recreate EfficientNet-B0 architecture
    model = efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2
    )

    # Load trained checkpoint
    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


model = load_model()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# IMAGE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image to analyze",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.write("")

    if st.button(
        "🔍 Analyze Image",
        use_container_width=True
    ):

        # Preprocess
        image_tensor = eval_transform(
            image
        ).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():

            outputs = model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[0]

        # Class probabilities
        real_probability = probabilities[0].item()
        fake_probability = probabilities[1].item()

        # Prediction
        predicted_class = torch.argmax(
            probabilities
        ).item()

        if predicted_class == 0:

            prediction = "REAL"
            confidence = real_probability

        else:

            prediction = "FAKE"
            confidence = fake_probability


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.markdown(
            f'<div class="prediction">Prediction: {prediction}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence">Confidence: {confidence * 100:.2f}%</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # PROBABILITY BREAKDOWN
        # ====================================================

        st.write("")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Real Probability",
                f"{real_probability * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Fake Probability",
                f"{fake_probability * 100:.2f}%"
            )


        # ====================================================
        # PROBABILITY BAR
        # ====================================================

        st.write("### Prediction Probability")

        st.progress(
            fake_probability,
            text=f"Fake probability: {fake_probability * 100:.2f}%"
        )


else:

    st.info(
        "Upload a JPG, JPEG, or PNG image to begin."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("ℹ️ About this model"):

    st.write(
        """
        This application uses a fine-tuned EfficientNet-B0
        convolutional neural network for binary classification
        of real and deepfake images.

        **Model:** EfficientNet-B0

        **Input Size:** 224 × 224

        **Classes:** Real / Fake

        **Preprocessing:** ImageNet normalization

        **Test Accuracy:** 97.95%

        **Test ROC-AUC:** 99.79%
        """
    )
