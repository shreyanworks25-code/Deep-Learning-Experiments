
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ==========================================
# CIFAR-10 CLASS NAMES
# ==========================================

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Deep Learning Experiments",
    page_icon="🧠",
    layout="centered"
)


# ==========================================
# TITLE
# ==========================================

st.title("🧠 Deep Learning Experiments")

st.write(
    "Interactive demonstration of deep learning models "
    "developed using TensorFlow and Keras."
)


# ==========================================
# MODEL SELECTION
# ==========================================

st.sidebar.header("Model Selection")

model_name = st.sidebar.selectbox(
    "Choose a model",
    [
        "CNN",
        "Transfer Learning"
    ]
)


if model_name == "CNN":
    model_path = "final_cnn_model.keras"
else:
    model_path = "transfer_learning_model.keras"


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)


try:
    model = load_model(model_path)

    st.success(
        f"{model_name} model loaded successfully!"
    )

except Exception as e:

    st.error("Could not load the model.")

    st.code(str(e))

    st.stop()


# ==========================================
# GET MODEL INPUT SIZE
# ==========================================

try:

    input_shape = model.input_shape

    img_height = input_shape[1]
    img_width = input_shape[2]

    st.info(
        f"Model input size: "
        f"{img_width} × {img_height}"
    )

except Exception as e:

    st.error(
        "Could not determine the model input size."
    )

    st.code(str(e))

    st.stop()


# ==========================================
# IMAGE UPLOAD
# ==========================================

st.subheader("Upload an Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Display original image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    # ======================================
    # PREDICTION
    # ======================================

    if st.button("Predict"):

        # Resize image according to model
        img = image.resize(
            (img_width, img_height)
        )


        # Convert image to NumPy array
        img_array = np.array(
            img,
            dtype=np.float32
        )


        # Normalize pixel values
        img_array = img_array / 255.0


        # Add batch dimension
        #
        # Before:
        # (height, width, 3)
        #
        # After:
        # (1, height, width, 3)

        img_array = np.expand_dims(
            img_array,
            axis=0
        )


        # ==================================
        # MODEL PREDICTION
        # ==================================

        prediction = model.predict(
            img_array,
            verbose=0
        )


        # ==================================
        # GET PREDICTED CLASS
        # ==================================

        predicted_class = np.argmax(
            prediction[0]
        )

        confidence = np.max(
            prediction[0]
        )


        # ==================================
        # DISPLAY RESULT
        # ==================================

        st.subheader("Prediction")

        st.success(
            f"Predicted Class: "
            f"{class_names[predicted_class].upper()}"
        )

        st.write(
            f"Confidence: {confidence:.2%}"
        )


        # ==================================
        # DISPLAY ALL CLASS PROBABILITIES
        # ==================================

        st.subheader(
            "Class Probabilities"
        )

        for i, probability in enumerate(
            prediction[0]
        ):

            st.write(
                f"{class_names[i]}: "
                f"{probability:.2%}"
            )


        # ==================================
        # INFORMATION
        # ==================================

        st.info(
            "The prediction depends on the "
            "model architecture, training data, "
            "class order, and preprocessing used "
            "during model training."
        )


# ==========================================
# ABOUT PROJECT
# ==========================================

st.divider()

st.subheader("About This Project")

st.write(
    """
This project explores fundamental deep learning
concepts including:

- Artificial Neural Networks (ANN)
- Convolutional Neural Networks (CNN)
- Regularization
- Data Augmentation
- Transfer Learning
- TensorFlow and Keras
- CIFAR-10 Image Classification
"""
)

