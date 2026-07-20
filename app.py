import streamlit as st

from config import APP_TITLE, APP_ICON, MODEL_INFO
from model import load_model
from preprocess import preprocess_image
from predict import predict_image


# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide"
)

# -------------------------------------------------------
# Custom Styling
# -------------------------------------------------------
st.markdown("""
<style>

.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#1565C0;
}

.sub-title{
    text-align:center;
    font-size:18px;
    color:gray;
    margin-bottom:30px;
}

.result-card{
    padding:20px;
    border-radius:12px;
    background-color:#f5f5f5;
    border-left:8px solid #1565C0;
}

.footer{
    font-size:14px;
    color:gray;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
st.sidebar.title("🧠 Model Information")

for key, value in MODEL_INFO.items():
    st.sidebar.write(f"**{key}:** {value}")

st.sidebar.markdown("---")

st.sidebar.info(
    """
This application predicts whether an uploaded MRI image is:

• Non Demented

or

• Very Mild Demented

using a CNN model trained with TensorFlow.
"""
)

# -------------------------------------------------------
# Title
# -------------------------------------------------------
st.markdown(
    f"<h1 class='main-title'>{APP_ICON} {APP_TITLE}</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='sub-title'>Upload an MRI brain image for prediction.</p>",
    unsafe_allow_html=True
)

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------
model = load_model()

if model is None:
    st.stop()

# -------------------------------------------------------
# Upload Section
# -------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose an MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image_array, display_image = preprocess_image(uploaded_file)

    col1, col2 = st.columns(2)

    # ---------------- Left Column ----------------

    with col1:

        st.subheader("Uploaded MRI Image")

        st.image(
            display_image,
            use_container_width=True
        )

    # ---------------- Right Column ----------------

    with col2:

        with st.spinner("Analyzing image..."):

            result = predict_image(model, image_array)

        st.subheader("Prediction")

        prediction = result["prediction"]
        confidence = result["confidence"]
        probability = result["probability"]

        if prediction == "Non Demented":
            st.success(f"Prediction: {prediction}")
        else:
            st.warning(f"Prediction: {prediction}")

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        st.write("Model Probability")

        st.progress(float(confidence))

        st.write(
            f"Raw Sigmoid Output: **{probability:.4f}**"
        )

        st.markdown("---")

        st.subheader("Interpretation")

        if prediction == "Non Demented":

            st.success(
                """
The uploaded MRI image was classified as
**Non Demented** by the CNN model.
"""
            )

        else:

            st.warning(
                """
The uploaded MRI image was classified as
**Very Mild Demented** by the CNN model.
"""
            )

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown("---")

st.warning(
"""
### Medical Disclaimer

This application is intended **only for educational and research purposes**.

It is **not a medical diagnostic tool** and should **not** be used as a substitute for professional clinical assessment or medical advice.

Always consult a qualified healthcare professional for diagnosis and treatment decisions.
"""
)

st.markdown(
    "<p class='footer'>Developed using TensorFlow • Streamlit</p>",
    unsafe_allow_html=True
)
