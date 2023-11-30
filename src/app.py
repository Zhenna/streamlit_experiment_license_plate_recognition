import streamlit as st
from main import run_license_plate_recognition
import os


def app():
    st.header("License Plate Recognition Web App")
    st.subheader("Powered by YOLOv5")
    st.write("Welcome!")

    with st.form("my_uploader"):
        uploaded_file = st.file_uploader(
            "Upload image", type=["png", "jpg", "jpeg"], accept_multiple_files=False
        )
        submit = st.form_submit_button(label="Upload")

    if uploaded_file is not None:
        # save uploaded image
        save_path = os.path.join("temp", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    if submit:
        # add spinner
        with st.spinner(text="Detecting license plate ..."):
            # display license plate as text
            text = run_license_plate_recognition(save_path).recognize_text()
            st.write(f"Detected License Plate Number: {text}")
            # show uploaded image with bounding box
            best_bb = run_license_plate_recognition(save_path).showBestPrediction()
            st.image(best_bb)


if __name__ == "__main__":
    app()
