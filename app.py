import streamlit as st
from PIL import Image
import numpy as np
from imageCompression import load_and_resize_image, compress_image
import matplotlib.pyplot as plt
from io import BytesIO
import requests

def main():
    st.title("Image Compression App")

    upload_option = st.radio("Select the upload method:", ("Upload from Your Files", "Upload from URL"))

    image = handle_upload(upload_option)

    if image is not None:
        np_img = np.array(image)
        resized_img = load_and_resize_image(np_img)

        col1, col2 = st.columns(2)
        with col1:
            bit_color = st.selectbox("Select the bit color for compression:", (2, 4, 8, 16, 24, 32, 64))
        with col2:
            output_format = st.selectbox("Select a format for Output Image", ("png", "jpg"))

        if st.columns(3)[1].button("Compress Image", key="compress", use_container_width=True):
            try:
                progress_bar = st.progress(0)
                compressed_img = compress_image(resized_img, bit_color, progress_bar)
                progress_bar.progress(100)
                display_compressed_image(compressed_img)
                provide_download_link(compressed_img, output_format)
            except Exception as e:
                st.error(f"Error compressing image: {str(e)}")
                # You can log the error for further analysis if needed

def handle_upload(upload_option):
    if upload_option == "Upload from Your Files":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            return Image.open(uploaded_file)
    elif upload_option == "Upload from URL":
        url_input = st.text_input("Enter an image URL")
        if url_input:
            return load_image_from_url(url_input)
    return None

def load_image_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            st.error("Error loading image from URL: HTTP Status Code " + str(response.status_code))
            return None
    except Exception as e:
        st.error("Error loading image from URL: " + str(e))
        return None

def display_compressed_image(compressed_img):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(compressed_img)
    ax.set_title("Compressed Image")
    ax.axis("off")
    st.pyplot(fig)

def provide_download_link(compressed_img, output_format):
    try:
        compressed_img_bytes = BytesIO()
        plt.imsave(compressed_img_bytes, compressed_img, format=output_format)
        compressed_img_bytes.seek(0)
        st.columns(3)[1].download_button(label="Download Image", data=compressed_img_bytes, file_name=f'compressed_image.{output_format}', mime=f'image/{output_format}', use_container_width=True)
    except Exception as e:
        st.error(f"Error creating download link: {str(e)}")

if __name__ == "__main__":
    main()
