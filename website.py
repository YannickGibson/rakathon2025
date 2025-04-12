import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

from dataloader.main_dataset import MainDataset

# Set page configuration
st.set_page_config( page_title="Model Visualization", initial_sidebar_state="collapsed", menu_items=None)

# Add custom CSS to change background color
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f8ff;  /* Light blue background */
    }
    </style>
    """, unsafe_allow_html=True)

# Function to create a placeholder image
def create_placeholder_image(width, height, text, color="lightgray"):
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.set_facecolor(color)
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12)
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.tight_layout(pad=0)
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    return img

dates = ["2023-05-06", "2024-01-11", "2025-02-14"]
depths = [10, 13, 16, 19, 22]

SAMPLE_PATH = "dataloader/data/full/"

main_dataset = MainDataset(
        DATASET_PATH=SAMPLE_PATH,
        limit_samples=1, limit_rs_pairs=1,
        mock=True, verbose=True)

# Add a slider in the parameter control section
st.markdown("### Demonstration")
selected_date = st.select_slider("Select Date", options=dates)
select_depth = st.select_slider("Select Depth", options=depths)

# Create a row for the 5 small images at the top
top_cols = st.columns(2)
with top_cols[0]:
    st.write(f"**Previous**")
    placeholder_img = create_placeholder_image(256, 256, f"Previous")
    st.image(placeholder_img, use_container_width=False, width=300)
with top_cols[1]:
    st.write(f"**Now**")
    placeholder_img = create_placeholder_image(256, 256, f"Now")
    st.image(placeholder_img, use_container_width=False, width=300)


# Create a row for the two large images below
bottom_cols = st.columns(2)
with bottom_cols[0]:
    st.write(f"**Overlaid**")
    placeholder_img = create_placeholder_image(256, 256, f"Overlaid")
    st.image(placeholder_img, use_container_width=False, width=300)

# Add placeholder for ground truth (right)
with bottom_cols[1]:
    st.write(f"**Prediction**")
    placeholder_img = create_placeholder_image(256, 256, f"Prediction")
    st.image(placeholder_img, use_container_width=False, width=300)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Verdict</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Stop the procedure.</h3>", unsafe_allow_html=True)