import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import io
import sys
import pickle
import time

# Use Streamlit's caching to load the dataset only once
@st.cache_data
def load_dataset():
    print("Loading dataset (this should only appear once)")
    with open('main_dataset.pkl', 'rb') as f:
        return pickle.load(f)

# Cache the placeholder image creation
@st.cache_data
def create_placeholder_image(width, height, text, color="lightgray"):
    print(f"Creating image for: {text} (should appear once per unique text)")
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.set_facecolor(color)
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12)
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.tight_layout(pad=0)
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)  # Important: close the figure to free memory
    return img

# Set page configuration
st.set_page_config(page_title="Model Visualization", initial_sidebar_state="collapsed", menu_items=None)

# Add custom CSS to change background color
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f8ff; /* Light blue background */
    }
    </style>
    """, unsafe_allow_html=True)

# Only after all functions are defined, add the module path and import
sys.path.append('dataloader')

# Use a try-except block for imports
try:
    from main_dataset import MainDataset
except ImportError as e:
    st.error(f"Error importing MainDataset: {e}")

# Preload these common UI elements
SAMPLE_PATH = "dataloader/data/full/"


# Load the dataset using the cached function (once)
if 'dataset' not in st.session_state:
    start_time = time.time()
    st.session_state.dataset = load_dataset()
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")

main_dataset = st.session_state.dataset
def format_review_date(date_str):
    return date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:]

dates = []
dates_mapping = {}
for i, dat in enumerate(main_dataset):
    dates.append(format_review_date(dat['item1']['review_date']))

print("wtfff")
print(dates)
st.write(dates)
# Update depths
depths_set = {}
depth_mapping = {}
for i, dat in enumerate(main_dataset):
    depth = format_review_date(dat['z-position'])
    if depth not in depths_set:
        depths_set.add(depth)
        depth_mapping[depth] = i
depths = list(depths_set)

# Create UI
st.markdown("### Demonstration")

# Use column layout for controls to save space
control_cols = st.columns(2)
with control_cols[0]:
    selected_date = st.select_slider("Select Date", options=dates, value=main_dataset[0]['item1']['review_date'])
with control_cols[1]:
    base_depth = format_review_date(main_dataset[0]['z-position'])
    select_depth = st.select_slider("Select Depth", options=depths, value=base_depth)

# Display data based on selections
# Pre-cache common images  
previous_img = create_placeholder_image(256, 256, f"Previous")
now_img = create_placeholder_image(256, 256, f"Now")
overlaid_img = create_placeholder_image(256, 256, f"Overlaid")
prediction_img = create_placeholder_image(256, 256, f"Prediction")

# Create layout
top_cols = st.columns(2)
with top_cols[0]:
    st.write(f"**Previous**")
    st.image(previous_img, use_container_width=False, width=300)
with top_cols[1]:
    st.write(f"**Now**")
    st.image(now_img, use_container_width=False, width=300)

bottom_cols = st.columns(2)
with bottom_cols[0]:
    st.write(f"**Overlaid**")
    st.image(overlaid_img, use_container_width=False, width=300)
with bottom_cols[1]:
    st.write(f"**Prediction**")
    st.image(prediction_img, use_container_width=False, width=300)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Verdict</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Stop the procedure.</h3>", unsafe_allow_html=True)

# Display selected values for debugging
st.sidebar.write(f"Selected date: {selected_date}")
st.sidebar.write(f"Selected depth: {select_depth}")