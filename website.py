import streamlit as st
from PIL import Image
import sys
import pickle
import numpy as np
from skimage import measure
import numpy as np
from datetime import datetime, timedelta

import web_utils

# Only after all functions are defined, add the module path and import
sys.path.append('dataloader')
from main_dataset import MainDataset
# Use Streamlit's caching to load and prepare all data only once
@st.cache_data
def load_and_prepare_dataset():
    dataset_path = "new_small_main_dataset.pkl"
    print(f"Loading dataset '{dataset_path}' and preparing mappings (this should only appear once)")
    
    time = datetime.now()
    # Load dataset
    with open(dataset_path, 'rb') as f:
        main_dataset = pickle.load(f)
    elapsed = (datetime.now() - time).total_seconds()
    print(f"Dataset loaded in {elapsed} seconds")
    
    print("Loading mappings...")
    time = datetime.now()
    # Prepare all mappings
    dates = set()
    dates_mapping = {}
    dates_depths = {}
    depth_mapping = {}
    
    # Process dataset once to build all mappings
    for i, dat in enumerate(main_dataset):
        date = format_review_date(dat['item1']['review_date'])
        if date not in dates:
            dates_mapping[date] = i
            dates.add(date)
            dates_depths[date] = []
            depth_mapping[date] = {}

        zpos = int(dat['item1']['z_position'])
        depth_mapping[date][zpos] = i
        dates_depths[date].append(zpos)
    
    # Convert sets to sorted lists for UI
    dates_list = sorted(list(dates))
    
    # Create depths set
    depths_set = set()
    for i, dat in enumerate(main_dataset):
        depth = int(dat["item1"]['z_position'])
        if depth not in depths_set:
            depths_set.add(depth)
    depths_list = sorted(list(depths_set))
    
    elapsed = (datetime.now() - time).total_seconds()
    print(f"Mappings prepared in {elapsed} seconds")

    # Return all prepared data
    return {
        'dataset': main_dataset,
        'dates': dates_list,
        'dates_mapping': dates_mapping,
        'dates_depths': dates_depths,
        'depth_mapping': depth_mapping,
        'depths': depths_list
    }

def format_review_date(date_str):
    return date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:]

# Set page configuration
st.set_page_config(
    page_title="Model Visualization",
    page_icon=":crystal_ball:",
    initial_sidebar_state="collapsed"
    )

# Add custom CSS to change background color
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f8ff; /* Light blue background */
    }
    </style>
    """, unsafe_allow_html=True)



# Load and prepare all data once
if "data" not in st.session_state:
    print("Data is not cached...")
    data = load_and_prepare_dataset()
else:
    data = st.session_state.data
main_dataset = data['dataset']
dates = data['dates']
dates_depths = data['dates_depths']
depth_mapping = data['depth_mapping']

# Set initial values
curr_date = dates[0]
st.session_state.data = data

# Create UI
st.markdown("### Demonstration")

# Use column layout for controls to save space
control_cols = st.columns(2)
with control_cols[0]:
    # Create zeroth date which will be date one day before the first one
    zeroth_date = datetime.strptime(curr_date, '%Y-%m-%d') - timedelta(days=1)
    zeroth_date = zeroth_date.strftime('%Y-%m-%d')
    selected_date = st.select_slider("Select Date", options=[zeroth_date] + dates, value=curr_date)

# Only show depths that are available for the selected date
with control_cols[1]:
    if selected_date in dates_depths:
        available_depths = sorted(dates_depths[selected_date])
        curr_depth = available_depths[0] if available_depths else 0
        select_depth = st.select_slider("Select Depth", options=available_depths, value=curr_depth)
    else:
        st.write("No depths available for this date")
        select_depth = 0

# Get data index if available
dat_index = 0
if selected_date in dates_depths and select_depth in depth_mapping.get(selected_date, {}):
    dat_index = depth_mapping[selected_date][select_depth]

# Create a single blank image to be reused

def draw_contours(img, contours_mask):
    ptv = contours_mask[2]
    contours = measure.find_contours(ptv, 0.5)
    mask = np.zeros_like(ptv, dtype=bool)
    for contour in contours:
        contour = np.round(contour).astype(int)
        mask[contour[:, 0], contour[:, 1]] = 1
    ptv = mask
    ctv = contours_mask[1]
    contours = measure.find_contours(ctv, 0.5)
    mask = np.zeros_like(ctv, dtype=bool)
    for contour in contours:
        contour = np.round(contour).astype(int)
        mask[contour[:, 0], contour[:, 1]] = 1
    ctv = mask
    gtv = contours_mask[0]
    contours = measure.find_contours(gtv, 0.5)
    mask = np.zeros_like(gtv, dtype=bool)
    for contour in contours:
        contour = np.round(contour).astype(int)
        mask[contour[:, 0], contour[:, 1]] = 1
    gtv = mask
    img = np.stack([img] * 3, axis=-1)
    # display image
    vis_img = img.copy()
    # overwrite areas where gtv is 1
    # PTV is red
    vis_img[ptv == 1] = [img.max(), 0, 0]
    # CTV is orange
    vis_img[ctv == 1] = [img.max(), img.max() * 0.65, 0]
    # GTV is pink
    vis_img[gtv == 1] = [img.max(), int(img.max() * 0.75), int(img.max() * 0.8)]

    normalized = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min())

    return normalized

bottom_cols = st.columns(1)
with bottom_cols[0]:
    date = main_dataset[dat_index]["item1"]["review_date"]
    st.write(f"**Previous ({format_review_date(date)})**")
    img = main_dataset[dat_index]["item1"]["ct"].numpy().squeeze()
    contours_mask = main_dataset[dat_index]["item1"]["masks"].numpy().squeeze()
    vis_img = draw_contours(img=img, contours_mask=contours_mask)
    
    # Create three columns and use the middle one for the image
    left_col, mid_col, right_col = st.columns([1, 2, 1])
    with mid_col:
        st.image(vis_img, use_container_width=True, width=300)

top_cols = st.columns(2)
with top_cols[0]:
    date = main_dataset[dat_index]["item2"]["review_date"]
    st.write(f"**Now ({format_review_date(date)})**")
    img=main_dataset[dat_index]["item2"]["ct"].numpy().squeeze()
    contours_mask=main_dataset[dat_index]["item2"]["masks"].numpy().squeeze()
    vis_img = draw_contours(img=img, contours_mask=contours_mask)
    st.image(vis_img, use_container_width=False, width=300)


# Create layout
with top_cols[1]:
    date = main_dataset[dat_index]["item1"]["review_date"]
    st.write(f"**Transformation of ({format_review_date(date)})**")
    ct_img1=main_dataset[dat_index]["item1"]["ct"].numpy().squeeze()
    ct_img2=main_dataset[dat_index]["item2"]["ct"].numpy().squeeze()
    masks = main_dataset[dat_index]["item1"]["masks"].numpy().squeeze()
    masks2 = main_dataset[dat_index]["item2"]["masks"].numpy().squeeze()
    z_pos = main_dataset[dat_index]["item1"]["z_position"]

    deformed_planned_ct, transformed_masks = web_utils.transform_ct(
    ct_img1, masks, ct_img2, masks2, z_pos, plot=False)
    vis_img = draw_contours(
        img=deformed_planned_ct,
        contours_mask=transformed_masks)

    st.image(vis_img, use_container_width=False, width=300)


st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Verdict</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Stop the procedure.</h3>", unsafe_allow_html=True)

# Display selected values for debugging
st.sidebar.write(f"Selected date: {selected_date}")
st.sidebar.write(f"Selected depth: {select_depth}")