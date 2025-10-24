import streamlit as st
import os
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, label, zoom
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import ee # Google Earth Engine library
import requests
import rasterio
from datetime import datetime, timedelta
import cv2 # OpenCV for contour detection
import pandas as pd # For CSV logging

# For Email alerts
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage # For attaching images

# --- New Imports for Geocoding ---
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import pyproj # --- NEW IMPORT --- For coordinate transformation

# --- Streamlit Page Configuration (MUST be the first Streamlit command) ---
# IMPORTANT: This must be the very first Streamlit command in your script
# Important: This must be the very first Streamlit command in your script
st.set_page_config(
    page_title="EcoVision: Environmental Change Detection",
    layout="wide",
    initial_sidebar_state="expanded",
    # REMOVE theme parameters from here:
    # primaryColor="#4CAF50",
    # backgroundColor="#F0F2F6",
    # secondaryBackgroundColor="#FFFFFF",
    # textColor="#333333",
    # font="sans serif"
)
st.markdown("""
<div style="background: linear-gradient(to right, #2c3e50, #4CAF50); padding: 20px; border-radius: 10px; margin-bottom: 30px;">
    <h1 style="color: white; text-align: center; margin: 0;">🌳🌊 EcoVision</h1>
    <h3 style="color: white; text-align: center; margin: 0;">Smart Alarm System for Environmental Change</h3>
</div>
""", unsafe_allow_html=True)
# Custom CSS to override Streamlit's default message box colors and other elements

# --- END THEME CONFIGURATION ---
# --- Configuration ---
# Paths to your saved best models. ADJUST THESE PATHS!
DEFORESTATION_MODEL_PATH = '/Users/sukhineshgopalan/Desktop/EcoVision/models/unet_forest_segmentation_best_iou.h5'
COASTAL_MODEL_PATH = '/Users/sukhineshgopalan/Desktop/EcoVision/models/best_unet_model_coastal_erosion3.h5' # <--- YOUR COASTAL MODEL PATH

# Image dimensions for Deforestation model
DEF_IMG_HEIGHT = 128
DEF_IMG_WIDTH = 128
DEF_NUM_CHANNELS = 3 # RGB for deforestation model (ensure this matches your training if NIR was included)
DEF_NUM_CLASSES = 1 # Binary segmentation (forest/non-forest)

# Image dimensions for Coastal Erosion model
COASTAL_IMG_HEIGHT = 256
COASTAL_IMG_WIDTH = 256
COASTAL_NUM_CHANNELS = 4 # <--- IMPORTANT: This MUST be 4 as per your training script's input_img_shape (H, W, 4)
COASTAL_NUM_CLASSES = 1 # Binary segmentation (water/land or erosion area)

# Define a folder to store downloaded images (relative to app.py)
OUTPUT_BASE_FOLDER = '/Users/sukhineshgopalan/Desktop/EcoVision/data/downloaded_images'
ALERTS_LOG_FOLDER = os.path.join(OUTPUT_BASE_FOLDER, 'alerts_log') # New folder for logs

# Ensure alerts log folder exists
os.makedirs(ALERTS_LOG_FOLDER, exist_ok=True)

# --- Email Configuration (for Email alerts) ---
# Ensure these environment variables are set before running the app
# Use App Password for Gmail if you are using a Gmail sender account
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_EMAIL_PASSWORD = os.environ.get("SENDER_EMAIL_PASSWORD") # Use App Password for Gmail
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# --- New Constants for Area Calculation ---
PIXEL_RESOLUTION_METERS = 10 # Assuming Sentinel-2 imagery (10 meters per pixel)
SQUARE_METERS_PER_PIXEL = PIXEL_RESOLUTION_METERS * PIXEL_RESOLUTION_METERS
SQ_KM_PER_PIXEL = SQUARE_METERS_PER_PIXEL / 1_000_000

# --- U-Net Model Architecture Definition (For Deforestation - using segmentation_models.keras) ---
# Keep this if your deforestation model was indeed built with sm.Unet
import segmentation_models as sm
os.environ['SM_FRAMEWORK'] = 'tf.keras' # Ensure the correct Keras backend is set

def build_sm_unet_model(input_shape, backbone='resnet34', classes=1, activation='sigmoid'):
    """
    Builds a U-Net model using the segmentation_models.keras library.
    This is for the DEFORESTATION model.
    """
    model = sm.Unet(
        backbone_name=backbone,
        input_shape=input_shape,
        encoder_weights=None, # We are loading custom weights, not pre-trained ImageNet weights
        classes=classes,
        activation=activation
    )
    return model

# --- CUSTOM U-Net Model Architecture Definition (For Coastal Erosion) ---
# THIS IS THE UNET DEFINITION FROM YOUR TRAINING SCRIPT
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model

def build_custom_unet_model(input_size):
    """
    Defines the U-Net model architecture identical to your training script for coastal erosion.
    input_size: Tuple (height, width, channels) of input images.
    """
    inputs = Input(input_size)

    # Encoder (Downsampling Path)
    # Block 1
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Block 2
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Block 3
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Block 4
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    # Decoder (Upsampling Path) - with Skip Connections
    # Block 6
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5)
    up6 = concatenate([up6, conv4], axis=3) # Skip connection
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    # Block 7
    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
    up7 = concatenate([up7, conv3], axis=3) # Skip connection
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    # Block 8
    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
    up8 = concatenate([up8, conv2], axis=3) # Skip connection
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    # Block 9
    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
    up9 = concatenate([up9, conv1], axis=3) # Skip connection
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    # Output layer for binary segmentation: 1 channel, sigmoid activation (for probability map)
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# --- Geocoding Function ---
@st.cache_data(ttl=3600*24) # Cache for 24 hours to reduce API calls
def get_location_name(lat, lon):
    geolocator = Nominatim(user_agent="ecovision_app") # IMPORTANT: Replace "ecovision_app" with a unique name for your application
    try:
        # Ensure lat and lon are not None or non-numeric before passing to geopy
        if lat is None or lon is None or not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            raise ValueError("Latitude and Longitude must be valid numbers.")

        location = geolocator.reverse((lat, lon), exactly_one=True, language='en')
        if location:
            address = location.raw['address']
            # Prioritize more specific names
            if 'city' in address: return address['city']
            if 'town' in address: return address['town']
            if 'village' in address: return address['village']
            if 'county' in address: return address['county']
            if 'state' in address: return address['state']
            if 'country' in address: return address['country']
            return location.address # Fallback to full address
        return "Unknown Area"
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        st.warning(f"Geocoding service error: {e}. Cannot get area name.")
        return "Unknown Area (Geocoding Failed)"
    except ValueError as e: # Catch the custom ValueError for invalid coordinates
        st.warning(f"Invalid coordinates for geocoding: {e}")
        return "Unknown Area (Invalid Coords)"
    except Exception as e:
        st.warning(f"Error getting location name: {e}")
        return "Unknown Area (Error)"

# --- Earth Engine Initialization ---
@st.cache_resource
def initialize_earth_engine():
    try:
        ee.Initialize()
        return "success", "Google Earth Engine initialized successfully."
    except Exception as e:
        return "error", f"Error initializing Earth Engine: {e}. " \
                        "Please ensure you have authenticated and installed 'earthengine-api'. " \
                        "If running for the first time, click 'Authenticate GEE' in the sidebar."

# --- Load the trained models (weights only) ---
@st.cache_resource
def load_dl_models_weights():
    # Load Deforestation Model (assuming it uses segmentation_models.Unet)
    def_model_status_type = "error"
    def_model_obj = None
    def_model_msg = "Deforestation Model: Not Loaded"
    try:
        def_model_architecture = build_sm_unet_model( # Using the segmentation_models builder
            input_shape=(DEF_IMG_HEIGHT, DEF_IMG_WIDTH, DEF_NUM_CHANNELS),
            backbone='resnet34' # Assuming resnet34 for deforestation model
        )
        def_model_architecture.load_weights(DEFORESTATION_MODEL_PATH)
        def_model_status_type = "success"
        def_model_obj = def_model_architecture
        def_model_msg = "Deforestation Model loaded successfully!"
    except Exception as e:
        def_model_msg = f"Error loading Deforestation Model: {e}. Check path/architecture."

    # Load Coastal Erosion Model (using your custom U-Net architecture)
    coastal_model_status_type = "error"
    coastal_model_obj = None
    coastal_model_msg = "Coastal Erosion Model: Not Loaded"
    try:
        coastal_model_architecture = build_custom_unet_model( # Using your custom U-Net builder
            input_size=(COASTAL_IMG_HEIGHT, COASTAL_IMG_WIDTH, COASTAL_NUM_CHANNELS)
        )
        coastal_model_architecture.load_weights(COASTAL_MODEL_PATH)
        coastal_model_status_type = "success"
        coastal_model_obj = coastal_model_architecture
        coastal_model_msg = "Coastal Erosion Model loaded successfully!"
    except Exception as e:
        coastal_model_msg = f"Error loading Coastal Erosion Model: {e}. Check path/architecture."

    return (def_model_status_type, def_model_obj, def_model_msg,
            coastal_model_status_type, coastal_model_obj, coastal_model_msg)

# Initialize GEE and load models when the app starts
gee_status_type, gee_status_msg = initialize_earth_engine()
# Store loaded models in session state for easy access without passing as args
(def_model_status_type, st.session_state.def_model, def_model_msg,
 coastal_model_status_type, st.session_state.coastal_model, coastal_model_msg) = load_dl_models_weights()


# --- GEE Image Fetching Function ---
@st.cache_data(ttl=3600)
def fetch_and_save_image_from_gee(aoi_coords, target_date, days_buffer, output_folder, filename_prefix, required_bands):
    """
    Fetches a cloud-free Sentinel-2 image for a given AOI and a date range around the target_date,
    and saves it as a TIFF file. Returns the path or None.
    days_buffer: Number of days before/after the target_date to search for an image.
    """
    min_lon, min_lat, max_lon, max_lat = aoi_coords
    aoi_geometry = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    # Convert target_date to datetime object if it's a string
    if isinstance(target_date, str):
        target_date_dt = datetime.strptime(target_date, "%Y-%m-%d")
    else:
        target_date_dt = target_date

    # Define date range for image search
    date_start = (target_date_dt - timedelta(days=days_buffer)).strftime("%Y-%m-%d")
    date_end = (target_date_dt + timedelta(days=days_buffer)).strftime("%Y-%m-%d")

    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(date_start, date_end) \
        .filterBounds(aoi_geometry) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) # Using 10 as a general starting point

    image = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()

    if image is None:
        st.warning(f"No suitable image found for {filename_prefix} in {date_start} to {date_end} with <10% cloud cover. Consider widening date range or changing AOI.")
        return None

    image_to_export = image.select(required_bands) # Select the specified bands

    output_path = os.path.join(output_folder, f"{filename_prefix}.tif")
    os.makedirs(output_folder, exist_ok=True)

    download_url = image_to_export.getDownloadURL({
        'scale': PIXEL_RESOLUTION_METERS, # Use the global pixel resolution for Sentinel-2
        'region': aoi_geometry.getInfo()['coordinates'],
        'format': 'GEO_TIFF'
    })

    st.sidebar.info(f"Downloading image for {filename_prefix}...")
    try:
        response = requests.get(download_url)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            f.write(response.content)
        st.sidebar.success(f"Downloaded: {os.path.basename(output_path)}")
        return output_path
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading image for {filename_prefix}: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during image saving for {filename_prefix}: {e}")
        return None

# --- Preprocessing function (prepares image for DL model) ---
def preprocess_image_for_prediction(image_array, target_height, target_width, num_channels):
    """
    Preprocesses an image array to be suitable for model prediction.
    This function must replicate the preprocessing steps used during training.
    Uses scipy.ndimage.zoom for robust resizing of multi-channel images.
    """
    if len(image_array.shape) == 4: # Remove batch dimension if present
        image_array = image_array[0]

    # Handle channel selection/padding
    if image_array.shape[2] > num_channels:
        image_array = image_array[:, :, :num_channels]
    elif image_array.shape[2] < num_channels:
        if image_array.shape[2] == 3 and num_channels == 4:
            # If 3 channels (RGB) are provided but 4 (RGBN) are expected,
            # append a zero channel as a last resort, assuming NIR is missing.
            st.warning(f"Input image has {image_array.shape[2]} channels, but model expects {num_channels}. Appending a zero channel.")
            padding_channel = np.zeros((image_array.shape[0], image_array.shape[1], 1), dtype=image_array.dtype)
            image_array = np.concatenate([image_array, padding_channel], axis=-1)
        else:
            raise ValueError(f"Image has {image_array.shape[2]} channels, expected {num_channels}. Cannot convert automatically.")

    image_array = image_array.astype(np.float32)

    # Normalize image to 0-1 based on expected max value (e.g., 10000 for Sentinel-2, 255 for 8-bit)
    if image_array.max() > 255.0: # Assuming 16-bit TIFFs with max values like 10000
        image_array = image_array / 10000.0
    elif image_array.max() > 1.0: # Assuming 8-bit images (0-255)
        image_array = image_array / 255.0

    # Resize image if dimensions don't match model's expected input size using scipy.ndimage.zoom
    if image_array.shape[0] != target_height or image_array.shape[1] != target_width:
        zoom_factors = (target_height / image_array.shape[0],
                        target_width / image_array.shape[1],
                        1) # No scaling for channel dimension
        # Using order=1 for bilinear interpolation, mode='reflect' to handle borders
        image_array = zoom(image_array, zoom_factors, order=1, mode='reflect')

    preprocessed_image = np.expand_dims(image_array, axis=0) # Add batch dimension
    return preprocessed_image

# --- Prediction function (uses the loaded DL model) ---
def predict_mask(model_obj, image_path, target_height, target_width, num_channels):
    """
    Loads an image, preprocesses it, and predicts a binary segmentation mask.
    Takes the specific model object and its expected input dimensions.
    Returns the binary mask, georeferencing transform, CRS, original height/width, and the preprocessed image.
    """
    if model_obj is None:
        st.error("Deep Learning Model not loaded. Cannot make prediction.")
        return None, None, None, None, None, None # --- MODIFIED: Added None for CRS ---

    if not os.path.exists(image_path):
        st.error(f"Image file not found at {image_path}")
        return None, None, None, None, None, None # --- MODIFIED: Added None for CRS ---

    try:
        with rasterio.open(image_path) as src_img:
            # Read all bands and transpose to HWC
            image_raw = src_img.read().transpose((1, 2, 0))
            original_height, original_width = image_raw.shape[0], image_raw.shape[1]
            transform = src_img.transform
            crs = src_img.crs # --- NEW LINE: Extract CRS ---
        
        # Debugging: Check the transform right after reading
        st.sidebar.write(f"DEBUG: Transform for {os.path.basename(image_path)}: {transform}")
        st.sidebar.write(f"DEBUG: CRS for {os.path.basename(image_path)}: {crs}") # --- NEW DEBUG LINE ---


        preprocessed_image = preprocess_image_for_prediction(image_raw, target_height, target_width, num_channels)
        
        # Check if preprocessed_image is valid
        if preprocessed_image is None or preprocessed_image.size == 0:
            st.error(f"Preprocessing of {image_path} resulted in an empty or invalid image.")
            return None, None, None, None, None, None # --- MODIFIED: Added None for CRS ---

        raw_prediction = model_obj.predict(preprocessed_image)

        binary_mask = (raw_prediction.squeeze() > 0.5).astype(np.uint8)

        # Resize the binary mask back to the original image dimensions IF dimensions are valid
        if original_height is not None and original_width is not None and \
           (original_height != target_height or original_width != target_width):
            # Resize binary mask using nearest neighbor to preserve distinct values
            zoom_factors = (original_height / binary_mask.shape[0],
                            original_width / binary_mask.shape[1])
            # Use order=0 for nearest neighbor interpolation for binary masks
            binary_mask = zoom(binary_mask, zoom_factors, order=0, mode='nearest').astype(np.uint8)


        return binary_mask, transform, crs, original_height, original_width, preprocessed_image # --- MODIFIED: Return CRS ---

    except Exception as e:
        st.error(f"Error during prediction for {image_path}: {e}")
        st.sidebar.error(f"DEBUG: Failed to open or process {os.path.basename(image_path)} due to: {e}")
        return None, None, None, None, None, None # --- MODIFIED: Added None for CRS ---

# --- Helper function to get geographic coordinates from pixel coordinates ---
def get_coords_from_pixel(transform, crs, row, col): # --- MODIFIED: Added 'crs' parameter ---
    """
    Converts pixel (row, col) to geographic (lon, lat) in WGS84 using the rasterio transform and CRS.
    Returns (lon, lat) or (None, None) if transform/crs is invalid or conversion fails.
    """
    if transform is None or crs is None: # --- MODIFIED: Check for crs as well ---
        return None, None
    try:
        # Get the projected coordinates (x, y) from the pixel (col, row)
        x_proj, y_proj = transform * (col, row) # --- MODIFIED: Use transform to get projected x,y ---

        # Define the transformer to convert from image's CRS to WGS84 (EPSG:4326)
        transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True) # --- NEW ---
        lon_wgs84, lat_wgs84 = transformer.transform(x_proj, y_proj) # --- NEW ---
        
        # Check for invalid coordinates that might arise from projection errors
        if not (-180 <= lon_wgs84 <= 180 and -90 <= lat_wgs84 <= 90):
            st.warning(f"Transformed coordinates ({lon_wgs84:.4f}, {lat_wgs84:.4f}) are out of WGS84 bounds.")
            return None, None

        return lon_wgs84, lat_wgs84 # --- MODIFIED: Return WGS84 lon, lat ---
    except Exception as e:
        st.warning(f"Failed to get coords from pixel ({row}, {col}) with transform and CRS: {e}")
        return None, None

# --- Helper function to extract coastline contours ---
def extract_coastline_contour(binary_mask, min_contour_area=50):
    """
    Extracts the coastline contour from a binary mask.
    Assumes `binary_mask` is 0s and 1s.
    Returns a list of contours found by OpenCV.
    """
    mask_for_cv = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_for_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    return filtered_contours


# --- Deforestation Change Detection Function ---
def detect_deforestation(before_image_path, after_image_path, min_change_area_pixels):
    # Access models from session state
    if st.session_state.def_model is None:
        st.warning("Deforestation Deep Learning Model not loaded, cannot perform change detection.")
        return None, []

    st.info(f"Analyzing deforestation between {os.path.basename(before_image_path)} and {os.path.basename(after_image_path)}...")

    # Pass st.session_state.def_model to predict_mask. We don't need preprocessed_image for display here.
    # --- MODIFIED: Unpack crs_before and crs_after from predict_mask ---
    mask_before, transform_before, crs_before, _, _, _ = predict_mask(st.session_state.def_model, before_image_path, DEF_IMG_HEIGHT, DEF_IMG_WIDTH, DEF_NUM_CHANNELS)
    mask_after, transform_after, crs_after, _, _, _ = predict_mask(st.session_state.def_model, after_image_path, DEF_IMG_HEIGHT, DEF_IMG_WIDTH, DEF_NUM_CHANNELS)

    if mask_before is None or mask_after is None:
        st.error("Failed to get masks for deforestation analysis. Check previous errors.")
        return None, []

    if mask_before.shape != mask_after.shape:
        st.error(f"Error: Deforestation masks have different shapes! Before: {mask_before.shape}, After: {mask_after.shape}")
        return None, []

    # Use the transform and CRS from the 'after' image for consistent georeferencing of alerts
    primary_transform = transform_after if transform_after is not None else transform_before
    primary_crs = crs_after if crs_after is not None else crs_before # --- NEW LINE ---
    if primary_transform is None or primary_crs is None: # --- MODIFIED ---
        st.error("Could not get a valid georeferencing transform or CRS for deforestation analysis. Cannot locate alerts.")
        return None, []


    # Calculate initial forested area
    initial_forest_pixels = np.sum(mask_before == 1)
    initial_forest_sqkm = initial_forest_pixels * SQ_KM_PER_PIXEL

    # Deforestation: Forest (1) in 'before' becomes Non-Forest (0) in 'after'
    deforestation_raw = ((mask_before == 1) & (mask_after == 0)).astype(np.uint8)

    kernel = np.ones((3,3), dtype=np.uint8)
    deforestation_processed = binary_erosion(deforestation_raw, structure=kernel, iterations=1).astype(np.uint8)
    deforestation_processed = binary_dilation(deforestation_processed, structure=kernel, iterations=1).astype(np.uint8)

    labeled_array, num_features = label(deforestation_processed)
    
    deforestation_alerts = []
    change_map_final = np.zeros_like(deforestation_raw, dtype=np.uint8)

    total_deforested_pixels_detected = 0

    for i in range(1, num_features + 1):
        region_pixels = (labeled_array == i)
        area_pixels = np.sum(region_pixels)

        if area_pixels >= min_change_area_pixels:
            change_map_final[region_pixels] = 1
            total_deforested_pixels_detected += area_pixels
            
            rows, cols = np.where(region_pixels)
            # Ensure rows/cols are not empty before calculating mean
            if rows.size > 0 and cols.size > 0:
                center_row = int(np.mean(rows))
                center_col = int(np.mean(cols))
                # --- MODIFIED: Pass primary_crs to get_coords_from_pixel ---
                center_lon, center_lat = get_coords_from_pixel(primary_transform, primary_crs, center_row, center_col)
            else:
                center_lon, center_lat = None, None # This is where None can originate

            location_name = "Unknown Area"
            if center_lat is not None and center_lon is not None:
                location_name = get_location_name(center_lat, center_lon) # Call geocoding only if coords are valid
            else:
                st.warning(f"Skipping geocoding for a deforestation alert due to invalid coordinates derived from transform.")

            area_sqkm = area_pixels * SQ_KM_PER_PIXEL
            
            alert_message = (
                f"Deforestation detected near {location_name} (Lat: {center_lat:.4f}°, Lon: {center_lon:.4f}°). "
                f"Area: {area_sqkm:.4f} sq. km ({area_pixels} pixels)."
            )
            deforestation_alerts.append({
                'type': 'Deforestation',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'center_lon': center_lon,
                'center_lat': center_lat,
                'area_pixels': area_pixels,
                'area_sqkm': area_sqkm, # Add sq. km here
                'location_name': location_name, # Add location name
                'message': alert_message
            })
    
    # Calculate approximate change percentage
    approx_change_percentage = 0
    if initial_forest_pixels > 0:
        approx_change_percentage = (total_deforested_pixels_detected / initial_forest_pixels) * 100

    # Store general stats for the overall analysis, potentially for a summary alert
    st.session_state['deforestation_summary'] = {
        'total_deforested_pixels': total_deforested_pixels_detected,
        'total_deforested_sqkm': total_deforested_pixels_detected * SQ_KM_PER_PIXEL,
        'initial_forest_pixels': initial_forest_pixels,
        'initial_forest_sqkm': initial_forest_sqkm,
        'approx_change_percentage': approx_change_percentage
    }

    return change_map_final, deforestation_alerts

# --- Coastal Erosion/Coastline Shift Detection Function ---
def detect_coastal_erosion(before_image_path, after_image_path, pixel_to_meter_ratio=10.0, min_shift_meters=10.0):
    # Access models from session state
    if st.session_state.coastal_model is None:
        st.warning("Coastal Erosion Deep Learning Model not loaded, cannot perform change detection.")
        # Updated return for preprocessed images and change_map
        return None, None, None, None, None, None, None, [], None

    st.info(f"Running coastline shift analysis between {os.path.basename(before_image_path)} and {os.path.basename(after_image_path)}...")

    # 1. Predict land masks for both images
    st.write("Predicting land mask for 'before' image...")
    # --- MODIFIED: Unpack crs_before from predict_mask ---
    mask_before_land, transform_before, crs_before, original_height_before, original_width_before, preprocessed_before = \
        predict_mask(st.session_state.coastal_model, before_image_path, COASTAL_IMG_HEIGHT, COASTAL_IMG_WIDTH, COASTAL_NUM_CHANNELS)
    
    if mask_before_land is None:
        st.error("Failed to get land mask for 'before' image. Check console for details.")
        return None, None, None, None, None, None, None, [], None

    st.write("Predicting land mask for 'after' image...")
    # --- MODIFIED: Unpack crs_after from predict_mask ---
    mask_after_land, transform_after, crs_after, original_height_after, original_width_after, preprocessed_after = \
        predict_mask(st.session_state.coastal_model, after_image_path, COASTAL_IMG_HEIGHT, COASTAL_IMG_WIDTH, COASTAL_NUM_CHANNELS)
    
    if mask_after_land is None:
        st.error("Failed to get land mask for 'after' image. Check console for details.")
        return None, None, None, None, None, None, None, [], None

    if mask_before_land.shape != mask_after_land.shape:
        st.error(f"Error: Predicted masks have different shapes! Before: {mask_before_land.shape}, After: {mask_after_land.shape}")
        return None, None, None, None, None, None, None, [], None

    # Use the transform and CRS from the 'after' image for consistent georeferencing of alerts
    primary_transform = transform_after if transform_after is not None else transform_before
    primary_crs = crs_after if crs_after is not None else crs_before # --- NEW LINE ---
    if primary_transform is None or primary_crs is None: # --- MODIFIED ---
        st.error("Could not get a valid georeferencing transform or CRS for coastal analysis. Cannot locate alerts.")
        # Still return other valid data for display if possible, but alerts will be empty
        # Placeholder for original RGB images for display purposes if actual loading fails
        dummy_shape = (mask_before_land.shape[0], mask_before_land.shape[1], 3)
        original_before_img_rgb_display_dummy = np.zeros(dummy_shape, dtype=np.uint8)
        original_after_img_rgb_display_dummy = np.zeros(dummy_shape, dtype=np.uint8)

        return (original_before_img_rgb_display_dummy, original_after_img_rgb_display_dummy,
                mask_before_land, mask_after_land, preprocessed_before, preprocessed_after,
                np.zeros((*mask_after_land.shape, 3), dtype=np.uint8), [], np.zeros_like(mask_after_land, dtype=np.uint8))


    # 2. Extract coastline contours from the predicted land masks
    # Use a slightly eroded mask to get a cleaner "coastline" that is less noisy
    kernel_small = np.ones((3,3), dtype=np.uint8)
    eroded_mask_before = binary_erosion(mask_before_land, structure=kernel_small, iterations=1).astype(np.uint8)
    eroded_mask_after = binary_erosion(mask_after_land, structure=kernel_small, iterations=1).astype(np.uint8)

    st.write("Extracting coastline contours from 'before' land mask...")
    coastline_before_contours = extract_coastline_contour(eroded_mask_before)
    st.write(f"Found {len(coastline_before_contours)} contours for 'before' coastline.")

    st.write("Extracting coastline contours from 'after' land mask...")
    coastline_after_contours = extract_coastline_contour(eroded_mask_after)
    st.write(f"Found {len(coastline_after_contours)} contours for 'after' coastline.")

    # 3. Load the original RGB images for background visualization
    original_before_img_rgb_display = None
    original_after_img_rgb_display = None
    try:
        with rasterio.open(before_image_path) as src_before:
            # Sentinel-2 bands are typically B4 (Red), B3 (Green), B2 (Blue) for natural color RGB
            # Assuming bands are 1-indexed (B2, B3, B4 -> indices 1,2,3 for RGB as per GEE band naming)
            # Or if you fetched B4,B3,B2 explicitly, it might be 0,1,2 in your array.
            # If `required_bands` was ['B2', 'B3', 'B4', 'B8'], then RGB would be indices 0,1,2
            # If `required_bands` was ['B4', 'B3', 'B2', 'B8'], then RGB would be indices 0,1,2
            img_data_before = src_before.read([1, 2, 3]).transpose((1, 2, 0)) # Assuming 1-based indexing for rasterio.read() bands
            if img_data_before.max() > 0:
                original_before_img_rgb_display = (img_data_before / img_data_before.max() * 255).astype(np.uint8)
            else:
                original_before_img_rgb_display = np.zeros_like(img_data_before[:,:,:3], dtype=np.uint8)

        with rasterio.open(after_image_path) as src_after:
            img_data_after = src_after.read([1, 2, 3]).transpose((1, 2, 0)) # Assuming 1-based indexing for rasterio.read() bands
            if img_data_after.max() > 0:
                original_after_img_rgb_display = (img_data_after / img_data_after.max() * 255).astype(np.uint8)
            else:
                original_after_img_rgb_display = np.zeros_like(img_data_after[:,:,:3], dtype=np.uint8)

    except Exception as e:
        st.error(f"Error loading original coastal images for contour display: {e}")
        dummy_shape = (mask_after_land.shape[0], mask_after_land.shape[1], 3)
        original_before_img_rgb_display = np.zeros(dummy_shape, dtype=np.uint8)
        original_after_img_rgb_display = np.zeros(dummy_shape, dtype=np.uint8)

    # 4. Create an image with both 'before' and 'after' coastlines overlaid on the 'after' image
    image_with_coasts_overlay = original_after_img_rgb_display.copy()
    
    cv2.drawContours(image_with_coasts_overlay, coastline_before_contours, -1, (0, 255, 0), 2) # Green color for 'Before'
    cv2.drawContours(image_with_coasts_overlay, coastline_after_contours, -1, (255, 0, 0), 2) # Red color for 'After'

    # 5. Measure Coastline Shift and Generate Alerts
    alerts = []
    # Initialize change_map_final to highlight areas of shift later
    change_map_final = np.zeros_like(mask_after_land, dtype=np.uint8)

    if not coastline_before_contours or not coastline_after_contours:
        st.warning("Could not detect sufficient coastlines in both images for shift analysis. Make sure land is visible.")
        return (original_before_img_rgb_display, original_after_img_rgb_display,
                mask_before_land, mask_after_land, preprocessed_before, preprocessed_after,
                image_with_coasts_overlay, alerts, change_map_final)

    # Convert contours to binary images for distance transform
    mask_before_contour = np.zeros_like(mask_before_land, dtype=np.uint8)
    cv2.drawContours(mask_before_contour, coastline_before_contours, -1, 1, thickness=1)

    mask_after_contour = np.zeros_like(mask_after_land, dtype=np.uint8)
    cv2.drawContours(mask_after_contour, coastline_after_contours, -1, 1, thickness=1)

    # Calculate distance transform (distance from non-zero pixels) for the 'before' coastline
    # This creates an image where each pixel value is its distance to the nearest 'before' coastline pixel.
    # We use 2 (L2) for Euclidean distance
    # IMPORTANT: cv2.distanceTransform works on background (0s). So if 1 is coastline, use 1-mask.
    dist_to_before = cv2.distanceTransform(1 - mask_before_contour, cv2.DIST_L2, 5) # 5 for 5x5 mask (better accuracy)

    # Where the 'after' coastline is, read the distance to the 'before' coastline
    # This gives us the shift distance at each point of the 'after' coastline
    rows_after_contour, cols_after_contour = np.where(mask_after_contour == 1)

    if rows_after_contour.size > 0:
        # Get shift distances for all points on the 'after' contour
        # Ensure indices are within bounds of dist_to_before
        valid_indices = (rows_after_contour < dist_to_before.shape[0]) & \
                        (cols_after_contour < dist_to_before.shape[1])
        
        rows_after_valid = rows_after_contour[valid_indices]
        cols_after_valid = cols_after_contour[valid_indices]

        shift_distances = dist_to_before[rows_after_valid, cols_after_valid]
    else:
        shift_distances = np.array([]) # No coastline to measure shift

    # Filter for significant shifts (erosion: land receded)
    # A positive shift means the 'after' coastline is further inland (more land eroded/shifted)
    # i.e., distance from new coastline to old land (old coastline) is large
    significant_shifts_pixels = shift_distances[shift_distances > (min_shift_meters / pixel_to_meter_ratio)]
    
    if significant_shifts_pixels.size > 0:
        avg_shift_pixel = np.mean(significant_shifts_pixels)
        avg_shift_meter = avg_shift_pixel * pixel_to_meter_ratio
        
        st.success(f"Average coastline shift detected: {avg_shift_meter:.2f} meters (or {avg_shift_pixel:.2f} pixels).")

        # Identify areas of significant shift for the change_map_final
        # These are the pixels on the 'after' contour that show significant shift
        
        # Create a mask for significant shift points
        significant_shift_points_mask = np.zeros_like(mask_after_land, dtype=np.uint8)
        
        if rows_after_valid.size > 0: # Check if there are valid points to process
            shift_filter_indices = np.where(shift_distances > (min_shift_meters / pixel_to_meter_ratio))[0]
            
            # Populate significant_shift_points_mask with the erosion points
            for idx_in_valid in shift_filter_indices:
                r, c = rows_after_valid[idx_in_valid], cols_after_valid[idx_in_valid]
                significant_shift_points_mask[r, c] = 1
        
        # Dilate the significant_shift_points_mask slightly to make the alerted area visible
        # This will be `change_map_final`
        if np.sum(significant_shift_points_mask) > 0:
            kernel_dilate = np.ones((5,5), dtype=np.uint8) # Slightly larger kernel for visibility
            change_map_final = binary_dilation(significant_shift_points_mask, structure=kernel_dilate, iterations=2).astype(np.uint8)
        else:
            change_map_final = np.zeros_like(mask_after_land, dtype=np.uint8)

        # Generate alerts for large shifted areas
        labeled_shift_areas, num_shift_features = label(change_map_final)
        
        for i in range(1, num_shift_features + 1):
            region_pixels = (labeled_shift_areas == i)
            area_pixels = np.sum(region_pixels)
            
            # Threshold for alert on shifted area (you can adjust this)
            min_alert_area_pixels = st.session_state.get('min_alert_area_pixels_coastal_slider', 200) # Use the slider value
            if area_pixels >= min_alert_area_pixels:
                rows, cols = np.where(region_pixels)
                
                center_lon, center_lat = None, None
                if rows.size > 0 and cols.size > 0: # Ensure valid pixels for centroid calculation
                    center_row = int(np.mean(rows))
                    center_col = int(np.mean(cols))
                    # --- MODIFIED: Pass primary_crs to get_coords_from_pixel ---
                    center_lon, center_lat = get_coords_from_pixel(primary_transform, primary_crs, center_row, center_col)
                
                location_name = "Unknown Area"
                shift_at_center = 0.0 # Initialize shift_at_center
                if center_lat is not None and center_lon is not None:
                    location_name = get_location_name(center_lat, center_lon) # Get location name
                    # Recalculate shift at center for the alert message, ensuring valid indices
                    # Check bounds before accessing dist_to_before
                    if (0 <= center_row < dist_to_before.shape[0] and
                        0 <= center_col < dist_to_before.shape[1]):
                        shift_at_center = dist_to_before[center_row, center_col] * pixel_to_meter_ratio
                    
                else:
                    st.warning(f"Skipping geocoding for a coastal alert due to invalid coordinates derived from transform.")

                area_sqkm = area_pixels * SQ_KM_PER_PIXEL # Area of the highlighted shifted region

                alert_message = (
                    f"Coastline receded by approx. {shift_at_center:.2f} meters "
                    f"near {location_name} (Lat: {center_lat:.4f}°, Lon: {center_lon:.4f}°). "
                    f"Shifted area: {area_sqkm:.4f} sq. km ({area_pixels} pixels)."
                )
                alerts.append({
                    'type': 'Coastal Erosion (Shift)',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'center_lon': center_lon,
                    'center_lat': center_lat,
                    'area_pixels': area_pixels, # This is the area of the shifted region
                    'area_sqkm': area_sqkm, # Add sq. km here for coastal alerts
                    'shift_meters': shift_at_center, # The shift distance at the center
                    'location_name': location_name, # Add location name
                    'message': alert_message
                })

    if not alerts:
        st.info("No significant coastline shifts detected above the set threshold.")

    return (original_before_img_rgb_display, original_after_img_rgb_display,
            mask_before_land, mask_after_land, preprocessed_before, preprocessed_after,
            image_with_coasts_overlay, alerts, change_map_final)


# --- Alerting Functions (EMAIL) ---

def send_email_alert(recipient_email, subject, body, attachment_path=None):
    """
    Sends an email alert with optional image attachment.
    """
    if not SENDER_EMAIL or not SENDER_EMAIL_PASSWORD:
        raise ValueError("Sender email or password environment variables are not set. Cannot send email.")

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    if attachment_path and os.path.exists(attachment_path):
        try:
            with open(attachment_path, 'rb') as fp:
                img = MIMEImage(fp.read())
            img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(attachment_path))
            msg.attach(img)
        except Exception as e:
            st.warning(f"Could not attach image {os.path.basename(attachment_path)}: {e}")

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Secure the connection
            server.login(SENDER_EMAIL, SENDER_EMAIL_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        return True
    except smtplib.SMTPAuthenticationError:
        raise Exception("Failed to authenticate with SMTP server. Check SENDER_EMAIL and SENDER_EMAIL_PASSWORD (especially if using Gmail, ensure it's an App Password).")
    except smtplib.SMTPConnectError:
        raise Exception("Failed to connect to SMTP server. Check server address and port, or network/firewall settings.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while sending email: {e}")

def log_alert_to_csv(alert_data, alert_type):
    """
    Logs alert data to a CSV file.
    """
    log_file_path = os.path.join(ALERTS_LOG_FOLDER, f"{alert_type.lower().replace(' ', '_')}_alerts.csv")
    
    # Define columns explicitly to ensure order and handle missing ones
    columns = [
        'timestamp', 'type', 'location_name', 'center_lat', 'center_lon',
        'area_pixels', 'area_sqkm', 'shift_meters', 'message',
        'recipient_email', 'map_snippet_path'
    ]
    
    # Ensure 'area_sqkm' and 'location_name' are correctly handled.
    # 'shift_meters' only applies to Coastal Erosion.
    
    # Convert alert_data to a DataFrame row, ensuring all columns are present
    log_data_row = {col: alert_data.get(col, '') for col in columns}
    df = pd.DataFrame([log_data_row])

    # Check if file exists to decide whether to write header
    file_exists = os.path.exists(log_file_path)
    df.to_csv(log_file_path, mode='a', header=not file_exists, index=False)
    st.sidebar.info(f"Alert logged to: {os.path.basename(log_file_path)}")


# --- Streamlit App Layout ---


st.markdown("---")

# --- Sidebar for GEE Authentication and General Info ---
st.sidebar.header("Configuration & Status")
st.sidebar.markdown("---")

# Display GEE initialization status
if gee_status_type == "success":
    st.sidebar.success(gee_status_msg)
else:
    st.sidebar.error(gee_status_msg)
    if 'ee_authenticated' not in st.session_state or not st.session_state.ee_authenticated:
        if st.sidebar.button("Authenticate GEE", key="gee_authenticate_button"):
            try:
                ee.Authenticate()
                st.session_state.ee_authenticated = True
                st.experimental_rerun()
            except Exception as e:
                st.sidebar.error(f"GEE Authentication failed: {e}")
    else:
        st.sidebar.success("GEE authenticated.")

# Display Model loading status for both models
st.sidebar.markdown("---")
st.sidebar.subheader("Model Loading Status:")
if def_model_status_type == "success":
    st.sidebar.success(def_model_msg)
else:
    st.sidebar.error(def_model_msg)

if coastal_model_status_type == "success":
    st.sidebar.success(coastal_model_msg)
else:
    st.sidebar.error(coastal_model_msg)


# --- Main Content Area ---
selected_module = st.radio(
    "Select Module:",
    ("Deforestation Detection", "Coastal Erosion Detection"),
    index=0,
    key="module_selection_radio"
)

# Initialize alerts lists in session state to persist them across reruns
if 'deforestation_alerts_list' not in st.session_state:
    st.session_state.deforestation_alerts_list = []
if 'coastal_alerts_list' not in st.session_state:
    st.session_state.coastal_alerts_list = []
if 'deforestation_summary' not in st.session_state: # Initialize deforestation summary
    st.session_state.deforestation_summary = {}

# --- DEFORESTATION DETECTION MODULE ---
if selected_module == "Deforestation Detection":
    st.header("🌳 Deforestation Detection Module")

    st.subheader("1. Define Area of Interest (AOI) & Dates")

    default_min_lon = st.session_state.get('def_min_lon', 74.79)
    default_min_lat = st.session_state.get('def_min_lat', 13.31)
    default_max_lon = st.session_state.get('def_max_lon', 74.82)
    default_max_lat = st.session_state.get('def_max_lat', 13.33)

    col_aoi1, col_aoi2 = st.columns(2)
    with col_aoi1:
        min_lon = st.number_input("Min Longitude:", value=default_min_lon, format="%.4f", key="def_min_lon_input")
        max_lon = st.number_input("Max Longitude:", value=default_max_lon, format="%.4f", key="def_max_lon_input")
    with col_aoi2:
        min_lat = st.number_input("Min Latitude:", value=default_min_lat, format="%.4f", key="def_min_lat_input")
        max_lat = st.number_input("Max Latitude:", value=default_max_lat, format="%.4f", key="def_max_lat_input")

    st.session_state['def_min_lon'] = min_lon
    st.session_state['def_min_lat'] = min_lat
    st.session_state['def_max_lon'] = max_lon
    st.session_state['def_max_lat'] = max_lat

    aoi_coords = [min_lon, min_lat, max_lon, max_lat]

    st.markdown("---")
    st.subheader("2. Select Dates for Analysis")
    today = datetime.now()
    
    before_date_dt_def = st.date_input("Select 'Before' Date (Earlier Image):", value=today - timedelta(days=365), key="def_before_date_input")
    before_date_str_def = before_date_dt_def.strftime("%Y-%m-%d")

    after_date_dt_def = st.date_input("Select 'After' Date (Later Image):", value=today, key="def_after_date_input")
    after_date_str_def = after_date_dt_def.strftime("%Y-%m-%d")

    if before_date_dt_def >= after_date_dt_def:
        st.warning("⚠️ 'Before' Date should be earlier than 'After' Date. Please adjust the dates.")
    
    st.info(f"The system will analyze images between **{before_date_str_def}** and **{after_date_str_def}**.")

    if st.button("Fetch Latest Satellite Images & Run Deforestation Analysis", key="fetch_deforestation_images_button"):
        if gee_status_type != "success" or ('ee_authenticated' in st.session_state and not st.session_state.ee_authenticated):
            st.error("GEE is not initialized or authenticated. Please check the sidebar for status and authenticate if needed.")
        elif st.session_state.def_model is None:
            st.error("Deforestation Deep Learning Model failed to load. Cannot run analysis.")
        elif before_date_dt_def >= after_date_dt_def:
            st.error("Please ensure the 'Before' Date is earlier than the 'After' Date.")
        else:
            with st.spinner("Fetching images from Google Earth Engine and running deforestation analysis..."):
                aoi_hash = f"{min_lon}_{min_lat}_{max_lon}_{max_lat}".replace('.', '_').replace('-', 'm')
                before_filename_prefix = f"deforestation_aoi_{aoi_hash}_before_{before_date_str_def.replace('-', '')}"
                after_filename_prefix = f"deforestation_aoi_{aoi_hash}_after_{after_date_str_def.replace('-', '')}"

                st.session_state['def_before_image_path'] = fetch_and_save_image_from_gee(
                    aoi_coords, before_date_dt_def, 60, # Increased buffer
                    OUTPUT_BASE_FOLDER, before_filename_prefix, ['B4', 'B3', 'B2', 'B8'] # RGB + NIR for deforestation
                )
                st.session_state['def_after_image_path'] = fetch_and_save_image_from_gee(
                    aoi_coords, after_date_dt_def, 60, # Increased buffer
                    OUTPUT_BASE_FOLDER, after_filename_prefix, ['B4', 'B3', 'B2', 'B8'] # RGB + NIR for deforestation
                )
            
            if st.session_state.get('def_before_image_path') and st.session_state.get('def_after_image_path'):
                st.success("Deforestation images fetched successfully! Running analysis...")

                original_before_img_rgb = None
                original_after_img_rgb = None
                try:
                    with rasterio.open(st.session_state['def_before_image_path']) as src_before:
                        # Assuming band order is B4, B3, B2 for display after being selected that way
                        img_data_before = src_before.read([1, 2, 3]).transpose((1, 2, 0))
                        if img_data_before.max() > 0:
                            original_before_img_rgb = (img_data_before / img_data_before.max() * 255).astype(np.uint8)
                        else:
                            original_before_img_rgb = np.zeros_like(img_data_before[:,:,:3], dtype=np.uint8)


                    with rasterio.open(st.session_state['def_after_image_path']) as src_after:
                        img_data_after = src_after.read([1, 2, 3]).transpose((1, 2, 0))
                        if img_data_after.max() > 0:
                            original_after_img_rgb = (img_data_after / img_data_after.max() * 255).astype(np.uint8)
                        else:
                            original_after_img_rgb = np.zeros_like(img_data_after[:,:,:3], dtype=np.uint8)
                except Exception as e:
                    st.error(f"Error loading original deforestation images for display: {e}")
                    st.sidebar.error(f"DEBUG: Error loading deforestation RGB for display: {e}")

                min_change_area_pixels = st.session_state.get('def_min_change_area_pixels_slider', 50) # Adjusted default value
                change_map, alerts = detect_deforestation(st.session_state['def_before_image_path'], st.session_state['def_after_image_path'], min_change_area_pixels)

                # Store alerts in session state
                st.session_state.deforestation_alerts_list = alerts

                st.markdown("---")
                st.subheader("3. Deforestation Analysis Results")

                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    if original_before_img_rgb is not None and original_before_img_rgb.size > 0:
                        st.image(original_before_img_rgb, caption="Original 'Before' Image (Deforestation)", use_container_width=True)
                    else:
                        st.warning("Could not display 'Before' original deforestation image.")
                with col_img2:
                    if original_after_img_rgb is not None and original_after_img_rgb.size > 0:
                        st.image(original_after_img_rgb, caption="Original 'After' Image (Deforestation)", use_container_width=True)
                    else:
                        st.warning("Could not display 'After' original deforestation image.")

                st.markdown("#### Predicted Forest Masks (for debugging/visual inspection)")
                # --- MODIFIED: Unpack crs for display masks as well (though not strictly used for display) ---
                mask_before_display, _, _, _, _, _ = predict_mask(st.session_state.def_model, st.session_state['def_before_image_path'], DEF_IMG_HEIGHT, DEF_IMG_WIDTH, DEF_NUM_CHANNELS)
                mask_after_display, _, _, _, _, _ = predict_mask(st.session_state.def_model, st.session_state['def_after_image_path'], DEF_IMG_HEIGHT, DEF_IMG_WIDTH, DEF_NUM_CHANNELS)
                
                mask_col1, mask_col2 = st.columns(2)
                with mask_col1:
                    if mask_before_display is not None and mask_before_display.size > 0:
                        st.image(mask_before_display * 255, caption="Before Forest Mask", use_container_width=True, channels="GRAY")
                    else:
                        st.warning("Could not display 'Before' forest mask.")
                with mask_col2:
                    if mask_after_display is not None and mask_after_display.size > 0:
                        st.image(mask_after_display * 255, caption="After Forest Mask", use_container_width=True, channels="GRAY")
                    else:
                        st.warning("Could not display 'After' forest mask.")

                st.markdown("#### Detected Deforestation Areas")
                map_snippet_path_def = None
                if change_map is not None and np.sum(change_map) > 0:
                    summary = st.session_state.get('deforestation_summary', {})
                    total_deforested_sqkm = summary.get('total_deforested_sqkm', 0)
                    initial_forest_sqkm = summary.get('initial_forest_sqkm', 0)
                    approx_change_percentage = summary.get('approx_change_percentage', 0)

                    st.success(f"Total Deforestation Detected: **{total_deforested_sqkm:.4f} sq. km** "
                               f"({np.sum(change_map)} pixels).")
                    
                    if initial_forest_sqkm > 0:
                        st.info(f"Approximately **{approx_change_percentage:.2f}%** of the "
                                f"original forested area ({initial_forest_sqkm:.4f} sq. km) has been deforested.")
                    fig, ax = plt.subplots(figsize=(10, 10))
                    if original_after_img_rgb is not None and original_after_img_rgb.size > 0:
                        ax.imshow(original_after_img_rgb)
                    ax.imshow(change_map, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
                    
                    # Add area name to plot
                    if st.session_state.deforestation_alerts_list:
                        # Take the name from the first alert, or a more general name if available
                        first_alert_location = st.session_state.deforestation_alerts_list[0].get('location_name', 'Detected Area')
                        ax.text(0.02, 0.98, f"Area: {first_alert_location}", transform=ax.transAxes, 
                                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

                    ax.set_title('Deforestation Overlay on After Image')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)

                    # --- Save Map Snippet for Deforestation ---
                    map_snippet_folder = os.path.join(ALERTS_LOG_FOLDER, "deforestation_maps")
                    os.makedirs(map_snippet_folder, exist_ok=True)
                    map_snippet_path_def = os.path.join(map_snippet_folder, f"deforestation_alert_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                    
                    fig_save, ax_save = plt.subplots(figsize=(6, 6)) # Smaller figure for snippet
                    if original_after_img_rgb is not None and original_after_img_rgb.size > 0:
                        ax_save.imshow(original_after_img_rgb)
                    ax_save.imshow(change_map, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
                    ax_save.set_title('Deforestation Detected')
                    if st.session_state.deforestation_alerts_list:
                        first_alert_location = st.session_state.deforestation_alerts_list[0].get('location_name', 'Detected Area')
                        ax_save.text(0.02, 0.98, f"Area: {first_alert_location}", transform=ax_save.transAxes, 
                                     fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
                    ax_save.axis('off')
                    plt.savefig(map_snippet_path_def, bbox_inches='tight', pad_inches=0.1)
                    plt.close(fig_save)
                    st.sidebar.info(f"Saved deforestation map snippet to: {os.path.basename(map_snippet_path_def)}")
                    # --- End Save Map Snippet ---

                    st.markdown("---")
                    st.subheader("4. Deforestation Alerts")
                    if alerts:
                        st.warning(f"🚨 **{len(alerts)}** significant deforestation alerts detected!")
                        for alert in alerts:
                            st.write(f"- {alert['message']}")
                        
                        # Store the map snippet path with alerts if email is sent later
                        for alert_detail in st.session_state.deforestation_alerts_list:
                            alert_detail['map_snippet_path'] = map_snippet_path_def if map_snippet_path_def else ''
                            
                    else:
                        st.info("No significant deforestation alerts generated above the set threshold.")
                else:
                    st.info("No significant deforestation detected in the selected area between the dates.")
            else:
                st.error("Failed to fetch one or both images for deforestation. Check the console/sidebar for GEE errors or try a different date range/AOI.")

    st.markdown("---")
    st.subheader("6. Adjust Deforestation Sensitivity")
    min_change_area_pixels = st.slider(
        "Minimum Deforestation Area (pixels) for Alert:",
        min_value=10, max_value=200, value=50, step=10, # Max value changed to 500
        help="Increase this value to detect only larger deforestation events. Decrease for smaller changes (may include noise).",
        key="def_min_change_area_pixels_slider"
    )

    # --- Email Alert Section for Deforestation (outside the "if button clicked" block) ---
    st.markdown("---")
    st.subheader("5. Send Alerts (Email Only)")
    
    # Persistent checkbox for enabling email alerts
    # If the checkbox is clicked, it triggers a rerun, but its state is saved.
    st.session_state.send_email_deforestation_enabled = st.checkbox(
        "Enable Deforestation Email Alert", 
        value=st.session_state.get('send_email_deforestation_enabled', False),
        key="email_checkbox_def_global"
    )

    if st.session_state.send_email_deforestation_enabled:
        # Only show recipient input and send button if enabled
        recipient_email_def = st.text_input(
            "Recipient Email Address (Deforestation Alerts):", 
            value=st.session_state.get("email_recipient_def", ""),
            key="email_recipient_def"
        )

        if st.button("Send Deforestation Email Now", key="send_email_button_def"):
            if not st.session_state.deforestation_alerts_list:
                st.warning("No deforestation alerts to send. Run analysis first.")
            elif not recipient_email_def.strip():
                st.warning("Please enter a valid recipient email address.")
            else:
                email_subject = f"EcoVision Alert: Deforestation Detected"
                
                email_body = "Dear User,\n\nEcoVision has detected deforestation event(s) in your monitored Area of Interest.\n\n"
                
                # Add overall summary to email body
                summary = st.session_state.get('deforestation_summary', {})
                total_deforested_sqkm = summary.get('total_deforested_sqkm', 0)
                initial_forest_sqkm = summary.get('initial_forest_sqkm', 0)
                approx_change_percentage = summary.get('approx_change_percentage', 0)

                if total_deforested_sqkm > 0:
                    email_body += f"Overall, approximately **{total_deforested_sqkm:.4f} sq. km** of forest area has been lost. "
                    if initial_forest_sqkm > 0:
                        email_body += f"This represents about **{approx_change_percentage:.2f}%** of the original forested area in the 'before' image timeframe.\n\n"
                    else:
                        email_body += "\n\n"

                email_body += f"Details of {len(st.session_state.deforestation_alerts_list)} detected event(s):\n"

                # Check if map snippet path is available from the last run
                map_snippet_path_def_for_email = st.session_state.deforestation_alerts_list[0].get('map_snippet_path') if st.session_state.deforestation_alerts_list else None
                
                for alert in st.session_state.deforestation_alerts_list:
                    email_body += f"- {alert['message']}\n"
                email_body += "\nReview your EcoVision dashboard for more details.\n\nSincerely,\nEcoVision Team"

                try:
                    send_email_alert(
                        recipient_email_def,
                        email_subject,
                        email_body,
                        attachment_path=map_snippet_path_def_for_email # Pass the saved path
                    )

                    for alert_detail in st.session_state.deforestation_alerts_list:
                        alert_detail['recipient_email'] = recipient_email_def
                        log_alert_to_csv(alert_detail, "Deforestation")

                    st.success("✅ Deforestation alert email sent successfully!")
                    # Clear the "Send Email" button/state after sending if desired, or let it rerun.
                    # For a button, it automatically triggers a rerun and then disappears.
                except Exception as e:
                    st.error(f"❌ Failed to send email: {e}")
                    st.exception(e) # Display full traceback for debugging


#-----------------------------------------------------------------------------------------------------------------------

elif selected_module == "Coastal Erosion Detection":
    st.header("🌊 Coastal Erosion Detection Module")

    st.subheader("1. Define Area of Interest (AOI) & Dates")

    default_min_lon = st.session_state.get('coastal_min_lon', 79.7826)
    default_min_lat = st.session_state.get('coastal_min_lat',11.7892)
    default_max_lon = st.session_state.get('coastal_max_lon', 79.8292)
    default_max_lat = st.session_state.get('coastal_max_lat',11.8347)

    col_aoi1, col_aoi2 = st.columns(2)
    with col_aoi1:
        min_lon = st.number_input("Min Longitude:", value=default_min_lon, format="%.4f", key="coastal_min_lon_input")
        max_lon = st.number_input("Max Longitude:", value=default_max_lon, format="%.4f", key="coastal_max_lon_input")
    with col_aoi2:
        min_lat = st.number_input("Min Latitude:", value=default_min_lat, format="%.4f", key="coastal_min_lat_input")
        max_lat = st.number_input("Max Latitude:", value=default_max_lat, format="%.4f", key="coastal_max_lat_input")

    st.session_state['coastal_min_lon'] = min_lon
    st.session_state['coastal_min_lat'] = min_lat
    st.session_state['coastal_max_lon'] = max_lon
    st.session_state['coastal_max_lat'] = max_lat

    aoi_coords = [min_lon, min_lat, max_lon, max_lat]

    st.markdown("---")
    st.subheader("2. Select Dates for Analysis")
    today = datetime.now()

    before_date_dt_coastal = st.date_input("Select 'Before' Date (Earlier Image):", value=today - timedelta(days=180), key="coastal_before_date_input")
    before_date_str_coastal = before_date_dt_coastal.strftime("%Y-%m-%d")

    after_date_dt_coastal = st.date_input("Select 'After' Date (Later Image):", value=today, key="coastal_after_date_input")
    after_date_str_coastal = after_date_dt_coastal.strftime("%Y-%m-%d")

    if before_date_dt_coastal >= after_date_dt_coastal:
        st.warning("⚠️ 'Before' Date should be earlier than 'After' Date. Please adjust the dates.")

    st.info(f"The system will analyze images between **{before_date_str_coastal}** and **{after_date_str_coastal}**.")
    
    st.markdown("---")
    st.subheader("3. Adjust Coastal Erosion Sensitivity (Coastline Shift)")
    
    # New sliders for coastline shift parameters
    # Assuming Sentinel-2 pixels are 10x10 meters. Adjust pixel_to_meter_ratio if using other satellite data.
    pixel_to_meter_ratio = st.slider(
        "Pixel to Meter Ratio (m/pixel):",
        min_value=1.0, max_value=30.0, value=10.0, step=1.0,
        help="Set this based on the resolution of your satellite imagery (e.g., 10 for Sentinel-2, 30 for Landsat).",
        key="coastal_pixel_to_meter_ratio_slider" # Added a key
    )
    
    min_shift_meters = st.slider(
        "Minimum Coastline Shift for Alert (meters):",
        min_value=1.0, max_value=100.0, value=10.0, step=1.0,
        help="Only shifts greater than this distance will trigger an alert.",
        key="coastal_min_shift_meters_slider" # Added a key
    )

    min_alert_area_pixels_coastal = st.slider(
        "Minimum Contiguous Shift Area for Alert (pixels):",
        min_value=10, max_value=200, value=10, step=5,
        help="Only contiguous shifted areas larger than this will trigger an alert. Helps filter noise.",
        key="min_alert_area_pixels_coastal_slider" # This key is used in detect_coastal_erosion
    )


    if st.button("Fetch Latest Satellite Images & Run Coastal Erosion Analysis", key="fetch_coastal_erosion_images_button"):
        if gee_status_type != "success" or ('ee_authenticated' in st.session_state and not st.session_state.ee_authenticated):
            st.error("GEE is not initialized or authenticated. Please check the sidebar for status and authenticate if needed.")
        elif st.session_state.coastal_model is None:
            st.error("Coastal Erosion Deep Learning Model failed to load. Cannot run analysis.")
        elif before_date_dt_coastal >= after_date_dt_coastal:
            st.error("Please ensure the 'Before' Date is earlier than the 'After' Date.")
        else:
            with st.spinner("Fetching images from Google Earth Engine and running coastal erosion analysis..."):
                aoi_hash = f"{min_lon}_{min_lat}_{max_lon}_{max_lat}".replace('.', '_').replace('-', 'm')
                before_filename_prefix = f"coastal_aoi_{aoi_hash}_before_{before_date_str_coastal.replace('-', '')}"
                after_filename_prefix = f"coastal_aoi_{aoi_hash}_after_{after_date_str_coastal.replace('-', '')}"

                required_bands_coastal = ['B4', 'B3', 'B2', 'B8'] # RGB and NIR

                st.session_state['coastal_before_image_path'] = fetch_and_save_image_from_gee(
                    aoi_coords, before_date_dt_coastal, 60, # Increased buffer
                    OUTPUT_BASE_FOLDER, before_filename_prefix, required_bands_coastal
                )
                st.session_state['coastal_after_image_path'] = fetch_and_save_image_from_gee(
                    aoi_coords, after_date_dt_coastal, 60, # Increased buffer
                    OUTPUT_BASE_FOLDER, after_filename_prefix, required_bands_coastal
                )

            if st.session_state.get('coastal_before_image_path') and st.session_state.get('coastal_after_image_path'):
                st.success("Coastal images fetched successfully! Running analysis...")

                # Pass the new parameters to detect_coastal_erosion
                original_before_img_rgb, original_after_img_rgb, mask_before_land, mask_after_land, \
                preprocessed_before_img, preprocessed_after_img, image_with_coasts_overlay, alerts, change_map = \
                    detect_coastal_erosion(st.session_state['coastal_before_image_path'],
                                           st.session_state['coastal_after_image_path'],
                                           pixel_to_meter_ratio, min_shift_meters)

                # Store alerts in session state
                st.session_state.coastal_alerts_list = alerts

                st.markdown("---")
                st.subheader("4. Coastal Erosion Analysis Results") # Renumbered from 3

                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    if original_before_img_rgb is not None and original_before_img_rgb.size > 0:
                        st.image(original_before_img_rgb, caption="Original 'Before' Image (Coastal)", use_container_width=True)
                    else:
                        st.warning("Could not display 'Before' original coastal image.")
                with col_img2:
                    if original_after_img_rgb is not None and original_after_img_rgb.size > 0:
                        st.image(original_after_img_rgb, caption="Original 'After' Image (Coastal)", use_container_width=True)
                    else:
                        st.warning("Could not display 'After' original coastal image.")

                st.markdown("#### Preprocessed Images (Model Input) - for Debugging")
                prep_col1, prep_col2 = st.columns(2)
                with prep_col1:
                    if preprocessed_before_img is not None and preprocessed_before_img.size > 0:
                        # Display only first 3 channels as RGB (assuming preprocessed_before_img is (1, H, W, C))
                        # Scale to 0-255 for display if values are 0-1
                        display_img = (preprocessed_before_img.squeeze()[:, :, :3] * 255).astype(np.uint8)
                        st.image(display_img, caption="Preprocessed 'Before' Image (Model Input)", use_container_width=True)
                    else:
                        st.warning("Could not display preprocessed 'Before' image.")
                with prep_col2:
                    if preprocessed_after_img is not None and preprocessed_after_img.size > 0:
                        display_img = (preprocessed_after_img.squeeze()[:, :, :3] * 255).astype(np.uint8)
                        st.image(display_img, caption="Preprocessed 'After' Image (Model Input)", use_container_width=True)
                    else:
                        st.warning("Could not display preprocessed 'After' image.")

                st.markdown("#### Predicted Land Masks")
                mask_col1, mask_col2 = st.columns(2)
                with mask_col1:
                    if mask_before_land is not None and mask_before_land.size > 0:
                        st.image(mask_before_land * 255, caption="Before Land Mask", use_container_width=True, channels="GRAY")
                    else:
                        st.warning("Could not display 'Before' land mask.")
                with mask_col2:
                    if mask_after_land is not None and mask_after_land.size > 0:
                        st.image(mask_after_land * 255, caption="After Land Mask", use_container_width=True, channels="GRAY")
                    else:
                        st.warning("Could not display 'After' land mask.")

                st.markdown("#### Detected Coastal Erosion Areas & Coastline Shift")
                map_snippet_path_coastal = None
                if image_with_coasts_overlay is not None and image_with_coasts_overlay.size > 0:
                    st.image(image_with_coasts_overlay, caption="Before (Green) & After (Red) Coastlines on After Image", use_container_width=True)
                    st.info("Green Line: Coastline 'Before' | Red Line: Coastline 'After'")

                    if change_map is not None and np.sum(change_map) > 0:
                        st.success(f"Significant Coastline Shift Detected. Total shifted area highlighted: {np.sum(change_map)} pixels.")
                        fig, ax = plt.subplots(figsize=(10, 10))
                        if original_after_img_rgb is not None and original_after_img_rgb.size > 0:
                            ax.imshow(original_after_img_rgb)
                        ax.imshow(change_map, cmap='Blues', alpha=0.5, vmin=0, vmax=1) # Using Blues for erosion
                        ax.set_title('Coastal Erosion (Shift) Overlay on After Image')
                        
                        # Add area name to plot
                        if st.session_state.coastal_alerts_list:
                            first_alert_location = st.session_state.coastal_alerts_list[0].get('location_name', 'Detected Area')
                            ax.text(0.02, 0.98, f"Area: {first_alert_location}", transform=ax.transAxes, 
                                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close(fig)

                        # --- Save Map Snippet for Coastal Erosion ---
                        map_snippet_folder = os.path.join(ALERTS_LOG_FOLDER, "coastal_erosion_maps")
                        os.makedirs(map_snippet_folder, exist_ok=True)
                        map_snippet_path_coastal = os.path.join(map_snippet_folder, f"coastal_erosion_alert_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

                        fig_save, ax_save = plt.subplots(figsize=(6, 6)) # Smaller figure for snippet
                        if original_after_img_rgb is not None and original_after_img_rgb.size > 0:
                            ax_save.imshow(original_after_img_rgb)
                        ax_save.imshow(change_map, cmap='Blues', alpha=0.5, vmin=0, vmax=1)
                        ax_save.set_title('Coastal Erosion (Shift) Detected')
                        if st.session_state.coastal_alerts_list:
                            first_alert_location = st.session_state.coastal_alerts_list[0].get('location_name', 'Detected Area')
                            ax_save.text(0.02, 0.98, f"Area: {first_alert_location}", transform=ax_save.transAxes, 
                                         fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
                        ax_save.axis('off')
                        plt.savefig(map_snippet_path_coastal, bbox_inches='tight', pad_inches=0.1)
                        plt.close(fig_save)
                        st.sidebar.info(f"Saved coastal erosion map snippet to: {os.path.basename(map_snippet_path_coastal)}")
                        # --- End Save Map Snippet ---

                        st.markdown("---")
                        st.subheader("5. Coastal Erosion Alerts") # Renumbered from 4
                        if alerts:
                            st.warning(f"🚨 **{len(alerts)}** significant coastal erosion (coastline shift) alerts detected!")
                            for alert in alerts:
                                st.write(f"- {alert['message']}")

                            # Store the map snippet path with alerts if email is sent later
                            for alert_detail in st.session_state.coastal_alerts_list:
                                alert_detail['map_snippet_path'] = map_snippet_path_coastal if map_snippet_path_coastal else ''
                                
                        else:
                            st.info("No significant coastal erosion (coastline shift) alerts generated above the set threshold.")
                    else:
                        st.info("No significant coastline shift detected in the selected area between the dates.")
                else:
                    st.error("Failed to display coastline shift or change map. Check the console/sidebar for errors.")
            else:
                st.error("Failed to fetch one or both images for coastal erosion. Check the console/sidebar for GEE errors or try a different date range/AOI.")

    st.markdown("---")
    st.subheader("6. Send Alerts (Email Only)") # Renumbered from 5
    
    # Persistent checkbox for enabling email alerts
    st.session_state.send_email_coastal_enabled = st.checkbox(
        "Enable Coastal Erosion Email Alert", 
        value=st.session_state.get('send_email_coastal_enabled', False),
        key="email_checkbox_coastal_global"
    )

    if st.session_state.send_email_coastal_enabled:
        # Only show recipient input and send button if enabled
        recipient_email_coastal = st.text_input(
            "Recipient Email Address (Coastal Erosion Alerts):", 
            value=st.session_state.get("email_recipient_coastal", ""),
            key="email_recipient_coastal"
        )

        if st.button("Send Coastal Erosion Email Now", key="send_email_button_coastal"):
            if not st.session_state.coastal_alerts_list:
                st.warning("No coastal erosion alerts to send. Run analysis first.")
            elif not recipient_email_coastal.strip():
                st.warning("Please enter a valid recipient email address.")
            else:
                email_subject = f"EcoVision Alert: Coastal Erosion Detected"
                email_body = f"Dear User,\n\nEcoVision has detected {len(st.session_state.coastal_alerts_list)} coastal erosion event(s) in your monitored Area of Interest.\n\nDetails:\n"

                # Check if map snippet path is available from the last run
                map_snippet_path_coastal_for_email = st.session_state.coastal_alerts_list[0].get('map_snippet_path') if st.session_state.coastal_alerts_list else None

                for alert in st.session_state.coastal_alerts_list:
                    email_body += f"- {alert['message']}\n"
                email_body += "\nReview your EcoVision dashboard for more details.\n\nSincerely,\nEcoVision Team"

                try:
                    send_email_alert(
                        recipient_email_coastal,
                        email_subject,
                        email_body,
                        attachment_path=map_snippet_path_coastal_for_email # Pass the saved path
                    )

                    for alert_detail in st.session_state.coastal_alerts_list:
                        alert_detail['recipient_email'] = recipient_email_coastal
                        log_alert_to_csv(alert_detail, "Coastal Erosion")

                    st.success("✅ Coastal erosion alert email sent successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to send email: {e}")
                    st.exception(e) # Display full traceback for debugging

#-----------------------------------------------------------------------------------------------------------------------

st.markdown("---")
st.markdown("### About EcoVision")
st.markdown("""
EcoVision is a smart alarm system designed to leverage satellite imagery and deep learning
to automatically monitor environmental changes like deforestation and coastal erosion.
Our goal is to provide timely and actionable alerts to relevant authorities,
aiding in conservation efforts.
""")










def detect_coastal_erosion(before_image_path, after_image_path, pixel_to_meter_ratio, min_shift_meters):
    # Get land masks from both images
    mask_before, transform_before, crs_before, _, _, _ = predict_mask(model, before_image_path, ...)
    mask_after, transform_after, crs_after, _, _, _ = predict_mask(model, after_image_path, ...)

    # Extract coastline contours
    coastline_before = extract_coastline_contour(mask_before)
    coastline_after = extract_coastline_contour(mask_after)
    
    # Calculate distance between coastlines
    dist_to_before = cv2.distanceTransform(1 - mask_before_contour, cv2.DIST_L2, 5)
    shift_distances = dist_to_before[coastline_after_pixels]
    
    # Find significant shifts
    significant_shifts = shift_distances > (min_shift_meters / pixel_to_meter_ratio)
    
    coastal_alerts = []
    if np.any(significant_shifts):
        # Calculate affected area
        affected_area = np.sum(significant_shifts)
        
        # Create alert
        avg_shift = np.mean(shift_distances[significant_shifts]) * pixel_to_meter_ratio
        alert_message = f"Coastline shifted by {avg_shift:.2f}m over {affected_area} pixels"
        coastal_alerts.append({
            'type': 'Coastal Erosion',
            'shift_meters': avg_shift,
            'area_pixels': affected_area,
            'message': alert_message
        })
    
    return overlay_image, coastal_alerts, change_map