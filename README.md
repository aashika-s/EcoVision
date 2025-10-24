EcoVision: Smart Alarm System for Environmental Change
EcoVision is an advanced change detection application built with Python, Streamlit, Google Earth Engine (GEE), and Deep Learning (U-Net segmentation models). It monitors specific geographical areas for critical environmental changes, such as deforestation and coastal erosion, and generates timely alerts via email.

Features
Dual-Module Analysis: Specialized modes for Deforestation Detection (using RGB + NIR bands) and Coastal Erosion/Shift Analysis (using water/land segmentation).

Satellite Imagery: Fetches recent, cloud-free imagery from Sentinel-2 via the Google Earth Engine API.

Deep Learning Segmentation: Uses pre-trained U-Net models for pixel-level land cover segmentation.

Geospatial Analysis: Calculates change area (in sq. km) and identifies the geographic coordinates of alerts.

Alerting System: Sends email notifications with coordinates, area details, and an attached map visualization.

Interactive Interface: Hosted via a modern Streamlit web application.

Setup and Installation
Follow these steps to get EcoVision running locally.

1. Prerequisites
You must have the following installed:

Python 3.8+

Git (for cloning and Git LFS)

Google Earth Engine (GEE) Account: You must have a registered GEE account.
