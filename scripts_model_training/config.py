import os
import pathlib

# Define the root directory of the project
# `pathlib.Path(__file__).parent.parent.absolute()` dynamically determines the absolute path of the project's root directory.
# - `__file__`: Refers to the current file's path (`config.py`).
# - `.parent.parent`: Moves two levels up from the current file's directory to the root directory of the project.
# - `.absolute()`: Converts the relative path to an absolute path.
# This ensures that the project can be run from any location without hardcoding paths.
rootDir = pathlib.Path(__file__).parent.parent.absolute()

# Configuration for logging into JSOC (Joint Science Operations Center)
# - `email`: The email address used to notify the user when data requests are processed.
# This email is required for submitting data requests to JSOC.
config = {"email": "uhek@umich.edu"}

# Define the output directory for downloaded data
# - `os.path.join`: Combines the root directory (`rootDir`) with the relative path to the "Operational Data" folder.
# - This is where all downloaded HMI (Helioseismic and Magnetic Imager) data will be stored.
# The directory structure ensures that data is organized and easy to locate.
outputPath = os.path.join(rootDir, "Input Data", "Operational Data")

# Define the HMI SHARP tiles and their corresponding segments to download
# - `hmiFiles`: A dictionary where:
#   - Keys represent the HMI data series (e.g., `hmi.sharp_cea_720s_nrt`).
#   - Values are lists of segments to download for each series.
#   - Example segments:
#     - `bitmap`: A binary map of the active region.
#     - `Br`, `Bp`, `Bt`: Magnetic field components (radial, poloidal, toroidal).
#     - `magnetogram`: Full-disk magnetogram data.
# This configuration allows the script to specify exactly which data segments are needed for processing.
hmiFiles = {
    "hmi.sharp_cea_720s_nrt": ["bitmap", "Br", "Bp", "Bt"],  # SHARP CEA data with multiple segments
    "hmi.sharp_720s_nrt": ["bitmap"],                       # SHARP data with only the bitmap segment
    "hmi.M_720s_nrt": ["magnetogram"],                      # Full-disk magnetogram data
}

# Define URLs for building HTML requests to JSOC
# - `urls`: A dictionary containing URL components for interacting with JSOC's API.
#   - `A1`: Base URL for submitting data export requests.
#   - `A2`: Additional parameters for the export request (e.g., process type, file format, notification email).
#   - `B1`: Base URL for checking the status of a submitted request.
#   - `B2`: Additional parameters for the status request (e.g., response format).
#   - `C1`: Base URL for accessing JSOC's main website.
# These URLs are used to construct API requests for downloading HMI data.
urls = {
    "A1": "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_fetch?op=exp_request&ds=",  # Data export request
    "A2": "&process=no_op&method=url&protocol=FITS&notify=",                      # Export request parameters
    "B1": "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_fetch?op=exp_status&requestid=",  # Request status check
    "B2": "&format=json",                                                        # Status request parameters
    "C1": "http://jsoc.stanford.edu",                                            # JSOC main website
}

losMagAgeLimit = 2 # In hours

