"""
Tool to download NRT input data in real time for operational analysis and forecasting.
This script downloads HMI data from JSOC (Joint Science Operations Center)

Developers
----------
For MagPy:
    
    Tilaye Tadesse: tilaye.t.asfaw@nasa.gov
    Ian Fernandes: fernandesi2244@gmail.com
           NASA JSC, Space Medicine Operations Division (SD2), Space Radiation Analysis Group (SRAG)

           NASA JOHNSON SPACE CENTER, HOUSTON, TX 77058


Last Updated 2021
### **Detailed Description of downloadHMI.py**

The downloadHMI.py script is a tool designed to download Near Real-Time (NRT) HMI (Helioseismic and Magnetic Imager) data from 
the Joint Science Operations Center (JSOC) for operational analysis and forecasting. It is part of the MagPy system, which is 
used for space weather forecasting and analysis.

---

### **Purpose**
The script automates the process of:
1. **Downloading HMI Data**:
   - SHARP (Spaceweather HMI Active Region Patch) files.
   - Line-of-Sight (LOS) magnetograms.
2. **Retrieving Solar Region Summary (SRS) files and NOAA events**.
3. **Ensuring Data Availability**:
   - It checks for missing data and retries downloads if necessary.

---

### **Key Features**
1. **Dynamic Time Handling**:
   - The script calculates the current UTC time and adjusts it to avoid missing data.
   - It uses a two-hour offset for SHARP and LOS data retrieval to account for potential delays in data availability.

2. **JSOC API Integration**:
   - The script interacts with the JSOC API to search for and download HMI data.
   - It supports multiple data series, including SHARP and LOS magnetograms.

3. **Retry Mechanism**:
   - If a download fails, the script retries up to a maximum number of attempts (`MAX_ATTEMPTS`).

4. **Configuration-Driven**:
   - The script uses a configuration file (`config.py`) to define email addresses, output paths, and data series.

---

### **Key Functions**

#### 1. **`currentTimeString`**
- **Purpose**:
  - Calculates the current UTC time, sets minutes and seconds to `00`, and returns the formatted time string in TAI format.
- **Usage**:
  - Standardizes the time format for downloading LOS magnetograms, SRS files, and NOAA events.
- **Output**:
  - A tuple containing the TAI time string and the current minute as an integer.

---

#### 2. **`harpTimeString`**
- **Purpose**:
  - Calculates the current UTC time and the UTC time minus two hours.
  - Returns two strings for downloading SHARP files and full-disk magnetograms.
- **Usage**:
  - Ensures that the time range for data retrieval avoids missing data by starting two hours before the current time.
- **Output**:
  - A tuple containing the start time (two hours before the current time) and the end time (10 minutes after the start time) in TAI format.

---

#### 3. **`exitRoutine`**
- **Purpose**:
  - Logs the runtime of the script and exits the program gracefully.
- **Parameters**:
  - `scriptName`: The name of the script.
  - `scriptStartTime`: The epoch time when the script started execution.
- **Usage**:
  - Called at the end of the script to log the runtime and perform cleanup.

---

#### 4. **`downloadHMIFiles`**
- **Purpose**:
  - Downloads SHARP files or LOS magnetograms from JSOC.
- **Parameters**:
  - `seriesName`: The name of the JSOC data series to download (e.g., `hmi.sharp_cea_720s_nrt`).
  - `harpTimeStart`: The start time for the data search in TAI format.
  - `harpTimeEnd`: The end time for the data search in TAI format.
- **Features**:
  - Determines the output folder based on the data series name.
  - Uses the JSOC API to search for and download data.
  - Implements a retry mechanism for failed downloads.
- **Output**:
  - Returns `True` if the download was successful, `False` otherwise.

---

#### 5. **`main`**
- **Purpose**:
  - The main function that orchestrates the entire workflow of the script.
- **Steps**:
  1. Logs the start time of the script.
  2. Calculates the current UTC time and the time range for HMI data retrieval.
  3. Iterates over the list of HMI data series defined in the configuration file.
  4. Calls `downloadHMIFiles` for each data series to download the required files.
  5. Handles exceptions and logs errors if any occur during the download process.
  6. Calls `exitRoutine` to log the runtime and exit the script.
- **Output**:
  - None. The function performs all operations and exits the script.

---

### **Workflow**

1. **Initialization**:
   - The script starts by logging the current time and calculating the time range for data retrieval.

2. **Data Series Iteration**:
   - It retrieves the list of HMI data series from the configuration file and iterates over each series.

3. **Data Download**:
   - For each data series, it calls `downloadHMIFiles` to search for and download the required files.

4. **Retry Mechanism**:
   - If a download fails, the script retries up to `MAX_ATTEMPTS` times.

---

### **Configuration**

The script relies on the `config.py` file for the following:
1. **Email Address**:
   - Used for JSOC notifications.
2. **Output Path**:
   - Specifies where the downloaded data will be stored.
3. **HMI Data Series**:
   - Defines the data series and segments to download.

---

### **Example Output**

#### Successful Execution:
```
########################################
downloadHMI.py began at 2025-05-12 14:00:00

Current UTC time: 2025.05.12_14:00:00_TAI
HMI Start time: 2025.05.12_12:00:00_TAI
HMI End time: 2025.05.12_12:10:00_TAI
----------------------------------------
Downloading HMI Data: hmi.sharp_cea_720s_nrt
----------------------------------------
Searching for HMI Data starting from 2025.05.12_12:00:00_TAI to 2025.05.12_12:10:00_TAI...
HMI files (hmi.sharp_cea_720s_nrt) successfully downloaded!
----------------------------------------
Downloading HMI Data: hmi.M_720s_nrt
----------------------------------------
Searching for HMI Data starting from 2025.05.12_12:00:00_TAI to 2025.05.12_12:10:00_TAI...
HMI files (hmi.M_720s_nrt) successfully downloaded!
########################################
downloadHMI.py ran for 120.5 seconds and ended at 2025-05-12 14:02:00
```

#### Failed Download:
```
----------------------------------------
Downloading HMI Data: hmi.sharp_cea_720s_nrt
----------------------------------------
Searching for HMI Data starting from 2025.05.12_12:00:00_TAI to 2025.05.12_12:10:00_TAI...
Caught Exception:
    FileNotFoundError('File not found')
Waiting 10 seconds to try again...
Maximum tries reached for downloading HMI data. Moving on...
```

---

### **Key Dependencies**
1. **`datetime`**:
   - Used for time calculations and formatting.
2. **`os`**:
   - Used for file path manipulations.
3. **`sunpy.net.jsoc`**:
   - Used to interact with the JSOC API for data retrieval.

---

### **Conclusion**
The downloadHMI.py script is a robust and automated tool for downloading HMI data from JSOC. 
It ensures data availability for operational analysis and forecasting while handling errors gracefully. 
The script is highly configurable and integrates seamlessly with the MagPy system.
"""

import datetime
import os
import pathlib
import re
import sys
import time

import urllib.request
from datetime import timedelta
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

import astropy.units as u
from numpy.core.numeric import outer
from sunpy.net import attrs as a
from sunpy.net import jsoc

import drms
import pandas as pd

import config

# including the 20 key parameters for model and 7 other feature 
key_list = ['HARPNUM','NOAA_AR','T_REC','LAT_MIN','LON_MIN','LAT_MAX','LON_MAX',
                'USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZD',
                'TOTUSJZ', 'MEANALP', 'MEANJZH', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP', 'MEANPOT',
                'TOTPOT', 'MEANSHR', 'SHRGT45', 'SIZE', 'SIZE_ACR', 'NACR', 'NPIX', 'QUALITY']

def currentTimeString():
    """
    Calculates the current UTC time, sets minutes and seconds to '00', 
    and returns a tuple containing the formatted TAI time string and 
    the current minute as an integer.

    Purpose:
    --------
    This function is used to standardize the time format for downloading 
    Line-of-Sight (LOS) magnetograms, Solar Region Summary (SRS) files, 
    and NOAA events.

    Returns:
    --------
    timeTuple : tuple(str, int)
        - The first element is a string representing the current UTC time 
          in TAI format (e.g., "2025.05.12_14:00:00_TAI").
        - The second element is an integer representing the current minute 
          (e.g., 45 if the current time is 14:45 UTC).
    """
    # Get the current UTC time
    dateTimeNow = datetime.datetime.utcnow()

    # Format the current time as a string in TAI format (YYYY.MM.DD_HH:00:00_TAI)
    dateTimeNowStr = datetime.datetime.strftime(dateTimeNow, "%Y.%m.%d_%H")
    dateTimeNowStr = dateTimeNowStr + ":00:00_TAI"

    # Extract the current minute as an integer
    dateMinNowInt = int(datetime.datetime.strftime(dateTimeNow, "%M"))

    # Return the formatted time string and the minute as a tuple
    timeTuple = (dateTimeNowStr, dateMinNowInt)
    return timeTuple


def harpTimeString():
    """
    Calculates the current UTC time and the UTC time minus two hours, 
    and returns two strings for downloading SHARP files and full-disk 
    magnetograms.

    Purpose:
    --------
    This function ensures that the time range for data retrieval avoids 
    missing data by starting two hours before the current time. It also 
    defines a 10-minute interval for the end time.

    Returns:
    --------
    harpTimeNowStr : str
        A string representing the start time in TAI format, which is 
        two hours before the current UTC time (e.g., "2025.05.12_12:00:00_TAI").
    harpTimeEndStr : str
        A string representing the end time in TAI format, which is 
        10 minutes after the start time (e.g., "2025.05.12_12:10:00_TAI").
    """
    # Get the current UTC time
    harpTimeNow = datetime.datetime.utcnow()

    # Subtract two hours from the current time to avoid missing data
    harpTimeMinusTwoHour = harpTimeNow - timedelta(hours=2)

    # Format the start time as a string in TAI format
    harpTimeMinusTwoHourStr = datetime.datetime.strftime(harpTimeMinusTwoHour, "%Y.%m.%d_%H")
    harpTimeNowStr = harpTimeMinusTwoHourStr + ":00:00_TAI"

    # Define the end time as 10 minutes after the start time
    harpTimeEndStr = harpTimeMinusTwoHourStr + ":10:00_TAI"

    # Return the start and end times as strings
    return harpTimeNowStr, harpTimeEndStr



def exitRoutine(scriptName, scriptStartTime):
    """
    Any final cleanup before exiting the script.

    Params:
    =======
    scriptName : str
      Str of the main file running this function
    scriptStartTime : int
      Int representing the time the function started
      in epoch secs
    """

    print(
        f"{scriptName} ran for {time.time() - scriptStartTime} seconds and ended at {datetime.datetime.now()}"
    )
    print("########################################\n\n")
    sys.exit()


'''def downloadHMIFiles(seriesName, harpTimeStart, harpTimeEnd):
    """
    Downloads SHARP files or LOS magnetograms from JSOC.

    Purpose:
    --------
    This function interacts with the JSOC API to search for and download 
    HMI data for a specified time range and data series.

    Parameters:
    -----------
    seriesName : str
        The name of the JSOC data series to download (e.g., "hmi.sharp_cea_720s_nrt").
    harpTimeStart : str
        The start time for the data search in TAI format (e.g., "2025.05.12_12:00:00_TAI").
    harpTimeEnd : str
        The end time for the data search in TAI format (e.g., "2025.05.12_12:10:00_TAI").

    Returns:
    --------
    bool
        True if the download was successful, False otherwise.
    """
    global num_tries
    MAX_ATTEMPTS = 10  # Maximum number of retry attempts
    num_tries += 1

    # Determine the output folder based on the data series name
    if seriesName == 'hmi.sharp_cea_720s_nrt':
        outputFolder = os.path.join(config.outputPath, 'HMI', 'NRT SHARPs_TEST')
    elif seriesName == 'hmi.sharp_720s_nrt':
        outputFolder = os.path.join(config.outputPath, 'HMI', 'NRT CCD SHARPs_TEST')
    elif seriesName == 'hmi.M_720s_nrt':
        outputFolder = os.path.join(config.outputPath, 'HMI', 'NRT LOS Full-Disk Magnetograms_TEST')
    else:
        # Handle unexpected seriesName values
        print(f"ERROR: Unknown seriesName '{seriesName}'. Cannot determine output folder.")
        return False

    # Retrieve the email address for JSOC notifications
    notifyAddress = config.config['email']

    # Retrieve the list of segments to download for the specified series
    segs = config.hmiFiles[seriesName]

    print(f"Searching for HMI Data starting from {harpTimeStart} to {harpTimeEnd}...")

    # Initialize the JSOC client
    client = jsoc.JSOCClient()

    # Combine all segments using bitwise AND to specify the desired data
    combinedSegs = a.jsoc.Segment(segs[0])
    for i in range(1, len(segs)):
        combinedSegs &= a.jsoc.Segment(segs[i])

    try:
        # Search for data in the specified time range and series
        response = client.search(
            a.Time(harpTimeStart, harpTimeEnd),
            a.jsoc.Series(seriesName),
            a.jsoc.Notify(notifyAddress),
            a.Sample(1 * u.hour),  # 1-hour cadence
            combinedSegs
        )

        # Download the data to the specified output folder
        client.fetch(response, path=outputFolder)
        time.sleep(1)  # Pause to clean up the output
        print(f'HMI files ({seriesName}) successfully downloaded!')

    except Exception as e:
        # Handle exceptions and retry if the maximum number of attempts has not been reached
        print('Caught Exception:')
        print('\t' + repr(e))
        if num_tries > MAX_ATTEMPTS:
            print('Maximum tries reached for downloading HMI data. Moving on...')
            return False
        print('Waiting 10 seconds to try again...')
        time.sleep(10)
        return downloadHMIFiles(seriesName, harpTimeStart, harpTimeEnd)

    return True '''

# Ke HU updated: Only SHARP key words as csv files using drms
def downloadHMIFiles(seriesName, harpTimeStart, harpTimeEnd):
    """
    Query JSOC for SHARP keyword metadata only and save as CSV (no FITS files).

    """
    #global num_tries
    #MAX_ATTEMPTS = 5  # Maximum number of retry attempts
    #num_tries += 1

    # Only allow SHARP series here
    if seriesName not in ('hmi.sharp_cea_720s_nrt', 'hmi.sharp_720s_nrt'):
        print(f"Skipping non-SHARP series: {seriesName}")
        return True
    
    # Output folder for SHARP CSVs
    outputFolder = os.path.join(config.outputPath, 'HMI', 'NRT_SHARP_Keywords')
    os.makedirs(outputFolder, exist_ok=True)

    # Build a DRMS record set for the time window
    # Example: hmi.sharp_cea_720s_nrt[][2025.11.07_01:00:00_TAI-2025.11.07_01:10:00_TAI]
    recset = f"{seriesName}[][{harpTimeStart}-{harpTimeEnd}]"

    print(f"Searching SHARP keywords via DRMS for {recset} ...")

    # Query only the keywords we want; no files are staged or downloaded
    client = drms.Client()            
    try:
        result = client.query(recset, key=key_list, rec_index=True)
        if result is None or len(result) == 0:
            print("No SHARP records returned.")
            return True

        # Convert to a DataFrame (drms returns a pandas DataFrame already in recent versions)
        df = pd.DataFrame(result)

        # Keep unique rows by (T_REC, HARPNUM) just in case
        if 'T_REC' in df.columns and 'HARPNUM' in df.columns:
            df = df.drop_duplicates(subset=['T_REC', 'HARPNUM'])
        
        # Save CSV
        stamp = harpTimeStart.replace(':','').replace('_','').replace('.','')
        csv_path = os.path.join(
            outputFolder,
            f"{seriesName.replace('.','_')}_{stamp}.csv"
        )
        df.to_csv(csv_path, index=False)
        print(f"Saved SHARP keyword table: {csv_path}")
        return csv_path

    except Exception as e:
      print('Caught Exception while querying SHARP keywords:')
      print('\t' + repr(e))
      print('Consider shorten the time range.\n')
      raise e



def main():
    """
    Main function to run the downloadHMI script.

    Purpose:
    --------
    This function handles the downloading of HMI data, SRS files, and NOAA events. 
    It also checks for the availability of necessary files for the MagPy process 
    and executes it if all conditions are met.

    Returns:
    --------
    None
    """
    global num_tries

    # Record the start time of the script
    scriptStartTime = time.time()
    scriptName = "downloadHMI.py"

    print("\n########################################")
    print(f"{scriptName} began at {datetime.datetime.now()}\n")

    # Get the current UTC time and the minute component
    currentTime, minuteInt = currentTimeString()
    print('Current UTC time:', currentTime)

    # Calculate the start and end times for HMI data retrieval
    harpTimeStart, harpTimeEnd = harpTimeString()
    print('HMI Start time:', harpTimeStart)
    print('HMI End time:', harpTimeEnd)

    # We use the next lines to download HMI data for a specific time range for historical analysis
    # Uncomment the following lines to set a specific time range for downloading HMI data inhistoric mode
    # One can also set the time range to download data for a specific date and time as a keyword argument

    # # -----------------------------------------------------------------------
    # currentTime ="2025.04.29_16:00:00_TAI" #2023-09-18-12 2023-11-14T08:59Z # 20250407_13
    # print('Current UTC time:', currentTime) # 20250414_1300
    # harpTimeStart, harpTimeEnd = harpTimeString()
    # harpTimeStart= "2025.04.29_14:00:00_TAI"
    # harpTimeEnd  = "2025.04.29_14:05:00_TAI"
    # print('HMI Start time:', harpTimeStart)
    # print('HMI End time:', harpTimeEnd)

    try:
        # only SHARP keywords in cea_nrt series
        JSOCDataSeriesEntries = ['hmi.sharp_cea_720s_nrt']
        for seriesName in JSOCDataSeriesEntries:
            print('-' * 40)
            print(f"Downloading SHARP keywords: {seriesName}")
            print('-' * 40)

            num_tries = 0
            downloadHMIFiles(seriesName, harpTimeStart, harpTimeEnd)
        
        # Retrieve the list of HMI data series to download from the configuration
        ####### JSOCDataSeriesEntries = config.hmiFiles
        '''# Iterate over each data series and download the files
        for seriesName in JSOCDataSeriesEntries:
            print('-' * 40)
            print(f"Downloading HMI Data: {seriesName}")
            print('-' * 40)

            # Reset the retry counter for each series
            num_tries = 0

            # Download the HMI data files for the current series
            downloadHMIFiles(seriesName, harpTimeStart, harpTimeEnd)'''

    except Exception as e:
        # Handle any exceptions that occur during the download process
        print(f'Exception occurred during one of the file downloads!!! Error message: {repr(e)}.')
        raise e

    # Perform final cleanup and exit the script
    exitRoutine(scriptName, scriptStartTime)
    print('-' * 40) 
    
    print('HMI files downloaded successfully!') 
    


if __name__ == "__main__":
    # Set the maximum number of attempts to download files
    num_tries = 0
    # main()
    main()  
    

    
  
    
    