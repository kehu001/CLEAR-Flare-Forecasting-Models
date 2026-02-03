import numpy as np
import pandas as pd
import time
import psutil

from utilities import *

# the needed SHARP parameters
params = ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZD',
                'TOTUSJZ', 'MEANALP', 'MEANJZH', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP', 'MEANPOT',
                'TOTPOT', 'MEANSHR', 'SHRGT45', 'SIZE', 'SIZE_ACR', 'NACR', 'NPIX']

class New_SampleConstruction:
    def __init__(self):
        self.inputs_profile = []
        self.labels = []
    
    def samples_from_harp(self, flare_list, sharps_all, sharp_params=params, input_window=24, lead_window=0, forecasting_window=24):
        # input_window: the number of hours used to predict the flare
        # lead_window: the number of hours ahead to predict the flare

        # a function used to generate the input features from the HARP data by NOAA AR
        # 3 criterions:
        # 1. the +- 70 longitude limitations (already processed before)
        # 2. the missing frames if less than 5% (6 frames) in the 24-hr long time series
        # 3. The starting times of two time sequences are separated by 2 hr.

        process = psutil.Process()
        start_time = time.time()
        max_cpu_mem = 0
        
        harpnums_unique = sharps_all['HARPNUM'].unique()
        for harpnum in harpnums_unique:
            print('Processing HARPNUM:', harpnum)
            oneharp = sharps_all[sharps_all['HARPNUM'] == harpnum]
            oneharp = oneharp.sort_values(by=['T_REC'])
            # delete the duplicate rows by 'T_REC', only 3 harps have duplicate rows
            oneharp = oneharp.drop_duplicates(subset=['T_REC'], keep='first')

            ar = oneharp['NOAA_AR'].values[0]
            if ar == 0: # non valid AR number
                continue
            else:
                flares_sub = flare_list[(flare_list['noaa_ar'] == ar)]
                if flares_sub.shape[0] == 0: # no flares in the AR
                    continue
                else:
                    # fine the qualified flares by ar and peak time
                    oneharp['T_REC'] = oneharp['T_REC'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    oneharp['T_REC'] = pd.to_datetime(oneharp['T_REC'])
                    time1 = oneharp['T_REC'].min()
                    time2 = oneharp['T_REC'].max()
                    flares_sub = flares_sub[(flares_sub['peak_time'] >= time1) & (flares_sub['peak_time'] <= time2 + pd.Timedelta(str(forecasting_window) + ' hours'))]
                    if flares_sub.shape[0] == 0: # no flares in the time range
                        continue
                    else:
                        lat_min = float(oneharp['LAT_MIN'].min())
                        lat_max = float(oneharp['LAT_MAX'].max())
                        lon_min = float(oneharp['LON_MIN'].min())
                        lon_max = float(oneharp['LON_MAX'].max())
                        inputs_profile_per_harp, labels_per_harp = self.for_one_harp(oneharp, sharp_params, harpnum, flares_sub, 
                                                                                     lat_min, lat_max, lon_min, lon_max, input_window, lead_window, forecasting_window)
                        self.inputs_profile += inputs_profile_per_harp
                        self.labels += labels_per_harp
            mem_now = process.memory_info().rss / (1024 ** 2)
            max_cpu_mem = max(max_cpu_mem, mem_now)

        end_time = time.time()
        print(f"\n[Resource Summary for Sample Construction]")
        print(f"Total construction time: {end_time - start_time:.2f} seconds")
        print(f"Max CPU memory used: {max_cpu_mem:.2f} MB")
        print(f"Total samples constructed: {len(self.inputs_profile)}")
                    
        
    def for_one_harp(self, oneharp, sharp_params, harpnum, flares_sub, lat_min, lat_max, lon_min, lon_max, 
                     input_window, lead_window, forecasting_window):
        # a function used to generate the input features from the HARP data, and the matching labels from flare list
        inputs_profile_per_harp = []
        labels_per_harp = []
        # loop over the time sequence
        start_time = oneharp['T_REC'].min()
        obs_time = start_time + pd.Timedelta(str(input_window) + ' hours')

        while obs_time <= oneharp['T_REC'].max():
            vobj = oneharp[(oneharp['T_REC'] >= start_time) & (oneharp['T_REC'] < obs_time)][sharp_params]
            # check the missing frames: TODO
            vobj_profile = {'HARPNUM': int(harpnum), 'obs_time': obs_time.strftime('%Y-%m-%d %H:%M:%S'), 
                            'lat_min': lat_min, 'lat_max': lat_max, 'lon_min': lon_min, 'lon_max': lon_max,
                            'vobj': vobj.to_numpy(dtype=float).tolist()}
            pred_time = obs_time + pd.Timedelta(str(lead_window) + ' hours')
            # the flares in the [obs_time+lead_window, obs_time+lead_window+24hr]
            flare_window = flares_sub[(flares_sub['peak_time'] >= pred_time) & (flares_sub['peak_time'] <= pred_time +pd.Timedelta(str(forecasting_window) + ' hours'))]
            #if flare_window.shape[0] == 0:
                # check if satisfy the quiet sample
                #pre = flares_sub[(flares_sub['peak_time'] < pred_time) & (flares_sub['peak_time'] >= pred_time - pd.Timedelta(str(quiet_window) + ' hours'))]
                #label = "Q" if pre.shape[0] == 0 else "N" # N: does not satisfy the quiet sample condition
            #0407update: do not record Q samples
                #start_time += pd.Timedelta('2 h')
                #obs_time += pd.Timedelta('2 h')
                #continue
            if flare_window.shape[0] > 0:
                argmax = flare_window['log_intensity'].argmax()
                label = flare_window['label'].iloc[argmax]
                inputs_profile_per_harp.append(vobj_profile)
                labels_per_harp.append(label)
                start_time += pd.Timedelta('2 h')
                obs_time += pd.Timedelta('2 h')
            else:
                start_time += pd.Timedelta('2 h')
                obs_time += pd.Timedelta('2 h')
        return inputs_profile_per_harp, labels_per_harp
    
    '''def save_samples(self, save_path):
        # save the samples to a pickle file
        samples = {'inputs_profile': self.inputs_profile, 'labels': self.labels}
        with open(save_path, 'wb') as f:
            pickle.dump(samples, f)
        print('The samples are saved to:', save_path)'''

# 2025/3/13 update: no quiet samples anymore since the A/B are actually the quiet samples/background samples
# 2025/4/20 update: ingore the middle C flares to avoid the confusion with the C flares 
def get_samples(inputs_profiles, labels, purpose, time1, time2):
    # get the samples for the training, validation, and testing
    # slice by time
    obs_point_time = pd.to_datetime(pd.Series([inputs_profiles[i]['obs_time'] for i in range(len(inputs_profiles))]))
    index = ((obs_point_time >= time1) & (obs_point_time < time2)).values
    inputs_profiles = [inputs_profiles[i] for i in range(len(inputs_profiles)) if index[i]]
    labels = [labels[i] for i in range(len(labels)) if index[i]]
    mag = np.array([label[0] for label in labels])

    if purpose == "Mplus_train":
        logintensity = pd.Series(labels).apply(get_log10_intensity)
        # Instead of using the label, use the logintensity to classify the samples, avoid the middle C flares
        cut_lower = get_log10_intensity('C2.0')
        cut_upper = get_log10_intensity('C8.5')

        P_inputs = np.array([inputs_profiles[i]['vobj'] for i in range(len(labels)) if logintensity[i] >= cut_upper])
        N_inputs = np.array([inputs_profiles[i]['vobj'] for i in range(len(inputs_profiles)) if logintensity[i] <= cut_lower])
        #P_inputs = np.array([inputs_profiles[i]['vobj'] for i in range(len(inputs_profiles)) if mag[i] in ['M', 'X']])
        #N_inputs = np.array([inputs_profiles[i]['vobj'] for i in range(len(inputs_profiles)) if mag[i] in ['A', 'B']])
        return P_inputs, N_inputs
    elif purpose == "Mplus_test":
        # Foe the test set, still use the label to classify the samples
        P_inputs = np.array([inputs_profiles[i]['vobj'] for i in range(len(inputs_profiles)) if mag[i] in ['M', 'X']])
        N_inputs = np.array([inputs_profiles[i]['vobj'] for i in range(len(inputs_profiles)) if mag[i] in ['A', 'B']])
        return P_inputs, N_inputs
    elif purpose == "Cplus": #TODO: change the positve/negative samples defination
        P_inputs = np.array([inputs_profiles[i]['vobj'] for i in range(len(inputs_profiles)) if mag[i] in ['C', 'M', 'X']])
        N_inputs = np.array([inputs_profiles[i]['vobj'] for i in range(len(inputs_profiles)) if mag[i] in ['A', 'B']])
        return P_inputs, N_inputs
    #elif purpose == "StrongWeak":
        #P_inputs = np.array([inputs_profiles[i]['vobj'] for i in range(len(inputs_profiles)) if mag[i] in ['M', 'X']])
        #N_inputs = np.array([inputs_profiles[i]['vobj'] for i in range(len(inputs_profiles)) if mag[i] in ['B', 'C']])
        #return P_inputs, N_inputs
        