import numpy as np
import pandas as pd
import mne
import torch
import os
import yasa
import time
from tqdm import tqdm
from SleepStageDetect import SleepStageModel as SleepModel
# from DataProcess import dataProcessor
from Simulation.Simuoutput import dataSim_yasa as Collector_Yasa
from Simulation.Simuoutput import dataSim as Collector_Model
import warnings

warnings.filterwarnings("ignore")


# Function Init
def timecal(useTime):
    m, s = divmod(useTime, 60)
    h, m = divmod(m, 60)
    return "%02dh %02dm %.3fs" % (h, m, s)


def All_Processing(database, indx, warmingTime, Time_Sld, C3, REog, EMG):
    now_init = time.time()

    # MASS channel init

    fs = 500
    label_stride = 30
    model_stride = 30
    yasa_stride = 30
    realtime_stride = 3

    # Sleep Stage mapping
    mapping_yasa = {"W": 0, "N1": 0, "N2": 2, "N3": 3, "R": 4}
    mapping_model = {"W/N1": 0, "N2": 2, "N3": 3, "REM": 4}

    # Paths init
    if "FL" in database:
        dataPath = "%s/%03d/%03d_eeg_data.npy" % (database, indx, indx)
        LabelPath = "%s/%03d/hypno_30s.csv" % (database, indx)
        P = "FL"
    else:
        dataPath = "%s/sub%02d_night/sub%02d_night_eeg_data.npy" % (database, indx, indx)
        LabelPath = "%s/sub%02d_night/hypno_30s.csv" % (database, indx)
        P = "IS"
    SaveLabel = "../BenchMark/%s/%03d" % (P, indx)
    if not os.path.exists(SaveLabel):
        os.mkdir(SaveLabel)
    print("Processing Database %s, index %03d\n" % (P, indx))

    # Data Loading
    print("Data loading...")

    now = time.time()
    data = np.load(dataPath)
    label = pd.read_csv(LabelPath)
    description = list(label["description"])
    total_length = []
    print("Done! Use time: %s\n" % timecal(time.time() - now))

    # Ground Truth
    print("Indexing ground truth...")
    now = time.time()
    GroundTruth = []
    for d in description:
        if "2" in d:
            temp_stage = 2
        elif "3" in d:
            temp_stage = 3
        elif "R" in d:
            temp_stage = 4
        else:
            temp_stage = 0

        for _ in range(label_stride):
            GroundTruth.append(temp_stage)
    GroundTruth = np.array(GroundTruth)
    total_length.append(len(GroundTruth))
    print("Done! Use time: %s\n" % timecal(time.time() - now))

    # Predict Offline Yasa
    print("Predicting offline data by Yasa...")
    now = time.time()
    data_C3 = data[C3]
    data_REOG = data[REog]
    data_EMG = data[EMG]
    data_yasa = [data_C3 / 1e6, data_REOG / 1e6, data_EMG / 1e6]
    info = mne.create_info(ch_names=['C3', 'REOG', "EMG"],
                           ch_types='eeg',
                           sfreq=fs)
    raw_yasa = mne.io.RawArray(data_yasa, info, verbose=False)
    sls = yasa.SleepStaging(raw_yasa, eeg_name='C3', eog_name='REOG', emg_name="EMG")
    pred_yasa = sls.predict()
    offline_yasa = []
    for p in pred_yasa:
        temp_stage = mapping_yasa[p]
        for _ in range(yasa_stride):
            offline_yasa.append(temp_stage)
    offline_yasa = np.array(offline_yasa)
    total_length.append(len(offline_yasa))
    print("Done! Use time: %s\n" % timecal(time.time() - now))

    # Predict Offline Model
    print("Predicting offline data by our model...")
    now = time.time()
    # Data Splitting and Model Init
    data_my = torch.tensor([data_C3, data_REOG, data_EMG])
    stride = model_stride * fs
    length = len(data_C3) // stride
    for T in tqdm(range(length), desc="Data Splitting"):
        onset = stride * T
        temp_dat = data_my[:, onset: stride + onset].unsqueeze(dim=0)
        spt_data = temp_dat if T == 0 else torch.cat([spt_data, temp_dat], dim=0)
    net = SleepModel()
    # Offline Predicting
    offline_model = []
    for i in tqdm(range(len(spt_data)), desc="Offline Predicting"):
        dat = spt_data[i]
        pred_model, _ = net.predict_offline(dat)
        for _ in range(model_stride):
            offline_model.append(mapping_model[pred_model])
    offline_model = np.array(offline_model)
    total_length.append(len(offline_model))
    print("Done! Use time: %s\n" % timecal(time.time() - now))

    data = None
    raw = None

    # Offline Data Saving
    length_min = min(total_length)
    GroundTruth = GroundTruth[:length_min]
    offline_yasa = offline_yasa[:length_min]
    offline_model = offline_model[:length_min]

    np.save(SaveLabel + "/GroundTruth.npy", GroundTruth)
    np.save(SaveLabel + "/Offline_yasa.npy", offline_yasa)
    np.save(SaveLabel + "/Offline_model.npy", offline_model)
    print("Saving Offline Data")

    # Realtime Predicted by Model
    print("Predicting realtime data by our model...")
    now_realtime = time.time()
    Collector_model = Collector_Model(dataPath, C3=C3, REog=REog, EMG=EMG)
    Collector_model.warmup(warmingTime)
    realtime_model = []
    usetime_model = []
    for _ in range(warmingTime):
        realtime_model.append(0)
    try:
        while True:
            now = time.time()
            dataout, (h, m, s) = Collector_model.dataout(Time_Sld, realtime_stride)

            pred, prob = net.predict_offline(dataout)

            useTime = float(time.time() - now)
            usetime_model.append(useTime)
            for _ in range(realtime_stride):
                realtime_model.append(mapping_model[pred])
            print("\r", pred, prob, "Time: %.3f seconds" % useTime, "%02d:%02d:%02d" % (h, m, s), end="")


    except BaseException:
        realtime_model = np.array(realtime_model)
        usetime_model = np.array(usetime_model)
        print()
    np.save(SaveLabel + "/Realtime_model.npy", realtime_model)
    np.save(SaveLabel + "/Realtime_model_time.npy", usetime_model)
    print("Done! Use time: %s\n" % timecal(time.time() - now_realtime))

    Collector_model = None
    net = None

    # Realtime Predicted by Yasa
    print("Predicting realtime data by yasa...")
    now_realtime = time.time()
    Collector_yasa = Collector_Yasa(dataPath, C3=C3, REog=REog)
    Collector_yasa.warmup(warmingTime)
    realtime_yasa = []
    usetime_yasa = []
    for _ in range(warmingTime):
        realtime_yasa.append(0)
    try:
        while True:
            now = time.time()

            dataout, (h, m, s) = Collector_yasa.dataout(realtime_stride)
            sls = yasa.SleepStaging(dataout, eeg_name='C3', eog_name='REOG')
            pred = sls.predict()

            useTime = float(time.time() - now)
            print("\r", pred[-1], "Use time: %.3f seconds" % useTime, "%02d:%02d:%02d" % (h, m, s), end="")
            for _ in range(realtime_stride):
                realtime_yasa.append(mapping_yasa[pred[-1]])
            usetime_yasa.append(useTime)
    except BaseException:
        realtime_yasa = np.array(realtime_yasa)
        usetime_yasa = np.array(usetime_yasa)
        print()
    np.save(SaveLabel + "/Realtime_yasa.npy", realtime_yasa)
    np.save(SaveLabel + "/Realtime_yasa_time.npy", usetime_yasa)
    print("Done! Use time: %s\n" % timecal(time.time() - now_realtime))

    useTime_all = timecal(time.time() - now_init)

    print("Database %s, index %03d done! Use time: %s\n" % (P, indx, useTime_all))


# Only Hyperparameters you will change:
warmingTime = 600
Time_Sld = 30
C3 = 2
REog = 6
EMG = 7

# database = "FL"
basePath = "G:/DataSets/CIBR_data/"
database = "IS"
index = 1

database = basePath + database
All_Processing(database, index, warmingTime, Time_Sld, C3, REog, EMG)

# for database in [2, 3, 4, 5]:
#     for indx in range(60):
#         if not os.path.exists("DataStim/SS%d/01-%02d-%04d PSG.edf" % (database, database, indx)):
#             continue
#         else:
#             All_Processing(database, indx, warmingTime, Time_Sld, C3[database-2], REog[database-2], EMG[database-2])
