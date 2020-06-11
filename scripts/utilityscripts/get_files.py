import os

data_set = '/Volumes/ArchishmaanHD1/data/raw_data/Respiratory_Sound_Database/audio_and_txt_files/'

for file in os.listdir(data_set):
    if file.endswith(".wav") and ("Al" in file or 'Ar' in file or 'Pr' in file or 'Pl' in file) and 'COPD' in file and 'AKGC417L' in file:
        print(file)
