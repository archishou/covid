import os
import csv

data_set = '/Volumes/ArchishmaanHD1/data/raw_data/Respiratory_Sound_Database/audio_and_txt_files/'
csv_path = '/Volumes/ArchishmaanHD1/data/raw_data/Respiratory_Sound_Database/file.csv'

with open(csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Patient ID", "Recording Index", "Recording Location",
                     "Acquisition Mode", "Recording Equipment", "Diagnosis"])
    for recording in os.listdir(data_set):
        current_filename = str(recording)
        file_parts = os.path.splitext(current_filename)
        if file_parts[1] == '.wav':    
            parts = file_parts[0].split('_')
            writer.writerow(parts)
