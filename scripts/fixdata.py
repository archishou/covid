import os
import csv
data_set = '/Volumes/ArchishmaanHD1/data/raw_data/Respiratory_Sound_Database/audio_and_txt_files/'
csv_path = '/Volumes/ArchishmaanHD1/data/raw_data/Respiratory_Sound_Database/patient_diagnosis.csv'
current_filename = ''
new_filename = ''
dict = {}
with open(csv_path) as fh:
    rd = csv.DictReader(fh, delimiter=',')
    for row in rd:
        dict[str(row['id'])] = str(row['diagnosis'])

for file in os.listdir(data_set):
    current_filename = str(file)
    patient_id = current_filename[:3]
    diagnosis = dict[patient_id]
    new_filename = os.path.splitext(current_filename)[0] + '_' + diagnosis + os.path.splitext(current_filename)[1]
    print(new_filename)
    os.rename(data_set + current_filename, data_set + new_filename)
