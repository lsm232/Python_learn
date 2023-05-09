import os

wav_path=''
label_file=''

def get_wavs_labels(wav_path=wav_path,label_file=label_file):
    wav_files=[]
    for (dir_path,dirnames,filenames) in os.walk(wav_path):
        for filename in filenames:
            wav_file=os.sep.join([dir_path,filename])
            if os.stat(wav_file).st_size<24000:
                continue
            wav_files.append(wav_file)
    labels_dict={}
    with open(label_file,'rb') as f:
        for label in f:
            label=label.strip(b'/n')
            label_id=label.split(b' ',1)[0]
            label_text=label.split(b' ',1)[1]
            labels_dict[label_id.decode('ascii')]=label_text.decode('utf-8')
    labels=[]
    new_wav_files=[]
    for wav_file in wav_files:
        if os.path.basename(wav_file).split('.')[0] in labels_dict:
            new_wav_files.append(wav_file)
            labels.append(labels_dict[os.path.basename(wav_file).split('.')[0]])
    return new_wav_files,labels