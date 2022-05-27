import os
import time
from importlib_resources import path
#os.system("ffmpeg  -i source.mp3  -vn -acodec copy -ss 00:03:21 -t 00:00:41 output.mp3")
DATA_PATH="D:\\project\\program\\vscode\\ai\\EMRecognition\\dataset1\\techno"

def cut_music(data_path):
    i=0
    for root,dir,files in os.walk(data_path):
        for file in files:
            s_time=[1,0,0,0,0,0,0,0,0]
            l_time=[1,0,0,0,0,30,0,0,0]
            file_path = os.path.join(root, file)
            for _ in range(9):
                start_time=time.strftime("%H:%M:%S",tuple(s_time)) 
                last_time=time.strftime("%H:%M:%S",tuple(l_time)) 
                order = "ffmpeg -i " + file_path + " -vn -acodec copy -ss "+start_time+" -t "+last_time+ " " + str(i) + ".wav"
                os.system(order)
                i=i+1
                s_time[5]=s_time[5]+20
                if(s_time[5]==60):
                    s_time[5]=0
                    s_time[4]=s_time[4]+1
                        
def convert_flac_to_mp3(data_path):
    pass
    for root,dir,files in os.walk(data_path):
        for file in files:
            if file[-4:]=="flac":
                file_path=os.path.join(root,file)
                order = "ffmpeg -i " + file_path + " -ab 320k -map_metadata 0 -id3v2_version 3 "+file[:-5]+".mp3"
                os.system(order)

def convert_mp3_to_wav(data_path):
    for root,dir,files in os.walk(data_path):
        for file in files:
            if file[-3:]=="mp3":
                file_path=os.path.join(root,file)
                order = "ffmpeg -i " + file_path + " -f wav "+file[:-4]+".wav"
                os.system(order) 
                   
if __name__ == "__main__":
    pass
    #cut_music(DATA_PATH)            
    #convert_flac_to_mp3(DATA_PATH)
    #convert_mp3_to_wav(DATA_PATH)