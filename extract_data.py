import json
import os
import math
import librosa

DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050         #每秒采样22050点
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION    #总采样点数

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.

        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],  #映射
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)   #将一段音频分段，每一段的采样点
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)#向上取整

    # loop through all genre sub-folder 遍历指定路径中的所有文件
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level 
        if dirpath is not dataset_path: #如果遍历到的目录是子文件夹
            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1] #将字符串按分隔符拆分为列表 [-1]取最后的一个元素  例：在/genres/blues 取blue
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all aud o files in genre sub-dir
            for f in filenames:

		# load audio file
                file_path = os.path.join(dirpath, f)    #拼接文件路径
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d     #分段音频中每一段音频的起始时间
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, d+1))

    # save MFCCs to json file
    with open(json_path, "w+") as fp:
        json.dump(data, fp, indent=4)
        
        
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
