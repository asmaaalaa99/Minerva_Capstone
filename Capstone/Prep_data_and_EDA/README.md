# Data Preprocessing Notebook
- [Data Preprocessing Notebook](#data-preprocessing-notebook)
  - [Introduction](#introduction)
  - [Data structure](#data-structure)
  - [How to Pre-process](#how-to-pre-process)
    - [1. Extract Frames from Videos](#1-extract-frames-from-videos)
    - [2. Crop faces in each of the images](#2-crop-faces-in-each-of-the-images)
    - [3. Build the Train and the Test datasets](#3-build-the-train-and-the-test-datasets)
## Introduction
This notebook preprocesses the videos downloaded from 
[FaceForensics++ dataset](https://github.com/ondyari/FaceForensics).  To download the dataset, you have to fill a form on the Github Repo. Once filled, a python script is sent via email to download the dataset. [FaceForensics++ dataset](https://github.com/ondyari/FaceForensics) provides data in two formats, images and videos, and three qualities, raw, c23, and c40.  Raw videos are images are the highest quality where the videos and the images did not undergo any compression. (c23) videos are raw videos that underwent medium compression and (c40) videos underwent high compression (c40). Since there is a tradeoff between data size and data quality, I decided to work with the c23 videos. Note: I worked perviously with C40 videos for the first version of this project, but the video quality was very poor, making it easy to recognize manipulated videos. The downloaded dataset has the following structure.  

## Data structure
After downloading the data, I got the following data structure. 

     📦data
    
    ┣ 📂manipulated_sequences
    
    ┃ ┣ 📂DeepFakeDetection
    
    ┃ ┃ ┣ 📂c23
    
    ┃ ┃ ┃ ┣ 📂videos
    
    ┃ ┃ ┃ ┃ ┣ 📜01_02__exit_phone_room__YVGY8LOK.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜01_02__hugging_happy__YVGY8LOK.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜01_02__meeting_serious__YVGY8LOK.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜....
    
    ┃ ┣ 📂Deepfakes
    
    ┃ ┃ ┣ 📂c23
    
    ┃ ┃ ┃ ┣ 📂videos
    
    ┃ ┃ ┃ ┃ ┣ 📜000_003.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜001_870.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜002_006.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜....
    
    ┃ ┣ 📂Face2Face
    
    ┃ ┃ ┣ 📂c23
    
    ┃ ┃ ┃ ┣ 📂videos
    
    ┃ ┃ ┃ ┃ ┣ 📜000_003.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜001_870.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜002_006.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜....
    
    ┃ ┣ 📂FaceShifter
    
    ┃ ┃ ┗ 📂c23
    
    ┃ ┃ ┃ ┗ 📂videos
    
    ┃ ┃ ┃ ┃ ┣ 📜000_003.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜001_870.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜002_006.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜....
    
    ┃ ┣ 📂FaceSwap
    
    ┃ ┃ ┗ 📂c23
    
    ┃ ┃ ┃ ┗ 📂videos
    
    ┃ ┃ ┃ ┃ ┣ 📜000_003.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜001_870.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜002_006.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜....
    
    ┣ 📂original_sequences
    
    ┃ ┣ 📂actors
    
    ┃ ┃ ┣ 📂c23
    
    ┃ ┃ ┃ ┣ 📂videos
    
    ┃ ┃ ┃ ┃ ┣ 📜01__exit_phone_room.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜01__hugging_happy.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜01__kitchen_pan.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜....
    
    ┃ ┣ 📂youtube
    
    ┃ ┃ ┣ 📂c23
    
    ┃ ┃ ┃ ┣ 📂videos
    
    ┃ ┃ ┃ ┃ ┣ 📜000.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜001.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜002.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜....

## How to Pre-process 
The pre-processing part is split into three sections.
### 1. Extract Frames from Videos 
After downloading the dataset, I used [extract_compressed_videos.py](https://github.com/ondyari/FaceForensics/blob/master/dataset/extract_compressed_videos.py)   to extract video frames, images, from all of the videos. The script uses the folder directory mentioned above to create image folders for each video. Here is an example of how each video has its folder of image.

     📦data
    
    ┣ 📂manipulated_sequences
    
    ┃ ┣ 📂DeepFakeDetection
    
    ┃ ┃ ┣ 📂c23
    
    ┃ ┃ ┃ ┣ 📂videos
    
    ┃ ┃ ┃ ┃ ┣ 📜01_02__exit_phone_room__YVGY8LOK.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜01_02__hugging_happy__YVGY8LOK.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜01_02__meeting_serious__YVGY8LOK.mp4
    
    ┃ ┃ ┃ ┃ ┣ 📜....
    
    ┃ ┃ ┃ ┣ 📂images
    
    ┃ ┃ ┃ ┃ ┗ 📂01_02__exit_phone_room__YVGY8LOK
    
    ┃ ┃ ┃ ┃ ┃ ┣ 📜01_02_1.png
    
    ┃ ┃ ┃ ┃ ┃ ┣ 📜01_02_1_2.png
    
    ┃ ┃ ┃ ┃ ┃ ┣ 📜....
    
    ┃ ┃ ┃ ┃ ┗ 📂01_02__hugging_happy__YVGY8LOK
    
    ┃ ┃ ┃ ┃ ┃ ┣ 📜....
    
    ┃ ┃ ┃ ┃ ┗ 📂01_02__meeting_serious__YVGY8LOK
    
    ┃ ┃ ┃ ┃ ┃ ┣ 📜....
    
    ┃ ┃ ┃ ┃ ┣ 📂....
    
    ┃ ┃ ┃ ┃ ┃ ┣ 📜....

### 2. Crop faces in each of the images 
I used Multi-task Cascaded Convolutional Neural Networks (MTCNN) implemented in PyTorch to extract the faces from every image. To do so, I used  [facenet-pytorch](https://github.com/timesler/facenet-pytorch).  To ensure that we have information about the images, I changed the image names to include the dataset name, video name, and frame number using the following line of code,

    f'{dataset}_{filename}{i}.jpg' for i in range(len(frames) 
 which generated the following image name

    youtube_00090_2.jpg

### 3. Build the Train and the Test datasets
To build the train and test datasets, I generated a Comma-separated values file (csv) that contains the video ids and their labels. Based on the ids, I split the dataframe into test and train. After using the ids, balanced the number of ids for each class, ensuring there is a balance between fake and real images. Then, I created a base_dir that has the train and the validation datasets, and it also contains two subfolders for each class: real and fake. Finally, each image is moved its corresponding folder for further use. I followed this structure to utilize Keras's image generator functionality that uses batches of images to save on the memory used during the training process. 

    📦base_dir
    
    ┣ 📂train_dir
    
    ┃ ┣ Fake
    
    ┃ ┃ ┣ 📜Deepfakes_00070_2.jpg
    
    ┃ ┃ ┣ 📜Deepfakes_00110.jpg
    
    ┃ ┃ ┣ 📜Deepfakes_00120_2.jpg
    
    ┃ ┃ ┣ 📜....
    

    ┃ ┣ 📂Real
    
    ┃ ┃ ┣ 📜youtube_00080_2.jpg
    
    ┃ ┃ ┣ 📜youtube_00100.jpg
    
    ┃ ┃ ┗ 📜youtube_05490.jpg
    
    ┃ ┃ ┣ 📜....
    
      
    
    ┣ 📂val_dir
    
    ┃ ┣ 📂Fake
    
    ┃ ┃ ┣ 📜Deepfakes_00020.jpg
    
    ┃ ┃ ┣ 📜Deepfakes_00790.jpg
    
    ┃ ┃ ┣ 📜Deepfakes_01070.jpg
    
    ┃ ┃ ┣ 📜....
    
    ┃ ┣ 📂Real
    
    ┃ ┃ ┣ 📜youtube_00090_2.jpg
    
    ┃ ┃ ┣ 📜youtube_00480.jpg
    
    ┃ ┃ ┣ 📜....





