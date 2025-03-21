# NYCU Computer Vision 2025 Sprint HW1
Student ID: 313553002  
Name: 蔡琮偉
## Introduction
 The objective of the homework is Image Classification with 100 class, use ResNet as backbone. To finish
 the task, I choose ResNet50 as the backbone considering the hardware limit and model size requirement,
 with some modification to change the original 1000 class classification to 100 class classification.
 In order to avoid overfitting, I also use some data augmentation technology, providing great im
provement to the model performance.  
 With the above method , the model has an accuracy of about 92.6% , which is beyond a strong
 baseline in the private leaderboard
## How to install
### Install the environment
`
conda env create -f environment.yml
`  
If there are torch install error, please go to https://pytorch.org/get-started/locally/ to get correct torch version  
### Pretained model and dataset download
Pretained model: https://drive.google.com/file/d/1xz5ITdQQgvOm7fJItMF445Q0rcK1Idks/view?usp=sharing  
dataset: https://drive.google.com/file/d/1fx4Z6xl5b6r4UFkBrn5l0oPEIagZxQ5u/view?usp=drive_link
### File structure
create model and data folder, put the model and unzip dataset to corresponding folder. It should look like this  
project-root/  
├── src/  
│   ├── dataset.py  
|   ├── eval.py  
|   ├── experiment.py  
|   ├── model.py  
|   ├── test.py  
|   ├── train.py  
│   ├── utils.py  
├── data/       
│   ├── test/  
│   ├── train/  
│   └── val/  
├── model/  
│   ├── pretained.pth  
├── README.md          
├── environment.yml    
└── .gitignore          
### How to use
For evaluation, change the function val's parameter to the pretained model path then run  
`
python src/eval.py
`  
For test, change the function test's parameter to the pretained model path then run  
`
python src/test.py
`  
For training, run  
`
python src/train.py
`  
## Performance snapshot
![image](https://github.com/user-attachments/assets/ea0162bd-5e79-4115-bfc1-93cab4adcb58)

