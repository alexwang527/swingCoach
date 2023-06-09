# swingCoach
Created to compare and score your golf swing against a professional golf swing at (setup, topswing, impact and finish)
![Screen Shot 2023-03-25 at 12 27 10 PM](https://user-images.githubusercontent.com/127553791/227737500-a8998d5d-bb8d-4291-9299-66208e2d2fcd.png)

## Getting Started
To get a local copy up and running follow these simple example steps.
### Setup
```
git clone https://github.com/alexwang527/swingCoach.git
```
### Install dependencies
Make sure pip is installed: https://pip.pypa.io/en/stable/installation/
```
pip install opencv-python
```
```
pip install numpy
```
```
pip install python-math
```
### Download Caffe model
Make sure to add it to the models folder
```
wget http://posefs1.perception.cs.cmu.edu/Users/tsimon/Projects/coco/data/models/mpi/pose_iter_160000.caffemodel
```
###  Run swingCoach
Add photos to images folder of your own swing at setup, topswing, impact and finish position to images folder, follow naming format ("amsetup.jpg", "amtopswing.jpg", "amimpact.jpg", "amfinish.jpg") or you can use default amatuer swing images already in images folder.
```
python3 main.py
```
Similarity scores for each swing position will be in terminal

## Referennces
https://github.com/CMU-Perceptual-Computing-Lab/openpose
