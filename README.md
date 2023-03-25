# swingCoach
App created to compare and score your golf swing against a professional golf swing at (setup, topswing, impact and finish)

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
```
python3 main.py
```
Similarity scores for each swing position will be in terminal
