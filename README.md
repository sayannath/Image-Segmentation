# Image Segmentation of People

## Architecture
* UNet
![](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

## Dataset
Kaggle [Link](https://www.kaggle.com/dataset/b9d4e32be2f57c2901fc9c5cd5f6633be7075f4b32d73348a6d5db245f2c1934)

## Steps to Run
```
1. python3 -m pip install --user virtualenv (In Mac or Linux)
   python -m pip install --user virtualenv (In Windows) 
   
   [Make sure Python3 is installed in your system]

2. python3 -m venv env (In Mac or Linux)
   python -m venv env (In Windows) 
   
3. source env/bin/activate  (In Mac or Linux)
   .\env\Scripts\activate (In Windows) 
   
4. pip3 install -r requirements.txt (In Mac or Linux)
   pip install -r requirements.txt (In Windows)

5. python3 train.py (In Mac or Linux)
   python train.py (In Windows)

6. python3 test.py (In Mac or Linux)
   python test.py (In Windows)
```

> Formatter Used: Black
