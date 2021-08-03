This is a fully convolutional network for nuclei classification and localization. For using it, users must: 

- Downloading dateset from [here](https://www2.warwick.ac.uk/fac/sci/dcs/research/tia/data), and unzip it to root path.
- Install necessary python enviroments, including pytorch, numpy and so on.
- Data preprecessing
```
python preprocess.py
```
- Model training
```
python train.py
```
- Testing model. We also provide trained weights for valitation
```
python test.py
```
