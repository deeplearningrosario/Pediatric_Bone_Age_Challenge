# Deep Learning For Computer Vision
This is a first approach to Deep Learning by an investigation group from Rosario, Santa Fe, Argentina.
We used another team's [solution](https://github.com/thevishalagarwal/BoneAgeEstimation) to submerge ourselves. However, we've since modified it to suit our needs and to apply any improvements we saw fit.

The order of execution would be
1. pickle_dataset
2. main
3. check

## Dependencies needed

* [python3](https://www.python.org/)
* [tensorflow](https://www.tensorflow.org/)
* [keras](https://keras.io)
* [opencv-python](https://www.opencv.org/)
* [pandas](https://pandas.pydata.org/)
* [numpy](http://www.numpy.org/)
* [matplotlib](http://www.matplotlib.org/)
* [sklearn](http://scikit-learn.org/stable/)

Install on Linux
```bash
sudo pip install keras

python3 -mpip install opencv-python --user

python3 -mpip install pandas --user

python3 -mpip install numpy --user
```


## Download DataSet [rsna-bone-age](https://www.kaggle.com/kmader/rsna-bone-age)

1. Generate the API key

Go to the __Kaggle account__, link https://www.kaggle.com/ `your user name` /account

Click __Create New API Token__ and then save the json file in 'home' (linux users), mode info [kaggle-api](https://github.com/Kaggle/kaggle-api#api-credentials)

2. Install kaggle cli

```shell
pip install kaggle
```

3. Move our API key to kaggle path

```shell
kaggle
```

```shell
mv ./kaggle.json ~/.kaggle/kaggle.json
```
or
```shell
mv ./kaggle.json /root/.kaggle/kaggle.json
```

4. Downloader dataset

In our repository

```shell
kaggle datasets download -d kmader/rsna-bone-age -p ./
```

5. Unzip

```shell
cd ./rsna-bone-age

unzip boneage-test-dataset.zip

unzip boneage-training-dataset.zip
```

6. Move CSV  to dataset folder

```shell
mv boneage-training-dataset.csv ./boneage-training-dataset

mv boneage-test-dataset.csv ./boneage-test-dataset
```

7. Move dataset to project root folder

```shell
mv ./boneage-training-dataset ../

mv ./boneage-test-dataset ../
```

8. Result

```shell
.
├── attention_model.py
├── boneage-test-dataset
├── boneage-training-dataset
├── check_no_gender.py
├── check.py
├── dataset_sample
├── .git
├── .gitignore
├── main_no_gender.py
├── main.py
├── pickle_dataset_multiprocessing.py
├── pickle_dataset.py
├── prueba.py
├── README.md
...
```
