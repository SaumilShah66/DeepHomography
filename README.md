# CMSC733 Project 1: MyAutoPano



## Dependencies
Python2.7
or
Python3.5
-- Tensorflow 1.14.0
-- Numpy
-- OpenCV
-- ImgUtils

## DataSet
You can download the MS-COCO dataset from (http://cocodataset.org/#home)[here]. Place your data on the Data folder alongside Code folder.

## How to run

### Phase 1
Unzip the file vasthana_p1.zip
Mode into the folder unzipped.

Open ternimal and run the below commands
```
$ cd Phase1
$ cd Code
$ python Wrapper.py 
```
With the last commond, a argument has to be passed specifying the directory of the images.

```
$ python Wrapper.py --ImageDirectory="Mention path to your directory containing images"
```
## Output image files
Program will generate various image output files in the CODE directory in Phase1 folder.
If running various image data or test cases, it is recommended to take the back-up of generated output files before running the Wrapper.py again.

### Phase 2

First go to Code directory and run gen.py to generate new data.
```
cd Code
python gen.py
```

After new data ha been generated, you can use following command to start training Supervised model.

```
python Train.py --CheckPointPath="../SupCheckpoints/" --ModelType="Sup" --MiniBatchSize=16 --LogsPath="SupLogs/"
```

Use following command to start training Unsupervised model.
```
python Train.py --CheckPointPath="../UnsupCheckpoints/" --ModelType="Unsup" --MiniBatchSize=16 --LogsPath="UnSupLogs/"
```

To run on test data, please make sure you have weights in the directory mentioned in the code. You can download weights from [here](https://drive.google.com/open?id=1_G3QWrqK-U-hNqy09AeWyurua4nZugKe)
```
python Wrapper.py --Model="Unsup" --TestNumber=1 --MaxPerturb=32
```

For supervised model
```
python Wrapper.py --Model="Sup" --TestNumber=1 --MaxPerturb=32
```

