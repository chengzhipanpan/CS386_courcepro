# CS386_courcepro
The course project for CS386
==================================
vgg16 pretrain is needed.\
Due to the file size limitation, it is not uploaded.\
The relative files is uploaded via [BaiduNetDisk](https://pan.baidu.com/s/19BsSV-ah9nwPhVkXwDlpAg).\
Also, for RBD part, RBD prior is needed. The images should be prcessed by the matlab code uploaded.

Environment
----------------
We basically implement the codes in Pytorch4.0.

Usage
----------------
Type `python3 main.py` in the console in each sub directory.

Dataset
----------------
We use MSRA-B 5k as our training dataset. We also manually divde MSRA-B into two parts, a 4000 sub part is used for training and the other 1000 is used for validation.
We also use ECSSD and HKU-IS dataset to validate the efficiency of our model.

Results
----------------
dog | bird | cat
----|------|----
foo | foo  | foo
bar | bar  | bar
baz | baz  | baz
F-score on different datasets:
Tables        | MSRA-B validation | ECSSD  | HKU-IS |
------------- |-------------------| -----  | -------|
 E-HED+RBD    |     0.9329        | 0.9268 | 0.9305 |
 E-HED+BRN    |     0.9302        | 0.9244 | 0.9267 |
MAE on different datasets:
Tables        | MSRA-B validation | ECSSD | HKU-IS |
------------- |:------------------| ----- | -------|
 E-HED+RBD    |       0.0593      |0.0414 | 0.0446 |
 E-HED+BRN    |       0.0548      |0.0408 | 0.0414 |
