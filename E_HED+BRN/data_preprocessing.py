import os
import shutil
# divide the original dataset into testset and validation set

data_root = r"C:\Users\hp\Desktop\MSRA-B"
image_list = os.listdir(data_root)
test_set = []
test_val = []
val_set = []
val_val = []
tmp1, tmp2 = 0,0
for x in image_list:
    if x[-3:] == 'jpg':
        if tmp1<4:
            test_set.append(x)
            tmp1+=1
        else:
            tmp1=0
            val_set.append(x)
    else:
        if tmp2<4:
            test_val.append(x)
            tmp2+=1
        else:
            tmp2=0
            val_val.append(x)

data_loc =  r"D:\MSRA-B"
if not os.path.exists(data_loc):
    os.makedirs(data_loc)

if not os.path.exists(os.path.join(data_loc,'test')):
    os.makedirs(os.path.join(data_loc,'test'))
p = os.path.join(data_loc,'test')
for x in test_set:
    shutil.copy(os.path.join(data_root,x),os.path.join(p,x))

if not os.path.exists(os.path.join(data_loc,'test_gt')):
    os.makedirs(os.path.join(data_loc,'test_gt'))
p = os.path.join(data_loc,'test_gt')
for x in test_val:
    shutil.copy(os.path.join(data_root,x),os.path.join(p,x))

if not os.path.exists(os.path.join(data_loc,'val')):
    os.makedirs(os.path.join(data_loc,'val'))
p = os.path.join(data_loc,'val')
for x in val_set:
    shutil.copy(os.path.join(data_root,x),os.path.join(p,x))

if not os.path.exists(os.path.join(data_loc,'val_gt')):
    os.makedirs(os.path.join(data_loc,'val_gt'))
p = os.path.join(data_loc,'val_gt')
for x in val_val:
    shutil.copy(os.path.join(data_root,x),os.path.join(p,x))