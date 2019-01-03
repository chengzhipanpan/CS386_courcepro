import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms

# change the label, add boundary, to imporve the performance. in the evening.
class ImageData(data.Dataset):
    """ image dataset
    img_root:    image root (root which contain images)
    label_root:  label root (root which contains labels)
    transform:   pre-process for image
    t_transform: pre-process for label
    filename:    MSRA-B use xxx.txt to recognize train-val-test data (only for MSRA-B)
    """

    def __init__(self, img_root, label_root, saliency_root,transform, t_transform, filename=None):
        if filename is None:
            self.image_path = [os.path.join(img_root,x) for x in os.listdir(img_root)]
            #在读dataset的时候也把saliency读进去

            self.saliency1_path = [os.path.join(saliency_root, x[:-4]+"_GS.png") for x in os.listdir(img_root)]
            self.saliency2_path = [os.path.join(saliency_root, x[:-4] + "_MR_stage2.png") for x in os.listdir(img_root)]
            self.saliency3_path = [os.path.join(saliency_root, x[:-4] + "_SF.png") for x in os.listdir(img_root)]
            self.saliency4_path = [os.path.join(saliency_root, x[:-4] + "_wCtr_Optimized.png") for x in os.listdir(img_root)]
            self.label_path = [os.path.join(label_root, x[:-3]+"png") for x in os.listdir(img_root)]
        else:
            lines = [line.rstrip('\n')[:-3] for line in open(filename)]
            self.image_path = list(map(lambda x: os.path.join(img_root, x + 'jpg'), lines))
            self.label_path = list(map(lambda x: os.path.join(label_root, x + 'png'), lines))

        self.transform = transform
        self.t_transform = t_transform

    def __getitem__(self, item):
        image = Image.open(self.image_path[item])
        saliency1 = (Image.open(self.saliency1_path[item]).convert('L'))
        saliency2 = (Image.open(self.saliency2_path[item]).convert('L'))
        saliency3 = (Image.open(self.saliency3_path[item]).convert('L'))
        saliency4 = (Image.open(self.saliency4_path[item]).convert('L'))
        label = Image.open(self.label_path[item]).convert('L')
        if self.transform is not None:
            image = self.transform(image)
            saliency1 = self.transform(saliency1)
            saliency2 = self.transform(saliency2)
            saliency3 = self.transform(saliency3)
            saliency4 = self.transform(saliency4)
        if self.t_transform is not None:
            label = self.t_transform(label)
        return image, saliency1,saliency2,saliency3,saliency4,label

    def __len__(self):
        return len(self.image_path)


# get the dataloader (Note: without data augmentation)
def get_loader(img_root, label_root, saliency_root, img_size, batch_size, filename=None, mode='train', num_thread=4, pin=True):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        t_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
            #transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
        dataset = ImageData(img_root, label_root, saliency_root, transform, t_transform, filename=filename)
        data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_thread,
                                      pin_memory=pin)
        return data_loader
    else:
        t_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
        dataset = ImageData(img_root, label_root, saliency_root, None, t_transform, filename=filename)
        return dataset


if __name__ == '__main__':
    import numpy as np
    data_loc = r"D:\MSRA-B"
    img_root = os.path.join(data_loc,'test')
    label_root = os.path.join(data_loc,'test_gt')
    saliency_root = os.path.join(data_loc, 'RES')
    #filename = '/home/ace/data/MSRA-B/train_cvpr2013.txt'
    loader = get_loader(img_root, label_root, saliency_root, 224,8, filename=None, mode='train')
    for i,(image, saliency1, saliency2, saliency3, saliency4, label) in enumerate(loader):
        print(np.array(image).shape)
        break