import argparse
import os
from dataset import get_loader
from solver import Solver


def main(config):
    if config.mode == 'train':
        train_loader = get_loader(config.train_path, config.label_path, config.img_size, config.batch_size,
                                  filename=None, num_thread=config.num_thread)
        if config.val:
            val_loader = get_loader(config.val_path, config.val_label, config.img_size, config.batch_size,
                                    filename=None, num_thread=config.num_thread)
        run = 0
        while os.path.exists("%s/run-%d" % (config.save_fold, run)): run += 1
        os.mkdir("%s/run-%d" % (config.save_fold, run))
        os.mkdir("%s/run-%d/logs" % (config.save_fold, run))
        os.mkdir("%s/run-%d/models" % (config.save_fold,run))
        config.save_fold = "%s/run-%d" % (config.save_fold, run)
        if config.val:
            train = Solver(train_loader, val_loader, None, config, config.model_type)
        else:
            train = Solver(train_loader, None, None, config, config.model_type)
        train.train()
    elif config.mode == 'test':
        test_loader = get_loader(config.test_path, config.test_label, config.img_size, config.batch_size, mode='test',
                                 filename=None, num_thread=config.num_thread)
        if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)
        test = Solver(None, None, test_loader, config, config.model_type)
        test.test(100, use_crf=config.use_crf)
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':
    #data_root = r"D:\MSRA-B"
    data_root = r"../pro3_withbrn/MSRA-B"
    vgg_path = './weights/vgg16_feat.pth'
    # # -----ECSSD dataset-----
    # train_path = os.path.join(data_root, 'ECSSD/images')
    # label_path = os.path.join(data_root, 'ECSSD/ground_truth_mask')
    #
    # val_path = os.path.join(data_root, 'ECSSD/val_images')
    # val_label = os.path.join(data_root, 'ECSSD/val_ground_truth_mask')

    # # -----MSRA-B dataset-----
    image_path = os.path.join(data_root, 'test')
    label_path = os.path.join(data_root, 'test_gt')
    #val_path = os.path.join(data_root, 'val')
    #valgt_path = os.path.join(data_root, 'val_gt')
    val_path = r"D:/ECSSD/images"
    valgt_path = r"D:/ECSSD/ground_truth_mask"
    #val_path = r"D:/HKU-IS/HKU-IS/imgs"
    #valgt_path = r"D:/HKU-IS/HKU-IS/gt"
    test_path = os.path.join(data_root, 'tt')
    test_label = os.path.join(data_root, 'tt_gt')
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=256)  # 256
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--clip_gradient', type=float, default=1.0)
    parser.add_argument('--cuda', type=bool, default=True)

    # Training settings
    parser.add_argument('--vgg', type=str, default=vgg_path)
    parser.add_argument('--train_path', type=str, default=image_path)
    parser.add_argument('--label_path', type=str, default=label_path)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)  # 8
    parser.add_argument('--val', type=bool, default=True)
    parser.add_argument('--val_path', type=str, default=image_path)
    parser.add_argument('--val_label', type=str, default=label_path)
    parser.add_argument('--num_thread', type=int, default=4)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_fold', type=str, default='./results')
    parser.add_argument('--epoch_val', type=int, default=5)
    parser.add_argument('--epoch_save', type=int, default=5)
    parser.add_argument('--epoch_show', type=int, default=1)
    parser.add_argument('--pre_trained', type=str, default=None)

    # Testing settings
    parser.add_argument('--test_path', type=str, default=val_path)
    parser.add_argument('--test_label', type=str, default=valgt_path)
    parser.add_argument('--model', type=str, default='./weights/mybest_11_27.pth')
    parser.add_argument('--test_fold', type=str, default='./results/test')
    parser.add_argument('--use_crf', type=bool, default=False)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--visdom', type=bool, default=False)
    parser.add_argument('--model_type', type=int, default=2, choices=[1,2])

    config = parser.parse_args()
    if not os.path.exists(config.save_fold): os.mkdir(config.save_fold)
    main(config)
