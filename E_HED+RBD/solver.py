import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam
from torch.backends import cudnn
from torchvision import transforms
from ehed_rbd import weights_init, build_modelv2
from loss import Loss
from tools.visual import Viz_visdom


class Solver(object):
    def __init__(self, train_loader, val_loader, test_dataset, config,mode):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_dataset = test_dataset
        self.config = config
        self.beta = 0.3
        self.select = [1, 2, 3, 6]
        self.device = torch.device('cuda:0')
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.mode = mode
        if self.config.mode == "train":
            self.lossfile = open("%s/logs/loss.txt" % config.save_fold, 'w')
            self.maefile = open("%s/logs/mae.txt" % config.save_fold, 'w')
        if self.config.cuda:
            cudnn.benchmark = True
            self.device = torch.device('cuda:0')
        if config.visdom:
            self.visual = Viz_visdom("DSS", 1)
        self.build_model()
        if self.config.pre_trained: self.net.load_state_dict(torch.load(self.config.pre_trained))
        if config.mode == 'train':
            self.log_output = open("%s/logs/log.txt" % config.save_fold, 'w')
        else:
            self.net.load_state_dict(torch.load(self.config.model))
            self.net.eval()
            self.test_output = open("%s/test.txt" % config.test_fold, 'w')
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            if p.requires_grad: num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def build_model(self):
        if (self.mode == 1):
            self.net = build_model().to(self.device)
        else:
            self.net = build_modelv2().to(self.device)
        if self.config.mode == 'train': self.loss = Loss().to(self.device)
        self.net.train()
        self.net.apply(weights_init)
        if self.config.load == '': self.net.base.load_state_dict(torch.load(self.config.vgg))
        if self.config.load != '': self.net.load_state_dict(torch.load(self.config.load))
        self.optimizer = Adam(self.net.parameters(), self.config.lr)
        self.print_network(self.net, 'DSS')

    # update the learning rate
    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    # evaluate MAE (for test or validation phase)
    def eval_mae(self, y_pred, y):
        return torch.abs(y_pred - y).mean()

    def eval_pr(self, y_pred, y, num):
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / y.sum()
        return prec, recall

    # validation: using resize image, and only evaluate the MAE metric
    def validation(self):
        avg_mae = 0.0
        self.net.eval()
        with torch.no_grad():
            for i, data_batch in enumerate(self.val_loader):
                images, y1, y2, y3, y4, labels = data_batch
                images, labels = images.to(self.device),  labels.to(self.device)
                y1 = y1.to(self.device)
                y2 = y2.to(self.device)
                y3 = y3.to(self.device)
                y4 = y4.to(self.device)
                prob_pred = self.net(images, y1, y2, y3, y4)
                avg_mae += self.eval_mae(prob_pred, labels).item()
        self.net.train()
        return avg_mae / len(self.val_loader)

    # test phase: using origin image size, evaluate MAE and max F_beta metrics
    def test(self, num, use_crf=False):
        if use_crf: from tools.crf_process import crf
        avg_mae, img_num = 0.0, len(self.test_dataset)
        avg_prec, avg_recall = torch.zeros(num), torch.zeros(num)
        with torch.no_grad():
            for i, (img, y1,y2,y3,y4, labels) in enumerate(self.test_dataset):
                img.show()
                images = self.transform(img).unsqueeze(0)
                y1 = self.transform(y1).unsqueeze(0)
                y2= self.transform(y2).unsqueeze(0)
                y3 = self.transform(y3).unsqueeze(0)
                y4 = self.transform(y4).unsqueeze(0)
                if(images.shape != torch.Size([1,3,256,256])):
                    continue
                labels = labels.unsqueeze(0)
                shape = labels.size()[2:]
                images = images.to(self.device)
                y1 = y1.to(self.device)
                y2 = y2.to(self.device)
                y3 = y3.to(self.device)
                y4 = y4.to(self.device)
                prob_pred = self.net(images, y1, y2, y3, y4)
                if (self.mode == 1):
                    prob_pred = torch.mean(torch.cat([prob_pred[i] for i in self.select], dim=1), dim=1, keepdim=True)
                    prob_pred = F.interpolate(prob_pred, size=shape, mode='bilinear', align_corners=True).cpu().data

                else:
                    prob_pred = F.interpolate(prob_pred, size=shape, mode='bilinear', align_corners=True).cpu().data

                if use_crf:
                    prob_pred = crf(img, prob_pred.numpy(), to_tensor=True)
                mae = self.eval_mae(prob_pred, labels)
                prec, recall = self.eval_pr(prob_pred, labels, num)
                print("[%d] mae: %.4f" % (i, mae))
                print("[%d] mae: %.4f" % (i, mae), file=self.test_output)

                #********************To present hard cases**********************************************
                """
                if (mae>0.2):
                    img.show()
                    ss = prob_pred[0][0].cpu().numpy()
                    ss = 256 * ss
                    ims1 = Image.fromarray(ss)
                    ims1.show()
                """
                avg_mae += mae
                avg_prec, avg_recall = avg_prec + prec, avg_recall + recall
        avg_mae, avg_prec, avg_recall = avg_mae / img_num, avg_prec / img_num, avg_recall / img_num
        score = (1 + self.beta ** 2) * avg_prec * avg_recall / (self.beta ** 2 * avg_prec + avg_recall)
        score[score != score] = 0
        print('average mae: %.4f, max fmeasure: %.4f' % (avg_mae, score.max()))
        print('average mae: %.4f, max fmeasure: %.4f' % (avg_mae, score.max()), file=self.test_output)

    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        best_mae = 1.0 if self.config.val else None
        self.lossfile.write("epoch\tavg_loss\n")
        self.maefile.write("epoch\tavg_mae\n")
        for epoch in range(self.config.epoch):
            loss_epoch = 0
            #learning rate decay.
            if epoch ==30:
                lr = self.config.lr
                self.update_lr(lr)
            mae = 0
            for i, data_batch in enumerate(self.train_loader):
                if (i + 1) > iter_num: break
                self.net.zero_grad()
                x, y1,y2,y3,y4, y = data_batch
                x, y1,y2,y3,y4, y = x.to(self.device), y1.to(self.device),y2.to(self.device),\
                                     y3.to(self.device),y4.to(self.device), y.to(self.device)

                y_pred = self.net(x, y1, y2 ,y3 ,y4)
                loss = self.loss(y_pred, y)
                loss.backward()
                utils.clip_grad_norm_(self.net.parameters(), self.config.clip_gradient)
                # utils.clip_grad_norm(self.loss.parameters(), self.config.clip_gradient)
                self.optimizer.step()
                loss_epoch += float(loss.item())
                tmp_mae = self.eval_mae(y_pred,y).item()
                mae += tmp_mae
                print('epoch: [%d/%d], iter: [%d/%d], loss: [%.4f], mae: [%.4f]' % (
                    epoch, self.config.epoch, i, iter_num, loss.item(),tmp_mae))
                if self.config.visdom:
                    error = OrderedDict([('loss:', loss.item())])
                    self.visual.plot_current_errors(epoch, i / iter_num, error)
            avg_loss = loss_epoch / iter_num
            self.lossfile.write("%d\t%.4f\n"%(epoch,avg_loss))
            avg_mae = mae / iter_num
            self.maefile.write("%d\t%.4f\n"%(epoch,avg_mae))
            if (epoch + 1) % self.config.epoch_show == 0:
                print('epoch: [%d/%d], epoch_loss: [%.4f]' % (epoch, self.config.epoch, loss_epoch / iter_num),
                      file=self.log_output)
                if self.config.visdom:
                    avg_err = OrderedDict([('avg_loss', loss_epoch / iter_num)])
                    self.visual.plot_current_errors(epoch, i / iter_num, avg_err, 1)
                    y_show = torch.mean(torch.cat([y_pred[i] for i in self.select], dim=1), dim=1, keepdim=True)
                    img = OrderedDict([('origin', x.cpu()[0] * self.std + self.mean), ('label', y.cpu()[0][0]),
                                       ('pred_label', y_show.cpu().data[0][0])])
                    self.visual.plot_current_img(img)

            if self.config.val and (epoch + 1) % self.config.epoch_val == 0:
                mae = self.validation()
                print('--- Best MAE: %.2f, Curr MAE: %.2f ---' % (best_mae, mae))
                print('--- Best MAE: %.2f, Curr MAE: %.2f ---' % (best_mae, mae), file=self.log_output)
                if best_mae > mae:
                    best_mae = mae
                    torch.save(self.net.state_dict(), '%s/models/mybest.pth' % self.config.save_fold)
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_fold, epoch + 1))

        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_fold)
