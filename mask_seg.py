import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import importlib
from torch.backends import cudnn
cudnn.enabled = True
from tool.infer_fun import create_pseudo_mask
from tool.GenDataset import make_data_loader
from network.sync_batchnorm.replicate import patch_replication_callback
from network.deeplab import DeepLab
from tool.loss import SegmentationLosses
from tool.lr_scheduler import LR_Scheduler
from tool.saver import Saver
from tool.summaries import TensorboardSummary
from tool.metrics import Evaluator
import cv2
from PIL import Image

def grayscale_guided_multi_level_fusion(cam_bn7, cam_b5_2, cam_b4_5, img_rgb, n_classes=4, alpha=1.5, th_gray=200):
    C, H, W = cam_bn7.shape
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    is_glass_slide = (img_gray > th_gray)
    
    weights = {'bn7': 0.6, 'b5_2': 0.2, 'b4_5': 0.2}
    cams_dict = {'bn7': cam_bn7, 'b5_2': cam_b5_2, 'b4_5': cam_b4_5}
    
    M_fused = np.zeros((n_classes + 1, H, W), dtype=np.float32)
    
    for layer_name, cam in cams_dict.items():
        M_l = np.zeros((n_classes + 1, H, W), dtype=np.float32)
        for c in range(n_classes):
            cam_c = cam[c]
            min_val = np.min(cam_c)
            max_val = np.max(cam_c)
            if max_val > min_val:
                M_l[c + 1] = (cam_c - min_val) / (max_val - min_val)
            else:
                M_l[c + 1] = 0.0
        M_l[0][is_glass_slide] = alpha
        M_l[0][~is_glass_slide] = 0.0
        M_fused += weights[layer_name] * M_l
        
    Y_pseudo = np.argmax(M_fused, axis=0).astype(np.uint8)
    return Y_pseudo

def gen_masks(args):
    pass

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.saver = Saver(args)
        self.summary = TensorboardSummary('logs')
        self.writer = self.summary.create_summary()
        kwargs = {'num_workers': args.workers, 'pin_memory': False}
        self.train_loader, self.val_loader, self.test_loader = make_data_loader(args, **kwargs)
        self.nclass = args.n_class
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        model_stage1 = getattr(importlib.import_module('network.resnet38_cls'), 'Net_CAM')(n_class=4)
        resume_stage1 = 'checkpoints/stage1_checkpoint_trained_on_'+str(args.dataset)+'.pth'
        weights_dict = torch.load(resume_stage1)
        model_stage1.load_state_dict(weights_dict)
        self.model_stage1 = model_stage1.cuda()
        self.model_stage1.eval()

        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.cuda:
                W = checkpoint['state_dict']
                if not args.ft:
                    del W['decoder.last_conv.8.weight']
                    del W['decoder.last_conv.8.bias']
                self.model.module.load_state_dict(W, strict=False)
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            one = torch.ones((output.shape[0],1,224,224)).cuda()
            output = torch.cat([output,(100 * one * (target==4).unsqueeze(dim = 1))],dim = 1)

            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('epoch%2d loss: %.3f' % (epoch, train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='validation')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            pred[target==4]=4
            self.evaluator.add_batch(target, pred)

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        ious = self.evaluator.Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)

        if mIoU > self.best_pred:
            self.best_pred = mIoU
            self.saver.save_checkpoint({
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, 'stage2_checkpoint_trained_on_'+self.args.dataset+'.pth')
            
    def load_the_best_checkpoint(self):
        checkpoint = torch.load('checkpoints/stage2_checkpoint_trained_on_'+self.args.dataset+'.pth')
        self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
        
    def test(self, epoch, Is_GM):
        self.load_the_best_checkpoint()
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='test')
        test_loss = 0.0
        
        output_dir = os.path.join('output', self.args.dataset)
        pred_dir = os.path.join(output_dir, 'pred')
        comp_dir = os.path.join(output_dir, 'comp')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        if not os.path.exists(comp_dir):
            os.makedirs(comp_dir)
            
        if self.args.dataset == 'luad':
            palette = np.array([
                [205, 51, 51],  
                [0, 255, 0],    
                [65, 105, 225], 
                [255, 165, 0],  
                [255, 255, 255] 
            ])
        elif self.args.dataset == 'bcss':
            palette = np.array([
                [255, 0, 0],     
                [0, 255, 0],     
                [0, 0, 255],     
                [153, 0, 255],   
                [255, 255, 255]  
            ])
        
        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            
            img_name = sample[0]['img_name'][0] if 'img_name' in sample[0] else f'img_{i}' 

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
                if Is_GM:
                    output = self.model(image)
                    _,y_cls = self.model_stage1.forward_cam(image)
                    y_cls = y_cls.cpu().data
                    pred_cls = (y_cls > 0.1)
            pred = output.data.cpu().numpy()
            if Is_GM:
                pred = pred*(pred_cls.unsqueeze(dim=2).unsqueeze(dim=3).numpy())
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            pred[target==4]=4
            self.evaluator.add_batch(target, pred)
            
            for b in range(pred.shape[0]):
                pred_img = np.zeros((pred.shape[1], pred.shape[2], 3), dtype=np.uint8)
                for cls in range(5):  
                    pred_img[pred[b] == cls] = palette[cls]
                img_filename = f"{img_name}_pred.png"
                cv2.imwrite(os.path.join(pred_dir, img_filename), pred_img[:, :, ::-1])  

        precision = self.evaluator.Precision()
        recall = self.evaluator.Recall()
        f1_score = self.evaluator.F1_Score()
        class_precision = self.evaluator.Class_Precision()
        class_recall = self.evaluator.Class_Recall()
        class_f1 = self.evaluator.Class_F1_Score()
        
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        ious = self.evaluator.Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

def seg_phase(args):
    trainer = Trainer(args)
    for epoch in range(trainer.args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)
    trainer.test(epoch, args.Is_GM)
    trainer.writer.close()

def test_only(args):
    trainer = Trainer(args)
    trainer.test(args.epochs-1, args.Is_GM)
    trainer.writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default='checkpoints/stage1_checkpoint_trained_on_bcss.pth', type=str)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'xception', 'drn', 'mobilenet'])
    parser.add_argument('--out-stride', type=int, default=16)
    parser.add_argument('--Is_GM', type=bool, default=True)
    parser.add_argument('--dataroot', type=str, default='datasets/BCSS-WSSS/')
    parser.add_argument('--dataset', type=str, default='bcss')
    parser.add_argument('--savepath', type=str, default='checkpoints/')
    parser.add_argument('--workers', type=int, default=10, metavar='N')
    parser.add_argument('--sync-bn', type=bool, default=None)
    parser.add_argument('--freeze-bn', type=bool, default=False)
    parser.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'focal'])
    parser.add_argument('--n_class', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30, metavar='N')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('--lr-scheduler', type=str, default='poly',choices=['poly', 'step', 'cos'])
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M')
    parser.add_argument('--nesterov', action='store_true', default=False )
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--resume', type=str, default='init_weights/deeplab-resnet.pth.tar')
    parser.add_argument('--checkname', type=str, default='deeplab-resnet')
    parser.add_argument('--ft', action='store_true', default=False)
    parser.add_argument('--eval-interval', type=int, default=1)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    
    seg_phase(args)