import os
import numpy as np
import argparse
import importlib
import torch
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tool import pyutils, torchutils
from tool.GenDataset import Stage1_TrainDataset
from tool.infer_fun import infer
from tool.infer_utils import compute_acc
from tqdm import tqdm
import matplotlib.pyplot as plt

from tool.tpgf import TPGFModule
from tool.bcar import BCARModule
from retrieval.text_encoder import TextEncoder

cudnn.enabled = True

def train_phase(args):
    model = getattr(importlib.import_module(args.network), 'Net')(args.init_gama, n_class=args.n_class)
    print(vars(args))
    
    text_encoder = TextEncoder().model.cuda()
    tpgf_module = TPGFModule(in_channels=4096).cuda()
    bcar_module = BCARModule(in_channels=4096, num_classes=args.n_class, lambda_val=0.5).cuda()
    
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.RandomVerticalFlip(p=0.5),
                                  transforms.ToTensor()]) 
    train_dataset = Stage1_TrainDataset(data_path=args.trainroot,transform=transform_train, dataset=args.dataset)
    train_data_loader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    pin_memory=False,
                                    drop_last=True)
    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches
    param_groups = model.get_parameter_groups()
    param_groups[0].extend(list(tpgf_module.parameters()) + list(bcar_module.parameters()))
    
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)
    
    if args.weights[-7:] == '.params':
        assert args.network == "network.resnet38_cls"
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
        model.load_state_dict(weights_dict, strict=False)
    elif args.weights[-4:] == '.pth':
        weights_dict = torch.load(args.weights)
        model.load_state_dict(weights_dict, strict=False)
        
    model = model.cuda()
    avg_meter = pyutils.AverageMeter(
            'loss',
            'avg_ep_EM',
            'avg_ep_acc')
    timer = pyutils.Timer()
    
    for ep in range(args.max_epoches):
        model.train()
        tpgf_module.train()
        bcar_module.train()
        args.ep_index = ep
        ep_count = 0
        ep_EM = 0
        ep_acc = 0

        pbar = tqdm(train_data_loader, 
            desc=f'Epoch {ep+1}/{args.max_epoches}',
            leave=True)

        for iter_i, (filename, data, label) in enumerate(pbar):    
            img = data
            label = label.cuda(non_blocking=True)
            enable_PDA = 0
            
            B = img.size(0)
            text_embeddings = torch.randn(B, args.n_class, 4096).cuda() 
            
            feature_map, feature, y = model(img.cuda(), enable_PDA)
            
            f_txt_global = torch.mean(text_embeddings, dim=1) 
            
            filtered_feature_map, attn = tpgf_module(feature_map, f_txt_global)
            cam_base = model.fc8(filtered_feature_map) 
            cam_refined = bcar_module(filtered_feature_map, text_embeddings, cam_base)
            
            y_pred = F.adaptive_avg_pool2d(cam_refined, (1, 1)).view(B, -1)
            y_pred = torch.sigmoid(y_pred)
            
            prob = y_pred.cpu().data.numpy()
            gt = label.cpu().data.numpy()
            for num, one in enumerate(prob):
                ep_count += 1
                pass_cls = np.where(one > 0.5)[0]
                true_cls = np.where(gt[num] == 1)[0]
                if np.array_equal(pass_cls, true_cls) == True:
                    ep_EM += 1
                acc = compute_acc(pass_cls, true_cls)
                ep_acc += acc
            avg_ep_EM = round(ep_EM/ep_count, 4)
            avg_ep_acc = round(ep_acc/ep_count, 4)
            loss = F.multilabel_soft_margin_loss(y_pred, label)
            
            avg_meter.add({'loss':loss.item(),
                            'avg_ep_EM':avg_ep_EM,
                            'avg_ep_acc':avg_ep_acc,
                           })
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            timer.update_progress(optimizer.global_step / max_step)
            pbar.set_postfix({
                'Loss': f'{avg_meter.get("loss"):.4f}',
                'Acc': f'{avg_ep_acc:.4f}',
            })

        timer.update_progress(optimizer.global_step / max_step)
        print('Epoch:%2d' % (ep),
                'Iter:%5d/%5d' % (optimizer.global_step, max_step),
                'Loss:%.4f' % (avg_meter.get('loss')),
                'avg_em:%.4f' % (avg_meter.get('avg_ep_EM')),
                'avg_acc:%.4f' % (avg_meter.get('avg_ep_acc')),
                'lr: %.4f' % (optimizer.param_groups[0]['lr']), 
                'Fin:%s' % (timer.str_est_finish()),
                flush=True)
                
        if model.gama > 0.65:
            model.gama = model.gama*0.98
            
    torch.save(model.state_dict(), os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'.pth'))
    torch.save(tpgf_module.state_dict(), os.path.join(args.save_folder, 'tpgf_checkpoint.pth'))
    torch.save(bcar_module.state_dict(), os.path.join(args.save_folder, 'bcar_checkpoint.pth'))

def test_phase(args):
    model = getattr(importlib.import_module(args.network), 'Net_CAM')(n_class=args.n_class)
    model = model.cuda()
    args.weights = os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'.pth')
    weights_dict = torch.load(args.weights)
    model.load_state_dict(weights_dict, strict=False)
    model.eval()
    
    score = infer(model, args.testroot, args.n_class)
    print(score)
    torch.save(model.state_dict(), os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--max_epoches", default=20, type=int)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--n_class", default=4, type=int)
    parser.add_argument("--weights", default='init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)
    parser.add_argument("--trainroot", default='datasets/BCSS-WSSS/train/', type=str)
    parser.add_argument("--testroot", default='datasets/BCSS-WSSS/test/', type=str)
    parser.add_argument("--save_folder", default='checkpoints/',  type=str)
    parser.add_argument("--init_gama", default=1, type=float)
    parser.add_argument("--dataset", default='bcss', type=str)
    args = parser.parse_args()

    test_phase(args)