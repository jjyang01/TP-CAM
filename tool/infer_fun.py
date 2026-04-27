import imp
import json
from pdb import set_trace
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from tool import pyutils, iouutils
from PIL import Image
import torch.nn.functional as F
import os.path
import cv2
from tool import infer_utils
from tool.GenDataset import Stage1_InferDataset
from torchvision import transforms
from tool.gradcam import GradCam
from tqdm import tqdm

'''
1_train_stage1.py
2_generate_PM.py
tool/GenDataset.py
tool/gradcam.py
tool/iouutils.py
'''
def CVImageToPIL(img):
    img = img[:,:,::-1]
    img = Image.fromarray(np.uint8(img))
    return img
def PILImageToCV(img):
    img = np.asarray(img)
    img = img[:,:,::-1]
    return img

def fuse_mask_and_img(mask, img):
    mask = PILImageToCV(mask)
    img = PILImageToCV(img)
    Combine = cv2.addWeighted(mask,0.3,img,0.7,0) #使用函数将mask和img以指定的权重（mask0.3，img0.7）进行融合，生成一个组合图像
    return Combine

def infer(model, dataroot, n_class):
    model.eval()
    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))
    cam_list = []
    gt_list = []    
    bg_list = []
    transform = transforms.Compose([transforms.ToTensor()]) 
    infer_dataset = Stage1_InferDataset(data_path=os.path.join(dataroot,'img'),transform=transform)
    infer_data_loader = DataLoader(infer_dataset,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=False)
    for iter, (img_name, img_list) in tqdm(enumerate(infer_data_loader),total=len(infer_data_loader),desc="Inferring"):
        img_name = img_name[0]; 

        img_path = os.path.join(os.path.join(dataroot,'img'),img_name+'.png')
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        def _work(i, img, thr=0.25):
            with torch.no_grad():
                with torch.cuda.device(i%n_gpus):
                    cam, y = model_replicas[i%n_gpus].forward_cam(img.cuda())
                    y = y.cpu().detach().numpy().tolist()[0]
                    label = torch.tensor([1.0 if j >thr else 0.0 for j in y])
                    cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(4, 1, 1).numpy()
                    return cam, label

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list.unsqueeze(0))),
                                            batch_size=12, prefetch_size=0, processes=8)
        cam_pred = thread_pool.pop_results()
        cams = [pair[0] for pair in cam_pred]
        label = [pair[1] for pair in cam_pred][0]
        sum_cam = np.sum(cams, axis=0)
        norm_cam = (sum_cam-np.min(sum_cam)) / (np.max(sum_cam)-np.min(sum_cam))

        # cam --> segmap
        cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)
        cam_score, bg_score = infer_utils.dict2npy(cam_dict, label, orig_img, None)
        seg_map = infer_utils.cam_npy_to_label_map(cam_score)
        # if iter%100==0:
        #     print(iter) #jiumeng
        cam_list.append(seg_map)
        gt_map_path = os.path.join(os.path.join(dataroot,'mask'), img_name + '.png')
        gt_map = np.array(Image.open(gt_map_path))
        gt_list.append(gt_map)
    return iouutils.scores(gt_list, cam_list, n_class=n_class)

      
def create_pseudo_mask(model, dataroot, fm, savepath, n_class, palette, dataset):
    # print(model)
    if fm=='b4_3':
        ffmm = model.b4_3
    elif fm=='b4_5':
        ffmm = model.b4_5
    elif fm=='b5_2':
        ffmm = model.b5_2
    elif fm=='b6':
        ffmm = model.b6
    elif fm=='bn7':
        ffmm = model.bn7
    else:
        print('error')
        return
    print(dataset)
    transform = transforms.Compose([transforms.ToTensor()])
    # 修改 data_path 为 img 子目录 
    infer_dataset = Stage1_InferDataset(data_path=os.path.join(dataroot,'train/img'),transform=transform)
    infer_data_loader = DataLoader(infer_dataset,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=False)
    for iter, (img_name, img_list) in tqdm(enumerate(infer_data_loader),total=len(infer_data_loader), desc=f"{fm} processing"):
        # 增加了进度条显示      
        img_name = img_name[0]
        # 修改 img_path 为 img 子目录下的路径
        img_path = os.path.join(os.path.join(dataroot,'train/img'),img_name+'.png')
        orig_img = np.asarray(Image.open(img_path))
        grad_cam = GradCam(model=model, feature_module=ffmm, \
                target_layer_names=["1"], use_cuda=True)
        cam = []
        for i in range(n_class):
            target_category = i
            grayscale_cam, _ = grad_cam(img_list, target_category)
            cam.append(grayscale_cam)
        norm_cam = np.array(cam)
        _range = np.max(norm_cam) - np.min(norm_cam)
        norm_cam = (norm_cam - np.min(norm_cam))/_range

        # get label from json
        # 修改 json_path 为 prompt 子目录下的路径
        json_path = os.path.join(os.path.join(dataroot, 'train/prompt'), img_name + '.json')
        try:
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
                code = data['code'][0] # 获取 code 字典
                label = torch.Tensor([code['a'], code['b'], code['c'], code['d']])
        except FileNotFoundError:
            print(f"JSON file not found for {img_path}")
            continue  # 如果找不到JSON文件，跳过当前图片
        except json.JSONDecodeError:
            print(f"Error decoding JSON file for {img_path}")
            continue  # 如果JSON文件格式错误，跳过当前图片

        cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)
        cam_score, bg_score = infer_utils.dict2npy(cam_dict, label, orig_img, None) #此处加入了背景，做修改
        ##  "bg_score" is the white area generated by "cv2.threshold".
        ##  Since lungs are the main organ of the respiratory system. There are a lot of alveoli (some air sacs) serving for exchanging the oxygen and carbon dioxide, which forms some white background in WSIs.
        ##  For LUAD-HistoSeg, we uses it in the pseudo-annotation generation phase to avoid some meaningless areas to participate in the training phase of stage2.
        if dataset == 'luad':
            bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
        ##  Since the white background of images of breast cancer is meaningful (e.g. fat, etc), we do not use it for the training set of BCSS-WSSS.
        elif dataset == 'bcss':
            bg_score = np.zeros((1,224,224))
            bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
        seg_map = infer_utils.cam_npy_to_label_map(bgcam_score) 
        visualimg  = Image.fromarray(seg_map.astype(np.uint8), "P")
        visualimg.putpalette(palette)
        visualimg.save(os.path.join(savepath, img_name+'.png'), format='PNG')
    print(f"{fm} pseudo_mask done")

        # if iter%100==0:           
        #     print(iter)
