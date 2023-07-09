import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
import datetime
import pytz
import torch, gc
import wandb
import numpy as np
gc.collect()
torch.cuda.empty_cache()
import albumentations as A



### 저장 폴더명 생성
kst = pytz.timezone('Asia/Seoul')
now = datetime.datetime.now(kst)
folder_name = now.strftime('%Y-%m-%d-%H-%M-%S')

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../data/medical'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_epoch', type=int, default=15)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, ignore_tags, optim, augmentation=None):    
                    
    dataset = SceneTextDataset(
        data_dir,
        ufo_split='train',
        split='train',
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # dataset = SceneTextDataset(
    #     data_dir,
    #     ufo_split='kfold_valid',
    #     split='train',
    #     image_size=image_size,
    #     crop_size=input_size,
    #     ignore_tags=ignore_tags
    # )
    # dataset = EASTDataset(dataset)
    # valid_num_batches = math.ceil(len(dataset) / 4)
    # valid_loader = DataLoader(
    #     dataset,
    #     batch_size=4,
    #     shuffle=True,
    #     num_workers=num_workers
    # )
    
    model_dir = os.path.join(model_dir, folder_name)
    print(model_dir)
                    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.load_state_dict(torch.load('/opt/ml/input/code/trained_models/2023-05-26-13-31-30/epoch_140.pth')) 
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 3, 2*(max_epoch // 3)], gamma=0.1)

    ###
    # scaler = torch.cuda.amp.GradScaler(growth_interval=100)
    best_loss = np.inf
    for epoch in range(max_epoch):
        ###
        model.train()
        
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                ###
                # with torch.cuda.amp.autocast(enabled=False):
                #     loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                # scaler.scale(loss).backward()
                # scaler.unscale_(optimizer)
                # max_norm = 2.
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                # scaler.step(optimizer)
                # scaler.update()
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_train = loss.item()
                epoch_loss += loss_train

                pbar.update(1)
                train_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(train_dict)

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))
        wandb.log({'train_loss': epoch_loss/num_batches, 'learning_rate': optimizer.param_groups[0]['lr']})

        
        if epoch_loss < best_loss:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)
            ckpt_fpath = osp.join(model_dir, f'best.pth')
            print("best model updated! saving...")
            torch.save(model.state_dict(), ckpt_fpath)
            best_loss=epoch_loss
            
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)
            ckpt_fpath = osp.join(model_dir, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)

        # ### validation
        # if (epoch + 1) % 2 == 0:
        #     with torch.no_grad():
        #         model.eval()
        #         epoch_loss, epoch_start = 0, time.time()
        #         with tqdm(total=valid_num_batches) as pbar:
        #             for img, gt_score_map, gt_geo_map, roi_mask in valid_loader:
        #                 pbar.set_description('[Valid Epoch {}]'.format(epoch + 1))
        
        #                 loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
        
        #                 loss_valid = loss.item()
        #                 epoch_loss += loss_valid
        
        #                 pbar.update(1)
        #                 valid_dict = {
        #                     'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
        #                     'IoU loss': extra_info['iou_loss']
        #                 }
        #                 pbar.set_postfix(valid_dict)
                
        #         print('Mean loss: {:.4f} | Elapsed time: {}'.format(
        #             epoch_loss / valid_num_batches, timedelta(seconds=time.time() - epoch_start)))
        #         wandb.log({'valid_loss': epoch_loss/valid_num_batches})


def main(args):
    wandb.init(project='project2_Betty', entity='cv-06')
    wandb.run.name = folder_name + '_clean15epochs'
    wandb.config.update(args)
    do_training(**args.__dict__)
    wandb.finish()

if __name__ == '__main__':
    args = parse_args()
    main(args)
