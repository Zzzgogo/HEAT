import argparse
import copy
import os
import os.path as osp
import time
import warnings
import pdb
import random
import torch
import math
import sys
import psutil
sys.path.append(os.getcwd())
sys.path.append('../')
from mmcv import Config
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import calculate_uncertainty
from mmdet.apis.train_Lambda_vaal import train_detector_vaal
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmdet.utils.active_datasets_vaal import *
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from mmdet.utils.functions import *

from det.train import *
from torch.utils.data.sampler import SubsetRandomSampler
from vaal.vaal_helper import *
from torch.utils.data import DataLoader
from ll4al.data.sampler import SubsetSequentialSampler
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

onlyEval = False
noEval = False
showEval = False
boolScaleUnc = False
ismini = False
load_cycle = -1 # set to -1 if you don't want to load checkpoint.
resume_cycle = -1
isSave = True
# editCfg = {}
editCfg = {'uncertainty_pool2':'objectSum_scaleMax_classSum'}
clsW = False
zeroRate = 0.15
saveMaxConf = False
useMaxConf = 'False'
score_thr = 0.3
iou_thr = 0.9
print(f'clsW is {clsW}, zeroRate is {zeroRate} useMaxConf is {useMaxConf}')
print(f'score_thr is {score_thr}, iou_thr is {iou_thr}')
if load_cycle >= 0 or resume_cycle >= 0:
    print(f'Load param of {load_cycle}cycle, Resume from {resume_cycle}cycle')

def parse_args():
    base_dir = '/home/zwj/AOD_MEH_HUA/'
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path',
                        default = base_dir+'configs/_base_/Config_RetinaNet_vaal.py',
                        )
    parser.add_argument('-p', '--data-path', default='/home/zwj/my_data/', help='dataset path')
    parser.add_argument('--dataset', default='voc2007', help='dataset')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-steps', default=[16, 19], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('-e', '--vaal_epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    
    parser.add_argument('--work-dir', help='the dir to save logs and models', default='WORK_DIR')
    parser.add_argument('--uncertainty', help='uncertainty type')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--load-from', help='the checkpoint file to load from')
    parser.add_argument('--bbox-head')
    parser.add_argument('--no-validate', default=False, help='whether not to evaluate during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int, default=1,
        help='number of gpus to use (only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-ids', type=int, default=[0], nargs='+',
        help='ids of gpus to use (only applicable to non-distributed training)')
    parser.add_argument('--deterministic',action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--Unc-type', type=str)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    for seed in range(2):
        torch.set_num_threads(2)
        args = parse_args()
        cfg = Config.fromfile(args.config)
        random_seed = (seed+6)*10
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)
        cfg.seed = random_seed
        cfg.onlyEval = onlyEval
        if noEval: args.no_validate = True
        if args.bbox_head: cfg.model.bbox_head.type = args.bbox_head
        str2unc = {'SACA':'scaleAvg_classAvg', 'SSCS':'scaleSum_classSum',
                'SACS':'scaleAvg_classSum', 'SSCA':'scaleSum_classAvg'}
        if args.Unc_type:
            cfg.uncertainty_pool2 = str2unc[args.Unc_type]
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        base_dir = '/home/zwj/AOD_MEH_HUA/' 
        if args.work_dir is not None:
            cfg.work_dir = osp.join(base_dir + 'work_dirs', args.work_dir)
        elif cfg.get('work_dir', None) is None:
            cfg.work_dir = osp.join(base_dir + 'work_dirs', osp.splitext(osp.basename(args.config))[0])
        if not os.path.exists(cfg.work_dir):
            os.mkdir(cfg.work_dir)
        if 'save_dir' not in args:
            cfg.save_dir = osp.join(cfg.work_dir, 'model_save')
            if not os.path.exists(cfg.save_dir):
                os.mkdir(cfg.save_dir)
        if ismini:
            cfg.data.test = ConfigDatasetAL(ismini)
            cfg.data.val = ConfigDatasetTEST(ismini)
        if args.resume_from is not None:
            cfg.resume_from = args.resume_from
        if args.gpu_ids is not None:
            cfg.gpu_ids = args.gpu_ids
        else:
            cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(args.launcher, **cfg.dist_params)
            _, world_size = get_dist_info()
            cfg.gpu_ids = range(world_size)
        if showEval:
            cfg.evaluation.show = True
            cfg.evaluation.out_dir = 'show_dir'
        if editCfg: EditCfg(cfg, editCfg)
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
        meta = dict()
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        meta['env_info'] = env_info
        meta['config'] = cfg.pretty_text
        meta['exp_name'] = osp.basename(args.config)
        ppid = os.getppid()
        cfg.data.workers_per_gpu = 8 if psutil.Process(ppid).name() == 'bash' else 0
        print(f'cfg.data.workers_per_gpu is set to {cfg.data.workers_per_gpu}')

        X_L, X_U, X_all, all_anns = get_X_L_0_prev(cfg)
        np.save(cfg.work_dir + '/X_L_' + '0' + '.npy', X_L)
        np.save(cfg.work_dir + '/X_U_' + '0' + '.npy', X_U)
        notResumed = True
        for cycle in cfg.cycles:
            if resume_cycle >= 0 and notResumed:
                X_L, X_U = ResumeCycle(cfg, cycle, resume_cycle)
                if not isinstance(X_L, np.ndarray): continue
                notResumed = False

            logger.info(f'Current cycle is {cycle} cycle.')
            print(f'len of X_U:{len(X_U)}, X_L:{len(X_L)}')
            cfg = create_X_L_file(cfg, X_L, all_anns, cycle)
            model = build_detector(cfg.model)
            model.init_weights()
            if 'bias' in cfg.model.train_cfg:
                if cfg.model.train_cfg.bias == 'uniform':
                    N, k = model.bbox_head.num_anchors, len(list(model.bbox_head.retina_cls.parameters())[1])
                    torch.nn.init.uniform_(list(model.bbox_head.retina_cls.parameters())[1],
                                        -math.sqrt(1 / (N * k)), math.sqrt(1 / (N * k)))
                else:
                    model.bbox_head.init_cfg['override'] = \
                        {'type': 'Normal', 'name': 'retina_cls', 'std': 0.01, 'bias': cfg.model.train_cfg.bias}

            if load_cycle >= 0:
                cfg_name = osp.splitext(osp.basename(args.config))[0]
                load_name = f'{cfg.save_dir}/{cfg_name}_Cycle{load_cycle}_Epoch{cfg.runner.max_epochs}_mycode.pth'
                checkpoint = load_checkpoint(model, load_name, map_location='cpu')
                print(f'model is loaded from {load_name}')
            datasets = [build_dataset(cfg.data.train)]

            if cfg.checkpoint_config is not None:
                cfg.checkpoint_config.meta = dict(mmdet_version=__version__ + get_git_hash()[:7], CLASSES=datasets[0].CLASSES)
            model.CLASSES = datasets[0].CLASSES

            for epoch in range(cfg.outer_epoch):
                if epoch == cfg.outer_epoch - 1:
                    cfg.lr_config.step = [1000]
                else:
                    cfg.lr_config.step = [1000]
                    cfg.evaluation.interval = 100

                cfg.optimizer['lr'] = 0.001
                if epoch == 0:
                    logger.info(f'Epoch = {epoch}, First Label Set Training')
                    cfg = create_X_L_file(cfg, X_L, all_anns, cycle) # reflect results of uncertainty sampling
                    datasets = [build_dataset(cfg.data.train)]
                    cfg.total_epochs = cfg.epoch_ratio[0]
                    cfg_bak = cfg.deepcopy()
                    train_detector_vaal(model, datasets, cfg, distributed=distributed,
                                    validate=(not args.no_validate), timestamp=timestamp, meta=meta)
                    cfg = cfg_bak
                    torch.cuda.empty_cache()

                cfg.evaluation.interval = 100
                if epoch == cfg.outer_epoch - 1:
                    cfg.lr_config.step = [2]
                    cfg.evaluation.interval = cfg.epoch_ratio[0]

                # ---------- Label Set Training ---------- #
                logger.info(f'Epoch = {epoch}, Fully-Supervised Learning')
                cfg = create_X_L_file(cfg, X_L, all_anns, cycle)
                datasets = [build_dataset(cfg.data.train)]
                cfg.total_epochs = cfg.epoch_ratio[0]
                cfg_bak = cfg.deepcopy()
                eval_res = train_detector_vaal(model, datasets, cfg, distributed=distributed,
                                            validate=(not args.no_validate), timestamp=timestamp, meta=meta)
                cfg = cfg_bak
                torch.cuda.empty_cache()

            # --------------vaal training---------------#
            device = torch.device("cuda:0")
            dataset, _ = get_dataset(args.dataset, "train", get_transform(train=True), args.data_path)
            
            # num_images = len(dataset)
            # indices = list(range(num_images))
            # random.shuffle(indices)
            # labeled_set = indices[:400]
            # unlabeled_set = list(set(indices) - set(labeled_set))

            # X_L = X_L.tolist()
            unlabeled_set = list(set(X_all) - set(X_L))
            train_sampler = SubsetRandomSampler(X_L)
            unlabeled_sampler = SubsetRandomSampler(unlabeled_set)
            train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, 4, drop_last=True)
            unlabeled_batch_sampler = torch.utils.data.BatchSampler(unlabeled_sampler, 4, drop_last=True)
            labeled_dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=train_batch_sampler, num_workers=1,
                                                collate_fn=utils.collate_fn)
            unlabeled_dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=unlabeled_batch_sampler,
                                                        num_workers=1,
                                                        collate_fn=utils.collate_fn)
            
            vae = VAE()
            params = [p for p in vae.parameters() if p.requires_grad]
            vae_optimizer = torch.optim.SGD(params, lr=args.lr / 10, momentum=args.momentum,
                                            weight_decay=args.weight_decay)
            vae_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(vae_optimizer, milestones=args.lr_steps,
                                                                    gamma=args.lr_gamma)
            torch.nn.utils.clip_grad_value_(vae.parameters(), 1e5)

            vae.to(device)
            discriminator = Discriminator()
            params = [p for p in discriminator.parameters() if p.requires_grad]
            discriminator_optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                                    weight_decay=args.weight_decay)
            discriminator_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(discriminator_optimizer,
                                                                            milestones=args.lr_steps,
                                                                            gamma=args.lr_gamma)
            discriminator.to(device)
            # def read_unlabeled_data(dataloader):
            #     while True:
            #         for images, _ in dataloader:
            #             yield list(image.to(device) for image in images)
            # labeled_data = read_unlabeled_data(labeled_dataloader)
            # unlabeled_data = read_unlabeled_data(unlabeled_dataloader)
            # vae.train()
            # discriminator.train()

            #for iter in range(args.vaal_epochs):
            def train_one_epoch( vae, vae_optimizer, discriminator, discriminator_optimizer,
                            labeled_dataloader, unlabeled_dataloader, device, cycle, iter):
                def read_unlabeled_data(dataloader):
                    while True:
                        for images, _ in dataloader:
                            yield list(image.to(device) for image in images)
                labeled_data = read_unlabeled_data(labeled_dataloader)
                unlabeled_data = read_unlabeled_data(unlabeled_dataloader)
                vae.train()
                discriminator.train()
                # metric_logger = utils.MetricLogger(delimiter="  ")
                # metric_logger.add_meter('task_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
                header = 'Cycle:[{}] Epoch: [{}]'.format(cycle, iter)

                vae_lr_scheduler = None
                discriminator_lr_scheduler = None
                if iter == 0:
                    warmup_factor = 1. / 1000
                    warmup_iters = min(1000, len(labeled_dataloader) - 1)
                    vae_lr_scheduler = utils.warmup_lr_scheduler(vae_optimizer, warmup_iters, warmup_factor)
                    discriminator_lr_scheduler = utils.warmup_lr_scheduler(discriminator_optimizer, warmup_iters, warmup_factor)


                for i in range(len(labeled_dataloader)):
                    unlabeled_imgs = next(unlabeled_data)
                    labeled_imgs = next(labeled_data)
                    recon, z, mu, logvar = vae(labeled_imgs)
                    unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, 1)
                    unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
                    transductive_loss = vae_loss(unlabeled_imgs, unlab_recon, unlab_mu, unlab_logvar, 1)

                    labeled_preds = discriminator(mu)
                    unlabeled_preds = discriminator(unlab_mu)

                    lab_real_preds = torch.ones(len(labeled_imgs)).cuda()
                    unlab_real_preds = torch.ones(len(unlabeled_imgs)).cuda()

                    if not len(labeled_preds.shape) == len(lab_real_preds.shape):
                        dsc_loss = bce_loss(labeled_preds, lab_real_preds.unsqueeze(1)) + bce_loss(unlabeled_preds,
                                                                                                unlab_real_preds.unsqueeze(1))
                    else:
                        dsc_loss = bce_loss(labeled_preds, lab_real_preds) + bce_loss(unlabeled_preds, unlab_real_preds)
                    total_vae_loss = unsup_loss + transductive_loss + dsc_loss
                    vae_optimizer.zero_grad()
                    total_vae_loss.backward()
                    vae_optimizer.step()

                    # Discriminator step
                    with torch.no_grad():
                        _, _, mu, _ = vae(labeled_imgs)
                        _, _, unlab_mu, _ = vae(unlabeled_imgs)

                    labeled_preds = discriminator(mu)
                    unlabeled_preds = discriminator(unlab_mu)

                    lab_real_preds = torch.ones(len(labeled_imgs)).cuda()
                    unlab_fake_preds = torch.zeros(len(unlabeled_imgs)).cuda()

                    if not len(labeled_preds.shape) == len(lab_real_preds.shape):
                        dsc_loss = bce_loss(labeled_preds, lab_real_preds.unsqueeze(1)) + bce_loss(unlabeled_preds,
                                                                                                unlab_fake_preds.unsqueeze(1))
                    else:
                        dsc_loss = bce_loss(labeled_preds, lab_real_preds) + bce_loss(unlabeled_preds, unlab_fake_preds)
                    discriminator_optimizer.zero_grad()
                    dsc_loss.backward()
                    discriminator_optimizer.step()

                    if vae_lr_scheduler is not None:
                        vae_lr_scheduler.step()
                    if discriminator_lr_scheduler is not None:
                        discriminator_lr_scheduler.step()
                    if i == len(labeled_dataloader) - 1:
                        print(header)
                        print('vae_loss: {} dis_loss:{}'.format(total_vae_loss, dsc_loss))
                        
                
            # for iter in range(args.vaal_epochs):
            #     train_one_epoch( vae, vae_optimizer, discriminator, discriminator_optimizer,
            #             labeled_dataloader, unlabeled_dataloader, device, cycle, iter)
            #     vae_lr_scheduler.step()
            #     discriminator_lr_scheduler.step()
            
            # unlabeled_loader = DataLoader(dataset, batch_size=1, sampler=SubsetSequentialSampler(unlabeled_set),
            #                           num_workers=1, pin_memory=True, collate_fn=utils.collate_fn)
            # tobe_labeled_inds = sample_for_labeling(vae, discriminator, unlabeled_loader, cfg.X_S_size)
            # tobe_labeled_set = [unlabeled_set[i] for i in tobe_labeled_inds]



            if cycle == cfg.cycles[-1]:
                for file in os.listdir(cfg.save_dir): # To save memory
                    if not '_mycode' in file:
                        os.remove(os.path.join(cfg.save_dir, file))
                cfg_name = osp.splitext(osp.basename(args.config))[0]
                save_name = f'{cfg.save_dir}/{cfg_name}_Cycle{cycle}_Epoch{cfg.runner.max_epochs}_seed_{(seed+6)*10}_mycode.pth'
                torch.save(model.state_dict(), save_name)

            if cycle != cfg.cycles[-1]:
                # get new labeled data
                # dataset_al = build_dataset(cfg.data.test)
                # data_loader = build_dataloader(dataset_al, samples_per_gpu=cfg.data.samples_per_gpu,
                #                             workers_per_gpu=cfg.data.workers_per_gpu, dist=False, shuffle=False)
                # if not distributed:
                #     poolModel = MMDataParallel(model, device_ids=cfg.gpu_ids)
                # else:
                #     poolModel = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()],
                #                 broadcast_buffers=False)
                # torch.cuda.empty_cache()
                # with torch.no_grad():
                #     uncOuts = calculate_uncertainty(cfg, poolModel, data_loader, return_box=False, showNMS = False,
                #                                     saveUnc=False, saveMaxConf=saveMaxConf, clsW=clsW, scaleUnc=boolScaleUnc,
                #                                     score_thr = score_thr, iou_thr = iou_thr)
                # if saveMaxConf:
                #     maxconf = uncOuts[1]
                #     uncertainty = uncOuts[0]
                # else:
                #     maxconf = None
                #     uncertainty = uncOuts
                # if torch.is_tensor(uncertainty):
                #     uncertainty = uncertainty.numpy()
                # elif not isinstance(uncertainty, np.ndarray):
                #     uncertainty = torch.stack(uncertainty).numpy()
                for iter in range(args.vaal_epochs):
                    train_one_epoch( vae, vae_optimizer, discriminator, discriminator_optimizer,
                            labeled_dataloader, unlabeled_dataloader, device, cycle, iter)
                    vae_lr_scheduler.step()
                    discriminator_lr_scheduler.step()
                
                unlabeled_loader = DataLoader(dataset, batch_size=1, sampler=SubsetSequentialSampler(unlabeled_set),
                                        num_workers=1, pin_memory=True, collate_fn=utils.collate_fn)
                tobe_labeled_inds = sample_for_labeling(vae, discriminator, unlabeled_loader, cfg.X_S_size)
                tobe_labeled_set = [unlabeled_set[i] for i in tobe_labeled_inds]
               
                X_L  = update_X_L(tobe_labeled_set, X_all, X_L)

                # save set and model
                # np.save(cfg.work_dir + '/X_L_' + str(cycle + 1) + '.npy', X_L)
                # np.save(cfg.work_dir + '/X_U_' + str(cycle + 1) + '.npy', X_U)
                # np.save(cfg.work_dir + '/Unc_' + str(cycle + 1) + '.npy', uncertainty)
                torch.cuda.empty_cache()
            DelJunkSave(cfg.work_dir)
if __name__ == '__main__':
    main()
