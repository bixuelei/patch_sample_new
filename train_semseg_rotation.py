"""
@Author: bixuelei
@Contact: bxueleibi@gmial.com
@File: train_semseg.py
@Time: 2022/1/10 7:49 PM
"""

#from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from dataloader import *
from model_rotation import *
import numpy as np
from torch.utils.data import DataLoader
from util import *
# from display import *
from torch.utils.tensorboard import SummaryWriter
from plyfile import PlyData, PlyElement
import time
import shutil
import datetime
from tqdm import tqdm
from torchsummary import summary
from display import *



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
classes = ['clamping_system', 'cover', 'gear_container', 'charger', 'bottom', 'side_bolt','cover_bolt']                                                                                                                               
labels2categories={i:cls for i,cls in enumerate(classes)}       #dictionary for labels2categories



def _init_(add_string,change):
    if not os.path.exists('outputs'):       #initial the file, if not exiting (os.path.exists() is pointed at ralative position and current cwd)
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.model+'/'+args.exp):
        os.makedirs('outputs/'+args.model+'/'+args.exp)
    if not os.path.exists('outputs/'+args.model+'/'+args.exp+'/'+change+add_string+'/'+'models'):
        os.makedirs('outputs/'+args.model+'/'+args.exp+'/'+change+add_string+'/'+'models') 


          
def train(args, io):
    NUM_POINT=args.npoints
    print("start loading training data ...")
    TRAIN_DATASET = MotorDataset(split='train', data_root=args.root, num_points=NUM_POINT, bolt_weight=args.bolt_weight, test_area=args.validation_symbol, sample_rate=1.0, transform=None)
    print("start loading test data ...")
    TEST_DATASET = MotorDataset_validation(split='test', data_root=args.root, num_points=NUM_POINT, bolt_weight=args.bolt_weight, test_area=args.validation_symbol, sample_rate=1.0, transform=None)
    train_loader = DataLoader(TRAIN_DATASET, num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True,worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    test_loader = DataLoader(TEST_DATASET, num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    io.cprint(args)
    #Try to load models
    tmp=torch.cuda.max_memory_allocated()
    if args.model == 'pointnet':
        model = POINTPLUS(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_semseg(args).to(device)
    elif args.model == 'PCT':
        model = PCT_semseg(args).to(device)
    elif args.model == 'PCT_patch':
        model = PCT_patch_semseg(args).to(device)
    else:
        raise Exception("Not implemented")
    #summary(model,input_size=(3,4096),batch_size=1,device='cuda')

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-5)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.1, args.epochs)
    
    ## if finetune is true, the the best_finetune will be cosidered first, then best.pth will be taken into consideration
    if args.finetune:
        if os.path.exists(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+"/models/best_finetune.pth"):
            checkpoint = torch.load(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+"/models/best_finetune.pth")
            print('Use pretrain finetune model to finetune')
            
        elif os.path.exists(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+"/models/best.pth"):
            checkpoint = torch.load(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+"/models/best.pth")
            print('Use pretrain model to finetune')
        else:
            print('no exiting pretrained model to finetune')
            exit(-1)
        if not os.path.exists(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+"/models/best_finetune.pth") and os.path.exists(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+"/models/best.pth"):
            start_epoch = 0
        else:
            start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        
    else:
        if os.path.exists(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+"/models/best.pth"):
            checkpoint = torch.load(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+"/models/best.pth")
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            print('Use pretrain model')
        else:
            start_epoch=0
            print('no exiting pretrained model,starting from scratch')



    criterion = cal_loss
    criterion2= mean_loss
    loss_cluster=nn.MSELoss()
    NUM_CLASS=7
    best_iou = 0
    best_bolts_iou=0
    weights__ = torch.Tensor(TRAIN_DATASET.labelweights).cuda()
    persentige = torch.Tensor(TRAIN_DATASET.persentage).cuda()
    io.cprint(persentige)
    # num_eachtype__ = TRAIN_DATASET.num_eachtype__
    # print(num_eachtype__)
    # num_eachtype = TRAIN_DATASET.num_eachtype
    # print(num_eachtype)
    scale=weights__*persentige
    scale=1/scale.sum()
    # weights__*=scale
    # weights__+=1
    # weights=torch.zeros([7,7],dtype=torch.float)
    # for i in range(7):
    #     weights[i][i]=weights__[i]*scale
    # weights__*=scale
    if not args.use_weigth:
        for i in range(7):
            weights__[i]=1
    # weights=weights.cuda()
    print(weights__[5])
    print(weights__[6])
    for epoch in range(start_epoch,args.epochs):
        ####################
        # Train
        ####################
        num_batches=len(train_loader)
        total_correct=0
        total_seen=0
        loss_sum=0
        model=model.train()
        args.training=True
        total_correct_class__ = [0 for _ in range(NUM_CLASS)]
        total_iou_deno_class__ = [0 for _ in range(NUM_CLASS)]
        for i,(points,target) in tqdm(enumerate(train_loader),total=len(train_loader),smoothing=0.9):
            points, target = points.to(device), target.to(device)       #(batch_size, num_points, features)    (batch_size, num_points)
            points=normalize_data(points)                               #[bs,4096,3]
            points,GT=rotate_per_batch(points)
            # Visuell_PointCloud_per_batch_according_to_label(points,target)
            points = points.permute(0, 2, 1)                            #(batch_size,features,numpoints)
            batch_size = points.size()[0]
            opt.zero_grad()
            seg_pred,trans,result,goal,mask = model(points.float(),target)        #(batch_size, class_categories, num_points)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()                      #(batch_size,num_points, class_categories)
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()              #array(batch_size*num_points)            loss = criterion(seg_pred.view(-1, NUM_CLASS), target.view(-1,1).squeeze(),weights,using_weight=args.use_weigth)     #a scalar
            loss = criterion(seg_pred.view(-1, NUM_CLASS), target.view(-1,1).squeeze(),weights__,using_weight=args.use_weigth)     #a scalar
            loss=loss+criterion2(result.view(-1, 3),goal.view(-1, 3),mask)*args.factor_cluster
            loss = loss+feature_transform_reguliarzer(trans)*args.factor_trans
            loss.backward()
            opt.step()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASS)                    # (batch_size*num_points , num_class)
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()                     # array(batch_size*num_points)
            correct = np.sum(pred_choice == batch_label)                            # when a np arraies have same shape, a==b means in conrrespondending position it equals to one,when they are identical
            total_correct += correct
            total_seen += (batch_size * NUM_POINT)
            loss_sum += loss
            for l in range(NUM_CLASS):
                total_correct_class__[l] += np.sum((pred_choice == l) & (batch_label == l))
                total_iou_deno_class__[l] += np.sum(((pred_choice == l) | (batch_label == l)))
        mIoU__ = np.mean(np.array(total_correct_class__) / (np.array(total_iou_deno_class__, dtype=np.float64) + 1e-6))

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        outstr = 'Train %d, loss: %.6f, train acc: %.6f ' % (epoch,(loss_sum / num_batches),(total_correct / float(total_seen)))
        io.cprint(outstr)
        writer.add_scalar('Training loss', (loss_sum / num_batches), epoch)
        writer.add_scalar('Training accuracy', (total_correct / float(total_seen)), epoch)
        writer.add_scalar('Training mean ioU', (mIoU__), epoch)
        writer.add_scalar('Training loU of cover_bolt',(total_correct_class__[6] / float(total_iou_deno_class__[6])), epoch)
        writer.add_scalar('Training loU of bolt',((total_correct_class__[6]+total_correct_class__[5]) / (float(total_iou_deno_class__[6])+float(total_iou_deno_class__[5]))), epoch)
        ####################
        # Validation
        ####################
        with torch.no_grad():         
            num_batches = len(test_loader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASS)
            total_seen_class = [0 for _ in range(NUM_CLASS)]
            total_correct_class = [0 for _ in range(NUM_CLASS)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASS)]
            noBG_seen_class = [0 for _ in range(NUM_CLASS-1)]
            noBG_correct_class = [0 for _ in range(NUM_CLASS-1)]
            noBG_iou_deno_class = [0 for _ in range(NUM_CLASS-1)]
            model=model.eval()
            args.training=False

            for i,(points,seg) in tqdm(enumerate(test_loader),total=len(test_loader),smoothing=0.9):
                points, seg = points.to(device), seg.to(device)
                points=normalize_data(points)
                points,GT=rotate_per_batch(points)
                points = points.permute(0, 2, 1)
                batch_size = points.size()[0]
                seg_pred_,trans_,_,_,_= model(points.float(),seg)
                seg_pred_ = seg_pred_.permute(0, 2, 1).contiguous()
                batch_label = seg.view(-1, 1)[:, 0].cpu().data.numpy()   #array(batch_size*num_points)
                loss = criterion(seg_pred_.view(-1, NUM_CLASS), seg.view(-1,1).squeeze(),weights__,using_weight=args.use_weigth)     #a scalar
                loss = loss+feature_transform_reguliarzer(trans_)*args.factor_trans
                seg_pred_ = seg_pred_.contiguous().view(-1, NUM_CLASS)   # (batch_size*num_points , num_class)
                pred_choice = seg_pred_.cpu().data.max(1)[1].numpy()  #array(batch_size*num_points)
                correct = np.sum(pred_choice == batch_label)
                total_correct += correct
                total_seen += (batch_size * NUM_POINT)
                loss_sum+=loss
                tmp, _ = np.histogram(batch_label, range(NUM_CLASS + 1))
                labelweights += tmp
                for l in range(NUM_CLASS):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_choice == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_choice == l) | (batch_label == l)))


               ####### calculate without Background ##############
                for l in range(1, NUM_CLASS):
                    noBG_seen_class[l-1] += np.sum((batch_label == l))
                    noBG_correct_class[l-1] += np.sum((pred_choice == l) & (batch_label == l))
                    noBG_iou_deno_class[l-1] += np.sum(((pred_choice == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6))

            outstr = 'Validation with backgroud----epoch: %d,  eval loss %.6f,  eval mIoU %.6f,  eval point acc %.6f, eval avg class acc %.6f' % (epoch,(loss_sum / num_batches),mIoU,
                                                        (total_correct / float(total_seen)),(np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))))

            io.cprint(outstr)
            noBG_mIoU = np.mean(np.array(noBG_correct_class) / (np.array(noBG_iou_deno_class, dtype=np.float64) + 1e-6))
            outstr_without_background='Validation without backgroud----epoch: %d, mIoU %.6f,  eval point acc: %.6f, eval avg class acc: %.6f' % (epoch,noBG_mIoU,
                                                        (sum(noBG_correct_class) / float(sum(noBG_seen_class))),(np.mean(np.array(noBG_correct_class) / (np.array(noBG_seen_class, dtype=np.float64) + 1e-6))))
            io.cprint(outstr_without_background)

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASS):
                iou_per_class_str += 'class %s percentage: %.4f, loss_weight: %.4f, IoU: %.4f \n' % (
                    labels2categories[l] + ' ' * (16 - len(labels2categories[l])), labelweights[l],weights__[l],
                    total_correct_class[l] / float(total_iou_deno_class[l]))
            io.cprint(iou_per_class_str)

            if mIoU >= best_iou:
                best_iou = mIoU
                if args.finetune:
                    savepath = str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+"/models/best_finetune_m.pth"
                else:
                    savepath = str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+"/models/best_m.pth"

                state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }
                io.cprint('Saving best model at %s' % savepath)
                torch.save(state, savepath)
            io.cprint('Best mIoU: %f' % best_iou)
            # cur_bolts_iou=(total_correct_class[5]+total_correct_class[6])/ (float(total_iou_deno_class[5])+ float(total_iou_deno_class[6]))
            cur_bolts_iou=total_correct_class[6] / float(total_iou_deno_class[6])
            if cur_bolts_iou >= best_bolts_iou:
                best_bolts_iou=cur_bolts_iou
                if args.finetune:
                    savepath = str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+"/models/best_finetune.pth"
                else:
                    savepath = str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+"/models/best.pth"
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }
                io.cprint('Saving best model at %s' % savepath)
                torch.save(state, savepath)
            io.cprint('Best IoU of bolt: %f' % best_bolts_iou)
            io.cprint('\n\n')
        writer.add_scalar('learning rate',opt.param_groups[0]['lr'],  epoch)
        writer.add_scalar('Validation loss', (loss_sum / num_batches), epoch)
        writer.add_scalar('Validation accuracy', (total_correct / float(total_seen)), epoch)
        writer.add_scalar('validation mean ioU', (mIoU), epoch)
        writer.add_scalar('Validation loU of bolt',(total_correct_class[5]+total_correct_class[6])/ (float(total_iou_deno_class[5])+ float(total_iou_deno_class[6])), epoch)
        writer.add_scalar('Validation loU of cover_bolt',(total_correct_class[6] / float(total_iou_deno_class[6])), epoch)

    io.close()

def test(args, io):
    NUM_POINT=args.npoints
    print("start loading test data ...")
    TEST_DATASET = MotorDataset_patch(split='Test', data_root=args.root, num_points=NUM_POINT, bolt_weight=args.bolt_weight, test_area=args.test_symbol, sample_rate=1.0, transform=None)
    test_loader = DataLoader(TEST_DATASET, num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'dgcnn':
        model = DGCNN_semseg(args).to(device)
    elif args.model == 'pointnet':
        model = POINTPLUS(args).to(device)
    elif args.model == 'PCT':
        model = PCT_semseg(args).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    print("Let's test and use", torch.cuda.device_count(), "GPUs!")


    

    try:
        if args.finetune:
            if os.path.exists(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+"/models/best_finetune.pth"):
                checkpoint = torch.load(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+"/models/best_finetune.pth")

        else:
            checkpoint = torch.load(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+"/models/best.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        print('No existing model, a trained model is needed')
        exit(-1)


    criterion = cal_loss
    NUM_CLASS=7

    ####################
    # Test
    ####################
    with torch.no_grad():         
        num_batches = len(test_loader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        labelweights = np.zeros(NUM_CLASS)
        total_seen_class = [0 for _ in range(NUM_CLASS)]
        total_correct_class = [0 for _ in range(NUM_CLASS)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASS)]
        noBG_seen_class = [0 for _ in range(NUM_CLASS-1)]
        noBG_correct_class = [0 for _ in range(NUM_CLASS-1)]
        noBG_iou_deno_class = [0 for _ in range(NUM_CLASS-1)]
        model=model.eval()

        for i,(data,seg) in tqdm(enumerate(test_loader),total=len(test_loader),smoothing=0.9):
            data, seg = data.to(device), seg.to(device)
            data=normalize_data(data)
            data,GT=rotate_per_batch(data)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred,trans,_,_ = model(data,seg)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            batch_label = seg.view(-1, 1)[:, 0].cpu().data.numpy()   #array(batch_size*num_points)
            # loss = criterion(seg_pred.view(-1, NUM_CLASS), seg.view(-1,1).squeeze(),weights,using_weight=args.use_weigth)     #a scalar
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASS)   # (batch_size*num_points , num_class)
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()  #array(batch_size*num_points)
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (batch_size * NUM_POINT)
            # loss_sum+=loss
            tmp, _ = np.histogram(batch_label, range(NUM_CLASS + 1))
            labelweights += tmp


            for l in range(NUM_CLASS):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_choice == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_choice == l) | (batch_label == l)))


            ####### calculate without Background ##############
            for l in range(1, NUM_CLASS):
                noBG_seen_class[l-1] += np.sum((batch_label == l))
                noBG_correct_class[l-1] += np.sum((pred_choice == l) & (batch_label == l))
                noBG_iou_deno_class[l-1] += np.sum(((pred_choice == l) | (batch_label == l)))

        labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6))

        outstr = 'Test with backgroud: mIoU %.6f,  Test point acc %.6f, Test avg class acc %.6f' % (mIoU,
                                                    (total_correct / float(total_seen)),(np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))))

        io.cprint(outstr)
        noBG_mIoU = np.mean(np.array(noBG_correct_class) / (np.array(noBG_iou_deno_class, dtype=np.float64) + 1e-6))
        outstr_without_background='Test without backgroud----mIoU %.6f,  Test point acc: %.6f, Test avg class acc: %.6f' % (noBG_mIoU,
                                                    (sum(noBG_correct_class) / float(sum(noBG_seen_class))),(np.mean(np.array(noBG_correct_class) / (np.array(noBG_seen_class, dtype=np.float64) + 1e-6))))
        io.cprint(outstr_without_background)

        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASS):
            iou_per_class_str += 'class %s percentage: %.4f, IoU: %.4f \n' % (
                labels2categories[l] + ' ' * (16 - len(labels2categories[l])), labelweights[l],
                total_correct_class[l] / float(total_iou_deno_class[l]))
        io.cprint(iou_per_class_str)
        io.cprint(iou_per_class_str)
        io.cprint('\n\n')


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Semantic Segmentation')
    parser.add_argument('--model', type=str, default='PCT_patch', metavar='N',
                        choices=['pointnet','dgcnn','PCT','PCT_patch'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--batch_size', type=int, default=2, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--root', type=str, default='/home/bi/study/thesis/data/test', 
                        help='file need to be tested')
    parser.add_argument('--exp', type=str, default='test', metavar='N',
                        help='experiment version to record reslut')
    parser.add_argument('--change', type=str, default='hhh', metavar='N',
                        help='experiment version to record reslut')
    parser.add_argument('--finetune', type=bool, default=False, metavar='N',
                        help='if we finetune the model')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--training', type=bool,  default=True,
                        help='evaluate the model')
    parser.add_argument('--factor_cluster', type=float, default=0.2, metavar='F',
                        help='factor of loss_cluster')
    parser.add_argument('--factor_trans', type=float, default=0.01, metavar='F',
                        help='factor of loss_cluster')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_weigth', type=bool, default=False,
                        help='enables using weights')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--npoints', type=int, default=2048, 
                        help='Point Number [default: 4096]')
    parser.add_argument('--validation_symbol', type=str, default='Validation', 
                        help='Which datablocks to use for validation')
    parser.add_argument('--test_symbol', type=str, default='Test', 
                        help='Which datablocks to use for test')
    parser.add_argument('--bolt_weight', type=float, default=15, 
                        help='Training weight of bolts before init [default: 1.0]')
    parser.add_argument('--num_heads', type=int, default=4, metavar='num_attention_heads',
                        help='number of attention_heads for self_attention ')
    parser.add_argument('--num_layers', type=int, default=1, metavar='num_attention_hea',
                        help='number of attention_heads for self_attention ')
    parser.add_argument('--self_encoder_latent_features', type=int, default=128, metavar='hidden_size',
                        help='number of hidden_size for self_attention ')
    parser.add_argument('--hidden_size', type=int, default=512, metavar='hidden_size',
                        help='number of hidden_size for self_attention ')
    args = parser.parse_args()

    if args.finetune==True:
        add_string='_finetune'
    else:
        add_string=''
    _init_(add_string,args.change)


    if not args.eval:
        if not os.path.exists(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+'/0'):
            os.makedirs(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+'/0')
            path=str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+'/0'
            writer = SummaryWriter(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+'/0')
            io = PrintLog(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+'/0'+'/run'+add_string+'.log')
        else:
            i=1
            while(True):
                if not os.path.exists(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+'/'+str(i)):
                    os.makedirs(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+'/'+str(i))
                    path=str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+'/'+str(i)
                    writer = SummaryWriter(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+'/'+str(i))
                    io = PrintLog(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+'/'+str(i)+'/run'+add_string+'.log')
                    break
                else:
                    i+=1
    else:
        i=0
        path=str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+'/'+str(i)
        while os.path.exists(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+'/'+str(i)):
            path=str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+'/'+str(i)
            io_test = PrintLog(str(BASE_DIR)+"/outputs/"+args.model+'/'+args.exp+'/'+args.change+add_string+'/'+str(i)+'/Test'+add_string+'.log')
            i+=1 

    os.system('cp train_semseg_rotation.py '+path+'/'+'train_semseg_rotation.py.backup')
    os.system('cp util.py '+path+'/'+'util.py.backup')
    os.system('cp model_rotation.py '+path+'/'+'model_rotation.py.backup')
    os.system('cp dataloader.py '+path+'/'+'dataloader.py.backup')
    os.system('cp train.sh '+path+'/'+'train.sh.backup')


    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if not args.eval:
        if args.cuda:
            io.cprint(
                'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
            torch.cuda.manual_seed(args.seed)
        else:
            io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io_test)
