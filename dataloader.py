"""
@Author: bixuelei
@Contact: bxueleibi@gmial.com
@File: dataloader.py
@Time: 2022/1/16 3:49 PM
"""

import os
import numpy as np
import random
from numpy.random import choice
from tqdm import tqdm           #used to display the circulation position, to see where the code is running at
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
# from display import Visuell_PointCloud
from pointnet_util import index_points#,Visuell,Visuell__
import torch
# from kmeans_pytorch import kmeans


####### normalize the point cloud############
def pc_normalize(pc):
    centroid=np.mean(pc,axis=0)
    pc=pc-centroid
    max_distance=np.sqrt(np.max(np.sum(pc**2,axis=1)))
    pc=pc/max_distance
    return pc


def Get_ObjectID(x) :    #get all kinds of ObjectID from numpy file

    dic = []
    for i in range(x.shape[0]):
        if x[i][6] not in dic:
            dic.append(x[i][6])

    return dic


def densify_blots(patch_motor):
    add=[]
    for i in range(len(patch_motor)):
        if (patch_motor[i][6]==6 or patch_motor[i][6]==5):
            add.append(patch_motor[i])
    add=np.array(add)
    twonn=NearestNeighbors(n_neighbors=2,algorithm='ball_tree').fit(add[:,0:3])
    _,indices=twonn.kneighbors(add[:,0:3])
    inter=[]
    for i in range(indices.shape[0]):
        interpolation=np.zeros(7)
        interpolation[3:7]=add[i][3:7]
        #if the bolt points are closest to eachonter
        if(indices[indices[i][1]][1]==i):
            interpolation[0:3]=add[i][0:3]+(add[indices[i][1]][0:3]-add[i][0:3])/3
            inter.append(interpolation)
        else:
            interpolation[0:3]=add[i][0:3]+(add[indices[i][1]][0:3]-add[i][0:3])/2
            inter.append(interpolation)
    patch_motor=np.concatenate((patch_motor,inter),axis=0)
    return patch_motor



def knn(x, k):
    """
    Input:
        points: input points data, [B, N, C]
    Return:
        idx: sample index data, [B, N, K]
    """
    # x=x.permute(0,2,1)
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]            # (batch_size, num_points, k)
    return idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist



def find_goals_kmeans(points__,target__):  # [bs,n_points,C] [bs,n_points]
    # target__=target__.astype(np.float32)
    # points = torch.from_numpy(points).cuda()
    # target= torch.from_numpy(target).cuda()
    goals_=[4,2,1,1,1,2,5]
    bs=points__.shape[0]
    mask=torch.ones((bs,16))
    # # device=points.device
    cover_bolts=torch.zeros(bs,goals_[6],3)
    side_bolts=torch.zeros(bs,goals_[5],3)
    bottoms=torch.zeros(bs,goals_[4],3)
    chargers=torch.zeros(bs,goals_[2],3)
    gearcontainers=torch.zeros(bs,goals_[3],3)
    covers=torch.zeros(bs,goals_[1],3)
    clampingsystem=torch.zeros(bs,goals_[0],3)
    for i in range(bs):
        target=torch.from_numpy(target__[i])
        points=torch.from_numpy(points__[i])
        points=points[:,0:3]
        index_clampingsystem=target==0
        points1=points[index_clampingsystem,:]
        # _,c1=KMeans(points1,goals_[1])
#         _,c1=kmeans(
#     X=points1, num_clusters=goals_[0], iter_limit = 300,distance='euclidean', device=torch.device('cuda')
# )
        _,c1=kmeans(X=points1, num_clusters=goals_[0],distance='euclidean', device=torch.device('cuda'))
#         c1=c1.cuda()
        #######################
        # Visuell(points1,c1)
        # together1=torch.cat([c1,points1],dim=0).unsqueeze(0)
        # index1=knn(together1.permute(0, 2, 1),2)
        # inter=index1[0,0:goals_[0],1]
        # Visuell__(together1.squeeze(0),inter)
        #########################
        # sqr1=square_distance(c1.unsqueeze(0),points1.unsqueeze(0))
        # together1=torch.cat([c1,points1],dim=0).unsqueeze(0)
        # index1=knn(together1.permute(0, 2, 1),2)[0,0:goals_[0],1].unsqueeze(0)
        # added1=index_points(together1,index1)
        dis1=square_distance(points1.unsqueeze(0).float(),c1.unsqueeze(0).float())
        index1=dis1.squeeze(0).max(dim=0, keepdim=False)[1]
        added1=index_points(points1.unsqueeze(0),index1.unsqueeze(0))
        _,sorted_index_clamping=added1.squeeze(0).sort(dim=0)
        sorted_index_clamping_x=sorted_index_clamping[:,0]
        goals_clamping=added1[:,sorted_index_clamping_x,:]
        clampingsystem[i,:,:]=goals_clamping[0,:,:]



        index_cover=target==1
        points2=points[index_cover,:]
        # _,c2=KMeans(points2,goals_[1])
        _,c2=kmeans(X=points2, num_clusters=goals_[1], distance='euclidean', device=torch.device('cuda'))
        # c2=c2.cuda()
        #######################
        # Visuell(points2,c2)
        # together2=torch.cat([c2,points2],dim=0).unsqueeze(0)
        # index2=knn(together2.permute(0, 2, 1),2)
        # inter=index2[0,0:goals_[1],1]
        # Visuell__(together2.squeeze(0),inter)
        ###########################
        dis2=square_distance(points2.unsqueeze(0).float(),c2.unsqueeze(0).float())
        index2=dis2.squeeze(0).max(dim=0, keepdim=False)[1]
        added2=index_points(points2.unsqueeze(0),index2.unsqueeze(0))
        # together2=torch.cat([c2,points2],dim=0).unsqueeze(0)
        # index2=knn(together2.permute(0, 2, 1),2)[0,0:goals_[1],1].unsqueeze(0)
        # added2=index_points(together2,index2)
        _,sorted_index_cover=added2.squeeze(0).sort(dim=0)
        sorted_index_cover_x=sorted_index_cover[:,0]
        goals_cover=added2[:,sorted_index_cover_x,:]
        covers[i,:,:]=goals_cover[0,:,:]

        
        index_gearcontainer=target==2
        if torch.sum(index_gearcontainer)>=1:
            points_gear=points[index_gearcontainer,:]
            points_gear_c=torch.sum(points_gear,dim=0)/torch.sum(points_gear,dim=0)

            dis3=square_distance(points_gear.unsqueeze(0).float(),points_gear_c.unsqueeze(0).unsqueeze(0).float())
            index3=dis3.squeeze(0).max(dim=0, keepdim=False)[1]
            added3=index_points(points_gear.unsqueeze(0),index3.unsqueeze(0))

            gearcontainers[i,:,:]=added3[0,:,:]
        else:
            gearcontainers[i,:,:]=torch.zeros((goals_[2],3))
            mask[i][6]=0    



        index_charger=target==3
        if torch.sum(index_charger)>=1:
            points_charger=points[index_charger,:]
            points_charger_c=torch.sum(points_charger,dim=0)/torch.sum(points_charger,dim=0)

            dis4=square_distance(points_charger.unsqueeze(0).float(),points_charger_c.unsqueeze(0).unsqueeze(0).float())
            index4=dis4.squeeze(0).max(dim=0, keepdim=False)[1]
            added4=index_points(points_charger.unsqueeze(0),index4.unsqueeze(0))

            chargers[i,:,:]=added4[0,:,:]
        else:
            chargers[i,:,:]=torch.zeros((goals_[2],3))
            mask[i][7]=0           

        index_bottom=target==4
        points_bottom=points[index_bottom,:]
        points_bottom_c=torch.sum(points_bottom,dim=0)/torch.sum(points_bottom,dim=0)
        # together5=torch.cat([points_bottom,points_bottom_c],dim=0).unsqueeze(0)

        dis5=square_distance(points_bottom.unsqueeze(0).float(),points_bottom_c.unsqueeze(0).unsqueeze(0).float())
        index5=dis5.squeeze(0).max(dim=0, keepdim=False)[1]
        added5=index_points(points_bottom.unsqueeze(0),index5.unsqueeze(0))
        # index5=knn(together5.permute(0, 2, 1),2)[0,0:goals_[4],1].unsqueeze(0)
        # added5=index_points(together5,index5)
        bottoms[i,:,:]=added5[0,:,:]

        # index_gearcontainer=target==2
        # if points[i,index_gearcontainer,:].unsqueeze(0).shape[1]!=0:
        #     points3=points[i,index_gearcontainer,:]
        #     # _,c3=KMeans(points3,goals_[2])
        #     _,c3=kmeans(X=points3, num_clusters=goals_[2], distance='euclidean', device=torch.device('cuda'))
        #     c3=c3.cuda()
        #     ########################
        #     # Visuell(points3,c3)
        #     # together3=torch.cat([c3,points3],dim=0).unsqueeze(0)
        #     # index3=knn(together3.permute(0, 2, 1),2)
        #     # inter=index3[0,0:goals_[2],1]
        #     # Visuell__(together3.squeeze(0),inter)
        #     ############################
        #     together3=torch.cat([c3,points3],dim=0).unsqueeze(0)
        #     index3=knn(together3.permute(0, 2, 1),2)[0,0:goals_[2],1].unsqueeze(0)
        #     added3=index_points(together3,index3)
        #     _,sorted_index_gearcontainer=added3.squeeze(0).sort(dim=0)
        #     sorted_index_gearcontainer_x=sorted_index_gearcontainer[:,0]
        #     goals_gearcontainer=added3[:,sorted_index_gearcontainer_x,:]
        #     gearcontainers[i,:,:]=goals_gearcontainer[0,:,:]
        # else:
        #     gearcontainers[i,:,:]=torch.zeros((goals_[2],3))
        #     mask[i][6]=0
            



        # index_charger=target[i,:]==3
        # if points[i,index_charger,:].unsqueeze(0).shape[1]!=0:
        #     points4=points[i,index_charger,:]
        #     # _,c4=KMeans(points4,goals_[3])
        #     _,c4=kmeans(X=points4, num_clusters=goals_[3], distance='euclidean', device=torch.device('cuda'))
        #     c4=c4.cuda()
        #     ########################
        #     # Visuell(points4,c4)
        #     # together4=torch.cat([c4,points4],dim=0).unsqueeze(0)
        #     # index4=knn(together4.permute(0, 2, 1),2)
        #     # inter=index4[0,0:goals_[3],1]
        #     # Visuell__(together4.squeeze(0),inter)
        #     ############################
        #     together4=torch.cat([c4,points4],dim=0).unsqueeze(0)
        #     index4=knn(together4.permute(0, 2, 1),2)[0,0:goals_[3],1].unsqueeze(0)
        #     added4=index_points(together4,index4)
        #     _,sorted_index_charger=added4.squeeze(0).sort(dim=0)
        #     sorted_index_charger_x=sorted_index_charger[:,0]
        #     goals_charger=added4[:,sorted_index_charger_x,:]
        #     chargers[i,:,:]=goals_charger[0,:,:]
        # else:
        #     chargers[i,:,:]=torch.zeros((goals_[3],3))
        #     mask[i][7]=0



        # index_bottom=target[i,:]==4
        # num5=torch.sum(index_bottom,dim=0)
        # if num5>=goals_[4]:
        #     points5=points[i,index_bottom,:]
        #     # _,c5=KMeans(points5,goals_[4])
        #     _,c5=kmeans(X=points5, num_clusters=goals_[4], distance='euclidean', device=torch.device('cuda'))
        #     c5=c5.cuda()
        #     ########################
        #     # Visuell(points5,c5)
        #     # together5=torch.cat([c5,points5],dim=0).unsqueeze(0)
        #     # index5=knn(together5.permute(0, 2, 1),2)
        #     # inter=index5[0,0:goals_[4],1]
        #     # Visuell__(together5.squeeze(0),inter)
        #     ############################
        #     together5=torch.cat([c5,points5],dim=0).unsqueeze(0)
        #     index5=knn(together5.permute(0, 2, 1),2)[0,0:goals_[4],1].unsqueeze(0)
        #     added5=index_points(together5,index5)
        #     _,sorted_index_bottom=added5.squeeze(0).sort(dim=0)
        #     sorted_index_bottom_x=sorted_index_bottom[:,0]
        #     goals_bottom=added5[:,sorted_index_bottom_x,:]
        #     bottoms[i,:,:]=goals_bottom[0,:,:]
        # else:
        #     bottoms[i,:,:]=torch.zeros((goals_[4],3))
        #     mask[i][8]=0


        index_side_bolts=target==5
        if torch.sum(index_side_bolts)>=2:
            points6=points[index_side_bolts,:]
            # _,c6=KMeans(points6,goals_[5],Niter=20)
            _,c6=kmeans(X=points6, num_clusters=goals_[5], distance='euclidean', device=torch.device('cpu'))
            # c6=c6.cuda()
            ########################
            # Visuell(points6,c6)
            # together6=torch.cat([c6,points6],dim=0).unsqueeze(0)
            # index6=knn(together6.permute(0, 2, 1),2)
            # inter=index6[0,0:goals_[5],1]
            # Visuell__(together6.squeeze(0),inter)
            ############################
            together6=torch.cat([c6,points6],dim=0).unsqueeze(0)
            index6=knn(together6.permute(0, 2, 1),2)[0,0:goals_[5],1].unsqueeze(0)
            added6=index_points(together6,index6)
            _,sorted_index_side_bolt=added6.squeeze(0).sort(dim=0)
            sorted_index_side_bolt_x=sorted_index_side_bolt[:,0]
            goals_side_bolts=added6[:,sorted_index_side_bolt_x,:]
            side_bolts[i,:,:]=goals_side_bolts[0,:,:]
        else:
            side_bolts[i,:,:]=torch.ones((2,3))
            mask[i][9:10]=0


        

        index_cover_bolts=target==6
        points7=points[index_cover_bolts,:]
        # _,c7=KMeans(points7,goals_[6],Niter=300)
        _,c7=kmeans(X=points7, num_clusters=goals_[6], distance='euclidean', device=torch.device('cpu'))
        # c7=c7.cuda()
        #######################
        # Visuell(points7,c7)
        # together7=torch.cat([c7,points7],dim=0).unsqueeze(0)
        # index7=knn(together7.permute(0, 2, 1),2)
        # inter=index7[0,0:goals_[6],1]
        # Visuell__(together7.squeeze(0),inter)
        ############################
        together7=torch.cat([c7,points7],dim=0).unsqueeze(0)
        index7=knn(together7.permute(0, 2, 1),2)[0,0:goals_[6],1].unsqueeze(0)
        added7=index_points(together7,index7)
        _,sorted_index_cover_bolt=added7.squeeze(0).sort(dim=0)
        sorted_index_cover_bolt_x=sorted_index_cover_bolt[:,0]
        goals_cover_bolts=added7[:,sorted_index_cover_bolt_x,:]
        cover_bolts[i,:,:]=goals_cover_bolts[0,:,:]
    goals=torch.cat((clampingsystem,covers,gearcontainers,chargers,bottoms,side_bolts,cover_bolts), dim=1)
    return goals.numpy(),mask.numpy()



class MotorDataset(Dataset):
    def __init__(self,split='train',data_root='directory to training data',num_points=4096,bolt_weight=1,test_area='Validation',block_size=1.0,sample_rate=1.0,transform=None):
        super().__init__()
        self.num_points=num_points
        self.block_size=block_size
        self.transform=transform      
        motor_list=sorted(os.listdir(data_root))        #list all subdirectory    
        #motor_filter=[motor for motor in motor_list if 'Type' in motor]     #filter all the file, whose name has no Type      
        if split == 'train':        #load training files or validation files
            motor_positions=[motor for motor in motor_list if '{}'.format(test_area) not in motor]
        else:
            motor_positions=[motor for motor in motor_list if '{}'.format(test_area) in motor]

        
        ######################load the np file###################################################    
        
        self.motors_points=[]       #initial object_motor_points and object_motor_lables
        self.motors_labels=[]      
        num_points_eachmotor=[]     #initial a list to count the num of points for each motor   
        label_num_eachtype=np.zeros(7)      #initial a array to count how much points is there for each type
        for motor_position in tqdm(motor_positions,total=len(motor_positions)):
            motor_directory=os.path.join(data_root,motor_position)
            motor_data=np.load(motor_directory)
            # motor_data=densify_blots(motor_data)
            # motor_data=densify_blots(motor_data)
            motor_points=motor_data[:,0:6]
            motor_labels=motor_data[:,6]            #result is a np array           
            num_eachtype_in_one_motor,_=np.histogram(motor_labels,bins=7,range=(0,7))       #count how much points is there for each type(usage of np.histotram)
            label_num_eachtype+=num_eachtype_in_one_motor
            self.motors_points.append(motor_points)
            self.motors_labels.append(motor_labels)
            num_points_eachmotor.append(motor_labels.size)
        #  record goals
        # self.goals,self.mask=find_goals_kmeans(np.array(self.motors_points),np.array(self.motors_labels))
        # self.goals=self.goals.reshape(self.goals.shape[0],-1)
        # np.savetxt("goals_6.txt",self.goals.reshape(self.goals.shape[0],-1))
        # np.savetxt("mask_6.txt",self.mask)

        self.goals=np.loadtxt("goals_6.txt")
        self.goals=self.goals.reshape(self.goals.shape[0],-1,3)
        self.mask=np.loadtxt("mask_6.txt")
        #id=Get_ObjectID(motor_data)
        #print(id)
        ############################################################################################


        ###########according to lable_num_eachmotor and bolt_weight, caculate the labelweights######
        label_num_eachtype=label_num_eachtype.astype(np.float32)
        label_num_eachtype__=label_num_eachtype+0
        self.persentage=label_num_eachtype__/np.sum(label_num_eachtype__)
        print(self.persentage)
        ####reversed order
        # label_num_eachtype[-2]*=30
        # label_num_eachtype[-1]/=10
        # label_num_eachtype[-2]/=bolt_weight
        labelweights=label_num_eachtype/np.sum(label_num_eachtype)
        labelweights=np.power(np.max(labelweights)/labelweights,1/3)
        #### normal order
        # label_num_eachtype[-1]*=bolt_weight
        # label_num_eachtype[-2]*=bolt_weight
        # labelweights=label_num_eachtype/np.sum(label_num_eachtype)
        # labelweights=np.power(labelweights/np.max(labelweights),1/3)
        self.num_eachtype__=label_num_eachtype__
        self.num_eachtype=label_num_eachtype
        self.labelweights=labelweights/np.sum(labelweights)
        ############################################################################################


        #############caculate the index for choose of points from the motor according to the number of points of a specific motor#########      
        sample_prob_eachmotor=num_points_eachmotor/np.sum(num_points_eachmotor)     #probability for choosing from a specific motor      
        num_interation=sample_rate*np.sum(num_points_eachmotor)/self.num_points     #num_of_all to choose npoints cloud       
        self.motors_indes=[]        #initial motors_indes list    
        for index in range(len(num_points_eachmotor)):      #allocate the index according to probability
            sample_times_to_onemotor=int(round(sample_prob_eachmotor[index]*num_interation))
            motor_indes_onemotor=[index]*sample_times_to_onemotor
            self.motors_indes.extend(motor_indes_onemotor)
        ####################################################################################################################################
         
                
    def __getitem__(self,index):
        
        points=self.motors_points[self.motors_indes[index]][:,0:3]      #initialize the parameter
        labels=self.motors_labels[self.motors_indes[index]]
        n_points=points.shape[0]   
        ########################have a randow choose of points from points cloud#######################
        choice=np.random.choice(n_points,self.num_points,replace=True)
        chosed_points=points[choice,:]
        chosed_labels=labels[choice]
        ###############################################################################################
        

        return chosed_points,chosed_labels,self.goals[self.motors_indes[index]],self.mask[self.motors_indes[index]]

    def __len__(self):                                                                            
        return len(self.motors_indes) 


class MotorDataset_validation(Dataset):
    def __init__(self,split='train',data_root='directory to training data',num_points=4096,bolt_weight=1,test_area='Validation',block_size=1.0,sample_rate=1.0,transform=None):
        super().__init__()
        self.num_points=num_points
        self.block_size=block_size
        self.transform=transform      
        motor_list=sorted(os.listdir(data_root))        #list all subdirectory    
        #motor_filter=[motor for motor in motor_list if 'Type' in motor]     #filter all the file, whose name has no Type      
        if split == 'train':        #load training files or validation files
            motor_positions=[motor for motor in motor_list if '{}'.format(test_area) not in motor]
        else:
            motor_positions=[motor for motor in motor_list if '{}'.format(test_area) in motor]

        
        ######################load the np file###################################################    
        
        self.motors_points=[]       #initial object_motor_points and object_motor_lables
        self.motors_labels=[]      
        num_points_eachmotor=[]     #initial a list to count the num of points for each motor   
        label_num_eachtype=np.zeros(7)      #initial a array to count how much points is there for each type
        for motor_position in tqdm(motor_positions,total=len(motor_positions)):
            motor_directory=os.path.join(data_root,motor_position)
            motor_data=np.load(motor_directory)
            motor_points=motor_data[:,0:6]
            motor_labels=motor_data[:,6]            #result is a np array           
            num_eachtype_in_one_motor,_=np.histogram(motor_labels,bins=7,range=(0,7))       #count how much points is there for each type(usage of np.histotram)
            label_num_eachtype+=num_eachtype_in_one_motor
            self.motors_points.append(motor_points)
            self.motors_labels.append(motor_labels)
            num_points_eachmotor.append(motor_labels.size)
        ############################################################################################


        ###########according to lable_num_eachmotor and bolt_weight, caculate the labelweights######
        label_num_eachtype[-1]/=bolt_weight
        labelweights=label_num_eachtype/np.sum(label_num_eachtype)
        labelweights=np.power(np.max(labelweights)/labelweights,1/3)
        self.labelweight=labelweights/np.sum(labelweights)
        ############################################################################################


        #############caculate the index for choose of points from the motor according to the number of points of a specific motor#########      
        sample_prob_eachmotor=num_points_eachmotor/np.sum(num_points_eachmotor)     #probability for choosing from a specific motor      
        num_interation=sample_rate*np.sum(num_points_eachmotor)/self.num_points     #num_of_all to choose npoints cloud       
        self.motors_indes=[]        #initial motors_indes list    
        for index in range(len(num_points_eachmotor)):      #allocate the index according to probability
            sample_times_to_onemotor=int(round(sample_prob_eachmotor[index]*num_interation))
            motor_indes_onemotor=[index]*sample_times_to_onemotor
            self.motors_indes.extend(motor_indes_onemotor)
        ####################################################################################################################################
         
                
    def __getitem__(self,index):
        
        points=self.motors_points[self.motors_indes[index]][:,0:3]      #initialize the parameter
        labels=self.motors_labels[self.motors_indes[index]]
        n_points=points.shape[0]   
        #points=rotate(points)   
        #normalized_points=pc_normalize(points)      #normalize the points


        ########################have a randow choose of points from points cloud#######################
        choice=np.random.choice(n_points,self.num_points,replace=True)
        chosed_points=points[choice,:]
        chosed_labels=labels[choice]
        ###############################################################################################

        return chosed_points,chosed_labels

    def __len__(self):                                                                            
        return len(self.motors_indes) 


class MotorDataset_patch(Dataset):
    def __init__(self,split='train',data_root='directory to training data',num_points=4096,bolt_weight=1,test_area='Validation',block_size=1.0,sample_rate=1.0,transform=None):
        super().__init__()
        self.num_points=num_points
        self.block_size=block_size
        self.transform=transform      
        motor_list=sorted(os.listdir(data_root))        #list all subdirectory    
        motor_filter=[motor for motor in motor_list if 'Type' in motor]     #filter all the file, whose name has no Type      
        if split == 'train':        #load training files or validation files
            motor_positions=[motor for motor in motor_filter if '{}'.format(test_area) not in motor]
        else:
            motor_positions=[motor for motor in motor_filter if '{}'.format(test_area) in motor]

        
        ######################load the np file###################################################    
        
        self.motors_points=[]       #initial object_motor_points and object_motor_lables
        self.motors_labels=[]      
        self.interation_times_eachmotor = []
        label_num_eachtype=np.zeros(6)      #initial a array to count how much points is there for each type
        for motor_position in tqdm(motor_positions,total=len(motor_positions)):
            motor_directory=os.path.join(data_root,motor_position)
            motor_data=np.load(motor_directory)
            motor_points=motor_data[:,0:3]
            motor_points=pc_normalize(motor_points)      #normalize the points
            motor_labels=motor_data[:,6]            
            motor_points_labels=[] 

            ################       judge if current motor points cloud can just include integel             ########
            ###############     times of sub points_clouds of num_point,if not, patch it to integel one     ########
            current_motor_size=motor_points.shape[0]
            if current_motor_size % self.num_points !=0:
                num_add_points=self.num_points-(current_motor_size % self.num_points)
                choice=np.random.choice(current_motor_size,num_add_points,replace=True)     #pick out some points from current cloud to patch up the current cloud
                add_points=motor_points[choice,:]
                motor_points=np.vstack((motor_points,add_points))
                add_labels=motor_labels[choice]
                motor_labels=np.hstack((motor_labels,add_labels))
            #########################################################################################################
            #########################################################################################################

            motor_points_labels=np.hstack((motor_points,motor_labels.reshape((motor_labels.size,1))))       #merge the labels and points in order to schuffle it
            np.random.shuffle(motor_points_labels)
            motor_points=motor_points_labels[:,0:3]     #get the schuffled points and lables
            motor_labels=motor_points_labels[:,3]
            self.interation_times_eachmotor.append(motor_labels.size/self.num_points)       #record how money 4096 points could be taken out for one motor points cloud after patch
            num_eachtype_in_one_motor,_=np.histogram(motor_labels,bins=6,range=(0,6))       #count how much points is there for each type(usage of np.histotram)
            label_num_eachtype+=num_eachtype_in_one_motor
            self.motors_points.append(motor_points)
            self.motors_labels.append(motor_labels)
        ############################################################################################


        ###########according to lable_num_eachmotor and bolt_weight, caculate the labelweights######
        label_num_eachtype[-1]/=bolt_weight
        labelweights=label_num_eachtype/np.sum(label_num_eachtype)
        labelweights=np.power(np.max(labelweights)/labelweights,1/3)
        self.labelweight=labelweights/np.sum(labelweights)
        ############################################################################################


        #############caculate the index for choose of points from the motor according to the number of points of a specific motor#########      
        self.motors_indes=[]        #initial motors_indes list    
        for index in range(len(self.interation_times_eachmotor)):      #allocate the index according to probability
            motor_indes_onemotor=[index]*int(self.interation_times_eachmotor[index])
            self.motors_indes.extend(motor_indes_onemotor)
        ####################################################################################################################################


        #####################################   set the dictionary for dataloader index according to motors_points structure        ########
        self.dic_block_accumulated_per_motors={}
        key=0
        for index in range(len(self.interation_times_eachmotor)):
            if index!=0:
                key=key+self.interation_times_eachmotor[index-1]
            for num_clouds_per_motor in range(int(self.interation_times_eachmotor[index])):
                self.dic_block_accumulated_per_motors[int(key+num_clouds_per_motor)]=num_clouds_per_motor
        ####################################################################################################################################


    def __getitem__(self,index):   
        points=self.motors_points[self.motors_indes[index]]      #initialize the points cloud for each motor
        labels=self.motors_labels[self.motors_indes[index]]
        

        sequence=np.arange(self.num_points)
        chosed_points=points[self.num_points*self.dic_block_accumulated_per_motors[index]+sequence,:]       #ensure all the points could be picked out by the ways of patch
        chosed_labels=labels[self.num_points*self.dic_block_accumulated_per_motors[index]+sequence]    
        return chosed_points,chosed_labels

    def __len__(self):                                                                            
        return len(self.motors_indes) 