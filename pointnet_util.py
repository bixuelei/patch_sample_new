## Borrowed from https://github.com/yanx27/Pointnet_Pointnet2_pytorch


import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
# import open3d
# from pykeops.torch import LazyTensor
# from kmeans_pytorch import kmeans

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

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


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points



def Visuell(sampled,add, SavePCDFile = False, FileName = None):
    #get only the koordinate from sampled
    sampled=sampled.cpu()
    sampled = np.asarray(sampled)
    add=add.cpu()
    add = np.asarray(add)
    PointCloud_koordinate = np.vstack((sampled,add))

    colors=[]
    for i in range(sampled.shape[0]):
        colors.append([70,70,70])
    for i in range(add.shape[0]):
        colors.append([255,0,0])
    colors=np.array(colors)
    colors=colors/255
    print(colors)

    #visuell the point cloud
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(PointCloud_koordinate)
    point_cloud.colors = open3d.utility.Vector3dVector(colors)
    open3d.visualization.draw_geometries([point_cloud])

    if SavePCDFile is True:
    # #save the pcd file
        open3d.io.write_point_cloud(FileName +'.pcd', point_cloud)



def Visuell__(sampled,add, SavePCDFile = False, FileName = None):
    #get only the koordinate from sampled
    sampled=sampled.cpu()
    sampled = np.asarray(sampled)
    add=add.cpu()
    add = np.asarray(add)
    PointCloud_koordinate=sampled
    colors=[]
    for i in range(sampled.shape[0]):
        if i not in add:
            colors.append([70,70,70])
        else:
            colors.append([255,0,0])
    colors=np.array(colors)
    colors=colors/255
    print(colors)

    #visuell the point cloud
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(PointCloud_koordinate)
    point_cloud.colors = open3d.utility.Vector3dVector(colors)
    open3d.visualization.draw_geometries([point_cloud])

    if SavePCDFile is True:
    # #save the pcd file
        open3d.io.write_point_cloud(FileName +'.pcd', point_cloud)



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



# def KMeans(x, K=10, Niter=10, verbose=False):
#     """Implements Lloyd's algorithm for the Euclidean metric."""

#     # start = time.time()
#     N, D = x.shape  # Number of samples, dimension of the ambient space

#     c = x[:K, :].clone()  # Simplistic initialization for the centroids

#     x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
#     c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

#     # K-means loop:
#     # - x  is the (N, D) point cloud,
#     # - cl is the (N,) vector of class labels
#     # - c  is the (K, D) cloud of cluster centroids
#     for i in range(Niter):

#         # E step: assign points to the closest cluster -------------------------
#         D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
#         cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

#         # M step: update the centroids to the normalized cluster average: ------
#         # Compute the sum of points per cluster:
#         c.zero_()
#         c.scatter_add_(0, cl[:, None].repeat(1, D), x)

#         # Divide by the number of points per cluster:
#         Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
#         c /= Ncl  # in-place division to compute the average

#     # if verbose:  # Fancy display -----------------------------------------------
#     #     if use_cuda:
#     #         torch.cuda.synchronize()
#     #     end = time.time()
#     #     print(
#     #         f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
#     #     )
#     #     print(
#     #         "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
#     #             Niter, end - start, Niter, (end - start) / Niter
#     #         )
#     #     )

#     return cl, c



# def find_goals_kmeans(points,target):  # [bs,n_points,C] [bs,n_points]
#     points=points.astype(float)
#     points = torch.from_numpy(points).cuda()
#     target= torch.from_numpy(target).cuda()
#     goals_=[4,2,1,1,1,2,5]
#     bs=points.shape[0]
#     mask=torch.ones((bs,16))
#     device=points.device
#     cover_bolts=torch.zeros(bs,goals_[6],3)
#     side_bolts=torch.zeros(bs,goals_[5],3)
#     bottoms=torch.zeros(bs,goals_[4],3)
#     chargers=torch.zeros(bs,goals_[2],3)
#     gearcontainers=torch.zeros(bs,goals_[3],3)
#     covers=torch.zeros(bs,goals_[1],3)
#     clampingsystem=torch.zeros(bs,goals_[0],3)
#     for i in range(bs):
#         index_clampingsystem=target[i,:]==0
#         points1=points[i,index_clampingsystem,:]
#         _,c1=kmeans(
#     X=points1, num_clusters=goals_[0], distance='euclidean', device=torch.device('cuda')
# )
#         c1=c1.cuda()
#         ########################
#         # Visuell(points1,c1)
#         # together1=torch.cat([c1,points1],dim=0).unsqueeze(0)
#         # index1=knn(together1.permute(0, 2, 1),2)
#         # inter=index1[0,0:goals_[0],1]
#         # Visuell__(together1.squeeze(0),inter)
#         ############################
#         together1=torch.cat([c1,points1],dim=0).unsqueeze(0)
#         index1=knn(together1.permute(0, 2, 1),2)[0,0:goals_[0],1].unsqueeze(0)
#         added1=index_points(together1,index1)
#         _,sorted_index_clamping=added1.squeeze(0).sort(dim=0)
#         sorted_index_clamping_x=sorted_index_clamping[:,0]
#         goals_clamping=added1[:,sorted_index_clamping_x,:]
#         clampingsystem[i,:,:]=goals_clamping[0,:,:]



#         index_cover=target[i,:]==1
#         num2=torch.sum(index_cover,dim=0)
#         if num2>=goals_[1]:
#             points2=points[i,index_cover,:]
#             # _,c2=KMeans(points2,goals_[1])
#             _,c2=kmeans(X=points2, num_clusters=goals_[1], distance='euclidean', device=torch.device('cuda'))
#             c2=c2.cuda()
#             ########################
#             # Visuell(points2,c2)
#             # together2=torch.cat([c2,points2],dim=0).unsqueeze(0)
#             # index2=knn(together2.permute(0, 2, 1),2)
#             # inter=index2[0,0:goals_[1],1]
#             # Visuell__(together2.squeeze(0),inter)
#             ############################
#             together2=torch.cat([c2,points2],dim=0).unsqueeze(0)
#             index2=knn(together2.permute(0, 2, 1),2)[0,0:goals_[1],1].unsqueeze(0)
#             added2=index_points(together2,index2)
#             _,sorted_index_cover=added2.squeeze(0).sort(dim=0)
#             sorted_index_cover_x=sorted_index_cover[:,0]
#             goals_cover=added2[:,sorted_index_cover_x,:]
#             covers[i,:,:]=goals_cover[0,:,:]
#         else:
#             covers[i,:,:]=torch.zeros((goals_[1],3))
#             mask[i][4]=0
#             mask[i][5]=0


#         index_gearcontainer=target[i,:]==2
#         if torch.sum(index_gearcontainer,dim=0) >=1:
#             points_gear=points[i,index_gearcontainer,:]
#             points_gear_c=torch.sum(points_gear,dim=0)/torch.sum(points_gear,dim=0)
#             together3=torch.cat([points_gear_c,points_gear],dim=0).unsqueeze(0)
#             index3=knn(together3.permute(0, 2, 1),2)[0,0:goals_[2],1].unsqueeze(0)
#             added3=index_points(together3,index3)
#             gearcontainers[i,:,:]=added3[0,:,:]
#         else:
#             gearcontainers[i,:,:]=torch.zeros((1,3))
#             mask[i][6]=0

#         index_charger=target[i,:]==3
#         if torch.sum(index_charger,dim=0) >=1:
#             points_charger=points[i,index_charger,:]
#             points_charger_c=torch.sum(points_charger,dim=0)/torch.sum(points_charger,dim=0)
#             together4=torch.cat([points_charger_c,points_charger],dim=0).unsqueeze(0)
#             index4=knn(together4.permute(0, 2, 1),2)[0,0:goals_[3],1].unsqueeze(0)
#             added4=index_points(together4,index4)
#             chargers[i,:,:]=added4[0,:,:]
#         else:
#             chargers[i,:,:]=torch.zeros((1,3))
#             mask[i][7]=0

#         index_bottom=target[i,:]==4
#         if torch.sum(index_bottom,dim=0) >=1:
#             points_bottom=points[i,index_bottom,:]
#             points_bottom_c=torch.sum(points_bottom,dim=0)/torch.sum(points_bottom,dim=0)
#             together5=torch.cat([points_bottom,points_bottom_c],dim=0).unsqueeze(0)
#             index5=knn(together5.permute(0, 2, 1),2)[0,0:goals_[4],1].unsqueeze(0)
#             added5=index_points(together5,index5)
#             bottoms[i,:,:]=added5[0,:,:]
#         else:
#             bottoms[i,:,:]=torch.zeros((1,3))
#             mask[i][8]=0
#         # index_gearcontainer=target[i,:]==2
#         # if points[i,index_gearcontainer,:].unsqueeze(0).shape[1]!=0:
#         #     points3=points[i,index_gearcontainer,:]
#         #     # _,c3=KMeans(points3,goals_[2])
#         #     _,c3=kmeans(X=points3, num_clusters=goals_[2], distance='euclidean', device=torch.device('cuda'))
#         #     c3=c3.cuda()
#         #     ########################
#         #     # Visuell(points3,c3)
#         #     # together3=torch.cat([c3,points3],dim=0).unsqueeze(0)
#         #     # index3=knn(together3.permute(0, 2, 1),2)
#         #     # inter=index3[0,0:goals_[2],1]
#         #     # Visuell__(together3.squeeze(0),inter)
#         #     ############################
#         #     together3=torch.cat([c3,points3],dim=0).unsqueeze(0)
#         #     index3=knn(together3.permute(0, 2, 1),2)[0,0:goals_[2],1].unsqueeze(0)
#         #     added3=index_points(together3,index3)
#         #     _,sorted_index_gearcontainer=added3.squeeze(0).sort(dim=0)
#         #     sorted_index_gearcontainer_x=sorted_index_gearcontainer[:,0]
#         #     goals_gearcontainer=added3[:,sorted_index_gearcontainer_x,:]
#         #     gearcontainers[i,:,:]=goals_gearcontainer[0,:,:]
#         # else:
#         #     gearcontainers[i,:,:]=torch.zeros((goals_[2],3))
#         #     mask[i][6]=0
            



#         # index_charger=target[i,:]==3
#         # if points[i,index_charger,:].unsqueeze(0).shape[1]!=0:
#         #     points4=points[i,index_charger,:]
#         #     # _,c4=KMeans(points4,goals_[3])
#         #     _,c4=kmeans(X=points4, num_clusters=goals_[3], distance='euclidean', device=torch.device('cuda'))
#         #     c4=c4.cuda()
#         #     ########################
#         #     # Visuell(points4,c4)
#         #     # together4=torch.cat([c4,points4],dim=0).unsqueeze(0)
#         #     # index4=knn(together4.permute(0, 2, 1),2)
#         #     # inter=index4[0,0:goals_[3],1]
#         #     # Visuell__(together4.squeeze(0),inter)
#         #     ############################
#         #     together4=torch.cat([c4,points4],dim=0).unsqueeze(0)
#         #     index4=knn(together4.permute(0, 2, 1),2)[0,0:goals_[3],1].unsqueeze(0)
#         #     added4=index_points(together4,index4)
#         #     _,sorted_index_charger=added4.squeeze(0).sort(dim=0)
#         #     sorted_index_charger_x=sorted_index_charger[:,0]
#         #     goals_charger=added4[:,sorted_index_charger_x,:]
#         #     chargers[i,:,:]=goals_charger[0,:,:]
#         # else:
#         #     chargers[i,:,:]=torch.zeros((goals_[3],3))
#         #     mask[i][7]=0



#         # index_bottom=target[i,:]==4
#         # num5=torch.sum(index_bottom,dim=0)
#         # if num5>=goals_[4]:
#         #     points5=points[i,index_bottom,:]
#         #     # _,c5=KMeans(points5,goals_[4])
#         #     _,c5=kmeans(X=points5, num_clusters=goals_[4], distance='euclidean', device=torch.device('cuda'))
#         #     c5=c5.cuda()
#         #     ########################
#         #     # Visuell(points5,c5)
#         #     # together5=torch.cat([c5,points5],dim=0).unsqueeze(0)
#         #     # index5=knn(together5.permute(0, 2, 1),2)
#         #     # inter=index5[0,0:goals_[4],1]
#         #     # Visuell__(together5.squeeze(0),inter)
#         #     ############################
#         #     together5=torch.cat([c5,points5],dim=0).unsqueeze(0)
#         #     index5=knn(together5.permute(0, 2, 1),2)[0,0:goals_[4],1].unsqueeze(0)
#         #     added5=index_points(together5,index5)
#         #     _,sorted_index_bottom=added5.squeeze(0).sort(dim=0)
#         #     sorted_index_bottom_x=sorted_index_bottom[:,0]
#         #     goals_bottom=added5[:,sorted_index_bottom_x,:]
#         #     bottoms[i,:,:]=goals_bottom[0,:,:]
#         # else:
#         #     bottoms[i,:,:]=torch.zeros((goals_[4],3))
#         #     mask[i][8]=0


#         index_side_bolts=target[i,:]==5
#         num6=torch.sum(index_side_bolts,dim=0)
#         if num6>=goals_[5]:
#             points6=points[i,index_side_bolts,:]
#             # _,c6=KMeans(points6,goals_[5],Niter=20)
#             _,c6=kmeans(X=points6, num_clusters=goals_[5], distance='euclidean', device=torch.device('cuda'))
#             c6=c6.cuda()
#             ########################
#             # Visuell(points6,c6)
#             # together6=torch.cat([c6,points6],dim=0).unsqueeze(0)
#             # index6=knn(together6.permute(0, 2, 1),2)
#             # inter=index6[0,0:goals_[5],1]
#             # Visuell__(together6.squeeze(0),inter)
#             ############################
#             together6=torch.cat([c6,points6],dim=0).unsqueeze(0)
#             index6=knn(together6.permute(0, 2, 1),2)[0,0:goals_[5],1].unsqueeze(0)
#             added6=index_points(together6,index6)
#             _,sorted_index_side_bolt=added6.squeeze(0).sort(dim=0)
#             sorted_index_side_bolt_x=sorted_index_side_bolt[:,0]
#             goals_side_bolts=added6[:,sorted_index_side_bolt_x,:]
#             side_bolts[i,:,:]=goals_side_bolts[0,:,:]
#         else:
#             side_bolts[i,:,:]=torch.zeros((goals_[5],3))
#             mask[i][9:10]=0
        

#         index_cover_bolts=target[i,:]==6
#         num7=torch.sum(index_cover_bolts,dim=0)
#         if  num7>=goals_[6]:
#             points7=points[i,index_cover_bolts,:]
#             # _,c7=KMeans(points7,goals_[6],Niter=20)
#             _,c7=kmeans(X=points7, num_clusters=goals_[6], distance='euclidean', device=torch.device('cuda'))
#             c7=c7.cuda()
#             ########################
#             # Visuell(points7,c7)
#             # together7=torch.cat([c7,points7],dim=0).unsqueeze(0)
#             # index7=knn(together7.permute(0, 2, 1),2)
#             # inter=index7[0,0:goals_[6],1]
#             # Visuell__(together7.squeeze(0),inter)
#             ############################
#             together7=torch.cat([c7,points7],dim=0).unsqueeze(0)
#             index7=knn(together7.permute(0, 2, 1),2)[0,0:goals_[6],1].unsqueeze(0)
#             added7=index_points(together7,index7)
#             _,sorted_index_cover_bolt=added7.squeeze(0).sort(dim=0)
#             sorted_index_cover_bolt_x=sorted_index_cover_bolt[:,0]
#             goals_cover_bolts=added7[:,sorted_index_cover_bolt_x,:]
#             cover_bolts[i,:,:]=goals_cover_bolts[0,:,:]
#         else:
#             cover_bolts[i,:,:]=torch.zeros((goals_[6],3))
#             mask[i][11:16]=0

#     goals=torch.cat((clampingsystem,covers,gearcontainers,chargers,bottoms,side_bolts,cover_bolts), dim=1).to(device)
#     return goals,mask



# def find_goals_fps(points,target):  # [bs,n_points,C] [bs,n_points]
#     bs=points.shape[0]
#     device=points.device
#     mask=torch.ones((bs,16))
#     clampingsystem=torch.zeros(bs,4,3)
#     covers=torch.zeros(bs,2,3)
#     chargers=torch.zeros(bs,1,3)
#     gearcontainers=torch.zeros(bs,1,3)
#     bottoms=torch.zeros(bs,1,3)
#     side_bolts=torch.zeros(bs,2,3)
#     cover_bolts=torch.zeros(bs,5,3)
#     for i in range(bs):
#         index_clampingsystem=target[i,:]==0
#         index_goals_clamping=farthest_point_sample(points[i,index_clampingsystem,:].unsqueeze(0),4)
#         goals_clamping=index_points(points[i,:,:].unsqueeze(0),index_goals_clamping)
#         _,sorted_index_clamping=goals_clamping.squeeze(0).sort(dim=0)
#         sorted_index_clamping_x=sorted_index_clamping[:,0]
#         goals_clamping=goals_clamping[:,sorted_index_clamping_x,:]
#         clampingsystem[i,:,:]=goals_clamping[0,:,:]


#         index_cover=target[i,:]==1
#         index_goals_cover=farthest_point_sample(points[i,index_cover,:].unsqueeze(0),2)
#         goals_cover=index_points(points[i,:,:].unsqueeze(0),index_goals_cover)
#         _,sorted_index_cover=goals_cover.squeeze(0).sort(dim=0)
#         sorted_index_cover_x=sorted_index_cover[:,0]
#         goals_cover=goals_cover[:,sorted_index_cover_x,:]
#         covers[i,:,:]=goals_cover[0,:,:]


#         index_gearcontainer=target[i,:]==2
#         if torch.sum(index_gearcontainer,dim=0) >=1:
#             points_gear=points[i,index_gearcontainer,:]
#             points_gear=torch.sum(points_gear,dim=0)/torch.sum(points_gear,dim=0)
#             gearcontainers[i,:,:]=points_gear
#         else:
#             gearcontainers[i,:,:]=torch.zeros((1,3))
#             mask[i][6]=0

#         index_charger=target[i,:]==3
#         if torch.sum(index_charger,dim=0) >=1:
#             points_charger=points[i,index_charger,:]
#             points_charger=torch.sum(points_charger,dim=0)/torch.sum(points_charger,dim=0)
#             chargers[i,:,:]=points_charger
#         else:
#             chargers[i,:,:]=torch.zeros((1,3))
#             mask[i][7]=0

#         index_bottom=target[i,:]==4
#         if torch.sum(index_bottom,dim=0) >=1:
#             points_bottom=points[i,index_bottom,:]
#             points_bottom=torch.sum(points_bottom,dim=0)/torch.sum(points_bottom,dim=0)
#             bottoms[i,:,:]=points_bottom
#         else:
#             bottoms[i,:,:]=torch.zeros((1,3))
#             mask[i][8]=0


#         index_side_bolts=target[i,:]==5
#         if torch.sum(index_side_bolts,dim=0)>=2:
#             index_goals_side_bolt=farthest_point_sample(points[i,index_side_bolts,:].unsqueeze(0),2)
#             goals_side_bolt=index_points(points[i,:,:].unsqueeze(0),index_goals_side_bolt)
#             _,sorted_index_side_bolt=goals_side_bolt.squeeze(0).sort(dim=0)
#             sorted_index_side_bolt_z=sorted_index_side_bolt[:,2]
#             goals_side_bolt=goals_side_bolt[:,sorted_index_side_bolt_z,:]
#             side_bolts[i,:,:]=goals_side_bolt[0,:,:]
#         else:
#             side_bolts[i,:,:]=torch.zeros((2,3))
#             mask[i][9:10]=0


#         index_cover_bolts=target[i,:]==6
#         if torch.sum(index_cover_bolts,dim=0)>=5:
#             index_goals_cover_bolt=farthest_point_sample(points[i,index_cover_bolts,:].unsqueeze(0),5)
#             goals_cover_bolt=index_points(points[i,:,:].unsqueeze(0),index_goals_cover_bolt)
#             _,sorted_index_cover_bolt=goals_cover_bolt.squeeze(0).sort(dim=0)
#             sorted_index_cover_bolt_x=sorted_index_cover_bolt[:,0]
#             goals_cover_bolt=goals_cover_bolt[:,sorted_index_cover_bolt_x,:]
#             cover_bolts[i,:,:]=goals_cover_bolt[0,:,:]
#         else:
#             cover_bolts[i,:,:]==torch.zeros((5,3))
#             mask[i][11:16]=0


#     goals=torch.cat((clampingsystem,covers,gearcontainers,chargers,bottoms,side_bolts,cover_bolts), dim=1).to(device)
#     mask=mask.to(device)
#     return goals,mask



def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    _,h,c=sqrdists.shape
    # radius1=torch.tensor(B,h,c,dtype=torch.float32)
    # radius1=radius ** 2
    # radius1=radius1.to(device)
    

    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points



class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

