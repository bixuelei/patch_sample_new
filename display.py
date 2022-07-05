
   
import os
import numpy as np
import random
import open3d 
import csv
import math
import matplotlib.pyplot as plt
from pyparsing import col
import torch

color_map={"back_ground":[0,0,128],
           "cover":[0,100,0],
           "gear_container":[0,255,0],
           "charger":[255,255,0],
           "bottom":[255,165,0],
           "bolts":[255,0,0],
           "side_bolts":[255,0,255]}



def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = batch_data.shape
    for b in range(B):
        pc = batch_data[b]
        centroid = torch.mean(pc, dim=0,keepdim=True)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1,keepdim=True)))
        pc = pc / m
        batch_data[b] = pc
    return batch_data



def Visuell_PointCloud(sampled, SavePCDFile = False, FileName = None):
    #get only the koordinate from sampled
    sampled = np.asarray(sampled)
    PointCloud_koordinate = sampled[:, 0:3]
    label=sampled[:,6]
    labels = np.asarray(label)
    colors=[]
    for i in range(labels.shape[0]):
        dp=labels[i]
        if dp==0:
            r=color_map["back_ground"][0]
            g=color_map["back_ground"][1]
            b=color_map["back_ground"][2]
            colors.append([r,g,b])
        elif dp==1:
            r=color_map["cover"][0]
            g=color_map["cover"][1]
            b=color_map["cover"][2]
            colors.append([r,g,b])
        elif dp==2:
            r=color_map["gear_container"][0]
            g=color_map["gear_container"][1]
            b=color_map["gear_container"][2]
            colors.append([r,g,b])
        elif dp==3:
            r=color_map["charger"][0]
            g=color_map["charger"][1]
            b=color_map["charger"][2]
            colors.append([r,g,b])
        elif dp==4:
            r=color_map["bottom"][0]
            g=color_map["bottom"][1]
            b=color_map["bottom"][2]
            colors.append([r,g,b])
        elif dp==5:
            r=color_map["side_bolts"][0]
            g=color_map["side_bolts"][1]
            b=color_map["side_bolts"][2]
            colors.append([r,g,b])
        else :
            r=color_map["bolts"][0]
            g=color_map["bolts"][1]
            b=color_map["bolts"][2]
            colors.append([r,g,b])
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



def Visuell_superpoint(sampled,original,SavePCDFile = False, FileName = None):
    #get only the koordinate from sampled
    sampled=sampled.cpu()
    if sampled.requires_grad==True:
        sampled=sampled.detach().numpy()
    else:
        sampled = np.asarray(sampled)
    sampled = np.asarray(sampled)

    original=original.cpu()
    if original.requires_grad==True:
        original=original.detach().numpy()
    else:
        original = np.asarray(original)
    original = np.asarray(original)

    PointCloud_koordinate = sampled[:, 0:3]
    label=sampled[:,3]
    labels = np.asarray(label)
    print(labels.shape)
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label>0 else 1))

    PointCloud_koordinate_original = original[:, 0:3]
    size_original=original.shape[0]
    colors_original=[[0.8627,0.8627,0.8627,1] for _ in range(size_original)]
    PointCloud_koordinate=np.concatenate((PointCloud_koordinate,PointCloud_koordinate_original),axis=0)
    colors=np.concatenate((colors,colors_original),axis=0)
    #visuell the point cloud
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(PointCloud_koordinate)
    point_cloud.colors = open3d.utility.Vector3dVector(colors[:, :3])
    open3d.visualization.draw_geometries([point_cloud])

    if SavePCDFile is True:
    # #save the pcd file
        open3d.io.write_point_cloud(FileName +'.pcd', point_cloud)



def Visuell_PointCloud_per_batch_according_to_get_cmap(sampled,target, SavePCDFile = False, FileName = None):
    #get only the koordinate from sampled
    sampled=sampled.cpu()
    target=target.cpu()
    if sampled.requires_grad==True:
        sampled=sampled.detach().numpy()
    else:
        sampled = np.asarray(sampled)
    PointCloud_koordinate_1 = sampled[0,:, 0:3]
    PointCloud_koordinate_2 = sampled[1,:, 0:3]
    PointCloud_koordinate_3 = sampled[2,:, 0:3]
    PointCloud_koordinate_4 = sampled[3,:, 0:3]
    PointCloud_koordinate_5 = sampled[4,:, 0:3]
    label1=target[0,:]
    labels1 = np.asarray(label1)
    #print(labels1.shape)
    max_label1 = labels1.max()

    label2=target[1,:]
    labels2 = np.asarray(label2)
    #print(labels1.shape)
    max_label2 = labels2.max()


    label3=target[2,:]
    labels3 = np.asarray(label3)
    #print(labels1.shape)
    max_label3 = labels3.max()


    label4=target[3,:]
    labels4 = np.asarray(label4)
    #print(labels1.shape)
    max_label4 = labels4.max()


    label5=target[4,:]
    labels5 = np.asarray(label5)
    #print(labels1.shape)
    max_label5 = labels5.max()

    colors1 = plt.get_cmap("tab20")(labels1 / (max_label1 if max_label1>0 else 1))
    colors2 = plt.get_cmap("tab20")(labels2 / (max_label2 if max_label2>0 else 1))
    colors3 = plt.get_cmap("tab20")(labels3 / (max_label3 if max_label3>0 else 1))
    colors4 = plt.get_cmap("tab20")(labels4 / (max_label4 if max_label4>0 else 1))
    colors5 = plt.get_cmap("tab20")(labels5 / (max_label5 if max_label5>0 else 1))

    #visuell the point cloud
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(PointCloud_koordinate_1)
    point_cloud.colors = open3d.utility.Vector3dVector(colors1[:, :3])
    open3d.visualization.draw_geometries([point_cloud])

        #visuell the point cloud
    point_cloud_2 = open3d.geometry.PointCloud()
    point_cloud_2.points = open3d.utility.Vector3dVector(PointCloud_koordinate_2)
    point_cloud_2.colors = open3d.utility.Vector3dVector(colors2[:, :3])
    open3d.visualization.draw_geometries([point_cloud_2])

        #visuell the point cloud
    point_cloud_3 = open3d.geometry.PointCloud()
    point_cloud_3.points = open3d.utility.Vector3dVector(PointCloud_koordinate_3)
    point_cloud_3.colors = open3d.utility.Vector3dVector(colors3[:, :3])
    open3d.visualization.draw_geometries([point_cloud_3])

        #visuell the point cloud
    point_cloud_4 = open3d.geometry.PointCloud()
    point_cloud_4.points = open3d.utility.Vector3dVector(PointCloud_koordinate_4)
    point_cloud_4.colors = open3d.utility.Vector3dVector(colors4[:, :3])
    open3d.visualization.draw_geometries([point_cloud_4])

        #visuell the point cloud
    point_cloud_5 = open3d.geometry.PointCloud()
    point_cloud_5.points = open3d.utility.Vector3dVector(PointCloud_koordinate_5)
    point_cloud_5.colors = open3d.utility.Vector3dVector(colors5[:, :3])
    open3d.visualization.draw_geometries([point_cloud_5])

    if SavePCDFile is True:
    # #save the pcd file
        open3d.io.write_point_cloud(FileName +'.pcd', point_cloud)



def Visuell_PointCloud_per_batch_according_to_label(sampled,target, SavePCDFile = False, FileName = None):
    #get only the koordinate from sampled
    sampled=sampled.cpu()
    target=target.cpu()
    if sampled.requires_grad==True:
        sampled=sampled.detach().numpy()
    else:
        sampled = np.asarray(sampled)
    PointCloud_koordinate_1 = sampled[0,:, 0:3]
    PointCloud_koordinate_2 = sampled[1,:, 0:3]
    PointCloud_koordinate_3 = sampled[2,:, 0:3]
    PointCloud_koordinate_4 = sampled[3,:, 0:3]
    PointCloud_koordinate_5 = sampled[4,:, 0:3]
    label1=target[0,:]
    labels1 = np.asarray(label1)
    colors1=np.zeros((2048,3))
    for i in range(2048):
        dp=labels1[i]
        if dp==0:
            r=color_map["back_ground"][0]
            g=color_map["back_ground"][1]
            b=color_map["back_ground"][2]
        elif dp==1:
            r=color_map["cover"][0]
            g=color_map["cover"][1]
            b=color_map["cover"][2]
        elif dp==2:
            r=color_map["gear_container"][0]
            g=color_map["gear_container"][1]
            b=color_map["gear_container"][2]
        elif dp==3:
            r=color_map["charger"][0]
            g=color_map["charger"][1]
            b=color_map["charger"][2]
        elif dp==4:
            r=color_map["bottom"][0]
            g=color_map["bottom"][1]
            b=color_map["bottom"][2]
        elif dp==5:
            r=color_map["side_bolts"][0]
            g=color_map["side_bolts"][1]
            b=color_map["side_bolts"][2]
        else:
            r=color_map["bolts"][0]
            g=color_map["bolts"][1]
            b=color_map["bolts"][2]
        colors1[i]=[r,g,b]
    colors1=colors1/255


    label2=target[1,:]
    labels2 = np.asarray(label2)
    colors2=np.zeros((2048,3))
    for i in range(2048):
        dp=labels2[i]
        if dp==0:
            r=color_map["back_ground"][0]
            g=color_map["back_ground"][1]
            b=color_map["back_ground"][2]
        elif dp==1:
            r=color_map["cover"][0]
            g=color_map["cover"][1]
            b=color_map["cover"][2]
        elif dp==2:
            r=color_map["gear_container"][0]
            g=color_map["gear_container"][1]
            b=color_map["gear_container"][2]
        elif dp==3:
            r=color_map["charger"][0]
            g=color_map["charger"][1]
            b=color_map["charger"][2]
        elif dp==4:
            r=color_map["bottom"][0]
            g=color_map["bottom"][1]
            b=color_map["bottom"][2]
        elif dp==5:
            r=color_map["side_bolts"][0]
            g=color_map["side_bolts"][1]
            b=color_map["side_bolts"][2]
        else:
            r=color_map["bolts"][0]
            g=color_map["bolts"][1]
            b=color_map["bolts"][2]
        colors2[i]=[r,g,b]
    colors2=colors2/255



    label3=target[2,:]
    labels3 = np.asarray(label3)
    colors3=np.zeros((2048,3))
    for i in range(2048):
        dp=labels3[i]
        if dp==0:
            r=color_map["back_ground"][0]
            g=color_map["back_ground"][1]
            b=color_map["back_ground"][2]
        elif dp==1:
            r=color_map["cover"][0]
            g=color_map["cover"][1]
            b=color_map["cover"][2]
        elif dp==2:
            r=color_map["gear_container"][0]
            g=color_map["gear_container"][1]
            b=color_map["gear_container"][2]
        elif dp==3:
            r=color_map["charger"][0]
            g=color_map["charger"][1]
            b=color_map["charger"][2]
        elif dp==4:
            r=color_map["bottom"][0]
            g=color_map["bottom"][1]
            b=color_map["bottom"][2]
        elif dp==5:
            r=color_map["side_bolts"][0]
            g=color_map["side_bolts"][1]
            b=color_map["side_bolts"][2]
        else:
            r=color_map["bolts"][0]
            g=color_map["bolts"][1]
            b=color_map["bolts"][2]
        colors3[i]=[r,g,b]
    colors3=colors3/255


    label4=target[3,:]
    labels4 = np.asarray(label4)
    #print(labels1.shape)
    colors4=np.zeros((2048,3))
    for i in range(2048):
        dp=labels4[i]
        if dp==0:
            r=color_map["back_ground"][0]
            g=color_map["back_ground"][1]
            b=color_map["back_ground"][2]
        elif dp==1:
            r=color_map["cover"][0]
            g=color_map["cover"][1]
            b=color_map["cover"][2]
        elif dp==2:
            r=color_map["gear_container"][0]
            g=color_map["gear_container"][1]
            b=color_map["gear_container"][2]
        elif dp==3:
            r=color_map["charger"][0]
            g=color_map["charger"][1]
            b=color_map["charger"][2]
        elif dp==4:
            r=color_map["bottom"][0]
            g=color_map["bottom"][1]
            b=color_map["bottom"][2]
        elif dp==5:
            r=color_map["side_bolts"][0]
            g=color_map["side_bolts"][1]
            b=color_map["side_bolts"][2]
        else:
            r=color_map["bolts"][0]
            g=color_map["bolts"][1]
            b=color_map["bolts"][2]
        colors4[i]=[r,g,b]
    colors4=colors4/255


    label5=target[4,:]
    labels5 = np.asarray(label5)
    colors5=np.zeros((2048,3))
    for i in range(2048):
        dp=labels5[i]
        if dp==0:
            r=color_map["back_ground"][0]
            g=color_map["back_ground"][1]
            b=color_map["back_ground"][2]
        elif dp==1:
            r=color_map["cover"][0]
            g=color_map["cover"][1]
            b=color_map["cover"][2]
        elif dp==2:
            r=color_map["gear_container"][0]
            g=color_map["gear_container"][1]
            b=color_map["gear_container"][2]
        elif dp==3:
            r=color_map["charger"][0]
            g=color_map["charger"][1]
            b=color_map["charger"][2]
        elif dp==4:
            r=color_map["bottom"][0]
            g=color_map["bottom"][1]
            b=color_map["bottom"][2]
        elif dp==5:
            r=color_map["side_bolts"][0]
            g=color_map["side_bolts"][1]
            b=color_map["side_bolts"][2]
        else:
            r=color_map["bolts"][0]
            g=color_map["bolts"][1]
            b=color_map["bolts"][2]
        colors5[i]=[r,g,b]
    colors5=colors5/255


    #visuell the point cloud
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(PointCloud_koordinate_1)
    point_cloud.colors = open3d.utility.Vector3dVector(colors1[:, :3])
    open3d.visualization.draw_geometries([point_cloud])

        #visuell the point cloud
    point_cloud_2 = open3d.geometry.PointCloud()
    point_cloud_2.points = open3d.utility.Vector3dVector(PointCloud_koordinate_2)
    point_cloud_2.colors = open3d.utility.Vector3dVector(colors2[:, :3])
    open3d.visualization.draw_geometries([point_cloud_2])

        #visuell the point cloud
    point_cloud_3 = open3d.geometry.PointCloud()
    point_cloud_3.points = open3d.utility.Vector3dVector(PointCloud_koordinate_3)
    point_cloud_3.colors = open3d.utility.Vector3dVector(colors3[:, :3])
    open3d.visualization.draw_geometries([point_cloud_3])

        #visuell the point cloud
    point_cloud_4 = open3d.geometry.PointCloud()
    point_cloud_4.points = open3d.utility.Vector3dVector(PointCloud_koordinate_4)
    point_cloud_4.colors = open3d.utility.Vector3dVector(colors4[:, :3])
    open3d.visualization.draw_geometries([point_cloud_4])

        #visuell the point cloud
    point_cloud_5 = open3d.geometry.PointCloud()
    point_cloud_5.points = open3d.utility.Vector3dVector(PointCloud_koordinate_5)
    point_cloud_5.colors = open3d.utility.Vector3dVector(colors5[:, :3])
    open3d.visualization.draw_geometries([point_cloud_5])

    if SavePCDFile is True:
    # #save the pcd file
        open3d.io.write_point_cloud(FileName +'.pcd', point_cloud)




def Visuell_PointCloud_per_batch_nocolor(sampled, SavePCDFile = False, FileName = None):
    #get only the koordinate from sampled
    sampled=sampled.cpu()
    if sampled.requires_grad==True:
        sampled=sampled.detach().numpy()
    else:
        sampled = np.asarray(sampled)
    PointCloud_koordinate_1 = sampled[0,:, 0:3]
    PointCloud_koordinate_2 = sampled[1,:, 0:3]
    PointCloud_koordinate_3 = sampled[2,:, 0:3]
    PointCloud_koordinate_4 = sampled[3,:, 0:3]
    PointCloud_koordinate_5 = sampled[4,:, 0:3]

    #visuell the point cloud
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(PointCloud_koordinate_1)

    open3d.visualization.draw_geometries([point_cloud])

        #visuell the point cloud
    point_cloud_2 = open3d.geometry.PointCloud()
    point_cloud_2.points = open3d.utility.Vector3dVector(PointCloud_koordinate_2)
    open3d.visualization.draw_geometries([point_cloud_2])

        #visuell the point cloud
    point_cloud_3 = open3d.geometry.PointCloud()
    point_cloud_3.points = open3d.utility.Vector3dVector(PointCloud_koordinate_3)
    open3d.visualization.draw_geometries([point_cloud_3])

        #visuell the point cloud
    point_cloud_4 = open3d.geometry.PointCloud()
    point_cloud_4.points = open3d.utility.Vector3dVector(PointCloud_koordinate_4)
    open3d.visualization.draw_geometries([point_cloud_4])

        #visuell the point cloud
    point_cloud_5 = open3d.geometry.PointCloud()
    point_cloud_5.points = open3d.utility.Vector3dVector(PointCloud_koordinate_5)
    open3d.visualization.draw_geometries([point_cloud_5])

    if SavePCDFile is True:
    # #save the pcd file
        open3d.io.write_point_cloud(FileName +'.pcd', point_cloud)



 
def main():

    # save_dir='/home/bi/study/thesis/data/current_finetune/A1/TrainingA1_1.npy'
    # if save_dir.split('.')[1]=='txt':
      
    #     patch_motor=np.loadtxt(save_dir)  
    # else:
    #     patch_motor=np.load(save_dir)   
    # print(len(patch_motor))
    # Visuell_PointCloud(patch_motor)


    file_path="/home/bi/study/thesis/data/synthetic/Dataset4"
    List_motor = os.listdir(file_path)
    if 'display.py' in List_motor :
        List_motor.remove('display.py')
    if '.DS_Store' in List_motor :
        List_motor.remove('.DS_Store')
    List_motor.sort()
    for dirs in List_motor :
        Motor_path = file_path + '/' + dirs
        # if "98" in dirs:
        if True:
            if dirs.split('.')[1]=='txt':
            
                patch_motor=np.loadtxt(Motor_path)  
            else:
                patch_motor=np.load(Motor_path)  
            if "Training" in dirs:     
                print(len(patch_motor))
                Visuell_PointCloud(patch_motor)


if __name__ == '__main__':
    main()