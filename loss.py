# %%
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


count_class=np.array([1.01,0.99,1.1,0.9,1.1,0.9,1.1,0.9,1.1,0.9,
                      0.9,1.1,1.1,1,
                      0.9,1.01,1.01,1.1,
                      0.99,0.99,1.01,0,
                      0.99,0.99,0.99,1,
                      0.9,0.99,0.99,1,
                      0.99,0.99,0.99,1,
                      0.99,1.1,0.9,1.1,0.9,1.1,1,0,0.9,1.1,1,0])
#除号右边是每个类别的个数，前五个是二分类，所以占10个位置，后六个是二分类占12个位置，中间4分类各占4个位置

def Lossfuction1():#第k个关键点粗定位损失函数
    loss1=nn.MSELoss()
    return loss1
def Lossfuction_offy(offymap_true,offymap):#第k个关键点细回归y损失函数
    loss2=F.smooth_l1_loss(offymap_true, offymap, reduction='mean')
    return loss2
def Lossfuction_offx(offxmap_true,offxmap):#第k个关键点细回归x损失函数
    loss3=F.smooth_l1_loss(offxmap_true, offxmap, reduction='mean')
    return loss3
def BCE():
    loss=nn.BCELoss()
    return loss
def CE():
    loss=nn.CrossEntropyLoss()
    return loss
def Lossfuction4(Classmap_true,Classmap,maskmap, idx, weight):#第k个关键点分类损失函数，锥体和椎间盘判别方式未写
    loss, batch = 0, Classmap_true.shape[0]
    if idx<=4:
        ls_func = BCE()
        for j in range(batch):
            q=Classmap_true[j,idx,:,:][[maskmap[j][idx]==1]]
            loss += float(1/weight[j]) * ls_func(Classmap[j,idx,:,:][[maskmap[j][idx]==1]],Classmap_true[j,idx,:,:][[maskmap[j][idx]==1]])*count_class[2*idx+int(q[0])]
    else:
        ls_bce = BCE()
        ls_ce = CE()
        for j in range(batch):
            o=Classmap[j,4*idx-15:4*idx-11,:,:][[[maskmap[j][idx]==1]*4]].reshape([4,-1]).t()
            q=Classmap_true[j,idx,:,:].long()[[maskmap[j][idx]==1]]
            p=Classmap_true[j, 6 + idx, :, :][[maskmap[j, idx, :, :] == 1]]
            #print(o)
            loss += float(1 / (weight[j])) * (ls_bce(Classmap[j, 24 + idx, :, :][[maskmap[j, idx, :, :] == 1]],
                                                     Classmap_true[j, 6 + idx, :, :][
                                                         [maskmap[j, idx, :, :] == 1]])*count_class[2*(idx-5)+34+int(p[0])] + ls_ce(o, q)*count_class[4*(idx-5)+10+int(q[0])])
            #print(q)
            #loss += float(1/(weight[j])) *(ls_bce(Classmap[j,24+idx,:,:][[maskmap[j,idx,:,:]==1]],Classmap_true[j,6+idx,:,:][[maskmap[j,idx,:,:]==1]])+ls_ce(o,q))
    return loss




if __name__ == '__main__':
    print('loss')
# %%
