# %%
#推理函数
import os
from glob import glob
import numpy as np
import torch
from union import union_model
from utils import c1_encoder, load_label, load_class
from dataloader import MyDataset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from test import evaluate
import argparse
# %%
'''
    根据路径读取数据
    文件的结构与老师给定的一致
    只需要将 os.getcwd() 改成相应的前缀即可
    :)
'''
'''
    配置相应的参数
'''
parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=1000,
                    help='epoch number')

parser.add_argument('--num_classes', type=int, default=11,
                    help='num_classes')

parser.add_argument('--predict_size', type=tuple, default=(32, 32),
                    help='predict_size')

parser.add_argument('--stride', type=int, default=16,
                    help='stride')

parser.add_argument('--sigma', type=float, default=4,
                    help='sigma')

parser.add_argument('--threshold', type=float, default=0.4,
                    help='threshold')

parser.add_argument('--model', type=str, default='/ADAM2/15 epoch model.pth',
                    help='path of model')

config = parser.parse_args(args=[])

'''
    根据高斯核函数定义的函数
    检查是否符合其定义
    :)
'''

# %%
device = ('cuda' if torch.cuda.is_available() else 'cpu')
def inference(Rmap,Offymap,Offxmap,Classmap):
    axisx=[]
    axisy=[]
    labels=[]
    for k in range(11):
        Rk=Rmap[k]
        Oyk=Offymap[k]
        Oxk=Offxmap[k]
        [gymax,gxmax]=divmod(np.argmax(Rk)+1,Rk.shape[0])
        if gxmax==0:
            gxmax+=Rk.shape[0]
            gymax-=1
        gxmax -= 1
        S=16
        x=(gxmax+0.5+Oxk[gymax][gxmax])*S
        y=(gymax+0.5+Oyk[gymax][gxmax])*S
        axisx.append(x)
        axisy.append(y)
        # tv=[0.5464,0.4019,0.2295,0.1194,0.2]
        # tdv5=[0.07,0.148,0.15,0.15,0.14,0.9999]


        tv=[0.496,0.555,0.61,0.7,0.2]
        tdv5=[0.07,0.148,0.15,0.15,0.14,0.9999]


        # tv = [0.5, 0.5, 0.5, 0.5, 0.5]
        # tdv5 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        if k<= 4:
            label=int(Classmap[k][gymax][gxmax]>tv[k])
            labels.append(label)
        else:
            label1=np.argmax(Classmap[(4*k-15):(4*k-11)][:,gymax,gxmax])
            label2=int(Classmap[k+24][gymax][gxmax]>tdv5[k-5])
            labels.append([label1,label2])
    return axisx,axisy,labels

# %%
def Model_Evaluation(model, valid_loader):
    model.eval()
    MSE_Loss = torch.nn.MSELoss()
    PRED_ACC, MSE = 0., 0.
    SUM_ACC=[0.]*17
    for vl_pth, vl_img, vl_htmp, vl_ofyp, vl_ofxp, vl_csmp, vl_mkmp, vl_iszt in valid_loader:
        # print([vl_pth[0][:-4]+'.txt'])
        _, gt_coords, gt_clsses, is_zhuiti= load_label([vl_pth[0][:-4] + '.txt'],noise=False)
        print(vl_pth)


        og_img = Image.open(vl_pth[0])
        (x_len, y_len) = og_img.size
        rmap, oymp, oxmp, clmp = model(vl_img.to(device))
        rmap, oymp, oxmp, clmp = rmap.cpu().detach().squeeze(0).numpy(), oymp.cpu().detach().squeeze(0).numpy(), oxmp.cpu().detach().squeeze(0).numpy(), clmp.cpu().detach().squeeze(0).numpy()
        out_rmap = rmap[0]
        gt_rmap = vl_htmp.squeeze(0)[1]

    
        x_pred, y_pred, lbl_pred = inference(rmap, oymp, oxmp, clmp)
        x_pred, y_pred = np.array(x_pred).astype(np.int32()), np.array(y_pred).astype(np.int32())
        coord_pred = np.append(x_pred[:,None], y_pred[:,None], axis=1)
        gt_coords, pred_coords = torch.from_numpy(np.array(gt_coords[::])).float(), torch.from_numpy(coord_pred).float().unsqueeze(0)
        # print(gt_coords, pred_coords)
        gt_coords[:,:,0]=gt_coords[:,:,0]*x_len/512
        gt_coords[:, :, 1] = gt_coords[:, :, 1] * y_len / 512
        pred_coords[:,:,0]=pred_coords[:,:,0]*x_len/512
        pred_coords[:, :, 1] = pred_coords[:, :, 1] * y_len / 512
        mse_loss = MSE_Loss(pred_coords, gt_coords)


        # print(gt_clsses[0])
        # print(lbl_pred)
        sum_acc=[0]*17
        pred_acc = 0
        for i in range(11):
            acc = 0
            acc1=0
            acc2=0
            if i < 5:
                acc = (gt_clsses[0][i] == lbl_pred[i])
                pred_acc += acc
                sum_acc[i]+=acc
            else:
                acc1 += int(gt_clsses[0][i] == lbl_pred[i][0])
                acc2 += int(gt_clsses[0][i+6] == lbl_pred[i][1])
                pred_acc += (acc1+acc2)
                sum_acc[i]+=acc1
                sum_acc[i+6]+=acc2
        
        pred_acc /= 17
        PRED_ACC += pred_acc
        for i in range(17):
            SUM_ACC[i] += sum_acc[i]
        MSE += mse_loss
        print("真实类别：",gt_clsses)
        print("预测类别：",lbl_pred)
        print("判断正确的分类（1为判断正确）：",sum_acc)
        print("准确率",pred_acc)
        print("真实坐标：",gt_coords)
        print("预测坐标：",pred_coords)
        print("Mseloss:",mse_loss)
        print('\n')
        t=vl_pth[0].split('\\')
        t=t[len(t)-1].split('.jpg')[0]
        fw=open(os.getcwd()+'\\test\\'+t+'_pre.txt','w')
        a={'identification':1,'disc':2}
        b={'identification':1,'vertebra':2}
        names=['L1','L2','L3','L4','L5','T12-L1','L1-L2','L2-L3','L3-L4','L4-L5','L5-S1']
        c=['v1','v2','v3','v4']
        for i in range(11):
            q=pred_coords[0][i].tolist()
            fw.write(str(int(q[0]))+','+str(int(q[1]))+',')
            if i<=4:
                b['identification']=names[i]
                if lbl_pred[i]==1:
                    b['vertebra']='v2'
                else:
                    b['vertebra'] = 'v1'
                fw.write(str(b))
            else:
                a['identification'] = names[i]
                if lbl_pred[i][1]==0:
                    a['disc']=c[lbl_pred[i][0]]
                else:
                    a['disc'] = c[lbl_pred[i][0]]+'v5'
                fw.write(str(a))
            fw.write("\n")
        fw.close()

    length = len(valid_loader)
    predict_accuracy = PRED_ACC / length
    predict_mse_loss = MSE / length
    predict_everyaccuracy=np.array(SUM_ACC)/float(length)
    print("every_acc:",predict_everyaccuracy)
    print("PRED_ACC: ", predict_accuracy, "\nMSE: ", predict_mse_loss)

# %%
if __name__ == '__main__':
    print('Loading Model ', os.getcwd()+config.model, '...')
    model = torch.load(os.getcwd()+config.model)
    pics_pth = os.path.join(os.getcwd(), 'test/data')
    lbls_pth = glob(os.path.join(pics_pth, '*.txt'))
    valid_txt_labels = load_label(lbls_pth)
    valid_img_files, valid_heatmap, valid_offymap, valid_offxmap, valid_clssmap, valid_maskmap, valid_is_zhuiti= load_class(valid_txt_labels)
    valid_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    valid_set = MyDataset(valid_img_files, valid_heatmap, valid_offymap, valid_offxmap, valid_clssmap, valid_maskmap, valid_is_zhuiti, transform=valid_transform)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=True, pin_memory=True)
    Model_Evaluation(model, valid_loader)
# %%
