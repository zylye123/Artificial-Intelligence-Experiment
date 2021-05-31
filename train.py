# %%
import os

import torch.optim
from utils import load_label, load_class
from union import union_model
from dataloader import MyDataset
from torch.utils.data import DataLoader
import argparse
from glob import glob
import torchvision.transforms as transforms
from PIL import Image
from loss import *
import time
from thop import profile

CUDA_LAUNCH_BLOCKING=1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cpu'

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=40,
                    help='epoch number')

parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')


config = parser.parse_args(args=[])




lr = config.lr
epochs = config.epochs

train_pth = os.path.join(os.getcwd(), 'train/data')
test_pth = os.path.join(os.getcwd(), 'test/data')
train_txts = glob(os.path.join(train_pth, '*.txt'))
test_txts = glob(os.path.join(test_pth, '*.txt'))

train_txt_labels = load_label(train_txts)
train_img_files, train_heatmap, train_offymap, train_offxmap, train_clssmap, train_maskmap, train_is_zhuiti= load_class(train_txt_labels)
test_txt_labels = load_label(test_txts)
test_img_files, test_heatmap, test_offymap, test_offxmap, test_clssmap, test_maskmap, test_is_zhuiti= load_class(test_txt_labels)

# %%

'''
    该模块验证读取与模型输出的一致性
    已全部注释
'''
# model = union_model(3)
# model = model.to(device)
# matrix = torch.zeros((1, 3, 512, 512))
# matrix = matrix.to(device)
# (out1, out2, out3, out4) = model(matrix)




# img_file, heatmap, offymap, offxmap, clssmap, maskmap, is_zhuiti =  train_img_files[0], train_heatmap[0], train_offymap[0], train_offxmap[0], train_clssmap[0], train_maskmap[0], train_is_zhuiti[0]

# print(type(train_img_files))
# print(type(train_heatmap))
# print(type(train_offymap))
# print(type(train_offxmap))
# print(type(train_clssmap))
# print(type(train_maskmap))
# print(type(train_is_zhuiti))


# print('Test Output: /train_img[0]:')
# print(img_file)
# print(heatmap.shape, out1.shape)
# print(offymap.shape, out2.shape)
# print(offxmap.shape, out3.shape)
# print(clssmap.shape, out4.shape)
# print(maskmap.shape)0
# print(len(is_zhuiti))


# %%
'''
    观察 loader 返还的每个部分的输出
'''
train_transform = transforms.Compose([ transforms.RandomRotation(5, Image.BILINEAR),transforms.Resize((512, 512)), transforms.RandomCrop((505,505)),transforms.Resize((512, 512)), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

train_set = MyDataset(train_img_files, train_heatmap, train_offymap, train_offxmap, train_clssmap, train_maskmap ,train_is_zhuiti, transform=train_transform)
test_set = MyDataset(test_img_files, test_heatmap, test_offymap, test_offxmap, test_clssmap, test_maskmap, test_is_zhuiti, transform=test_transform)

train_loader = DataLoader(train_set, batch_size=2, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=2, shuffle=True, pin_memory=True)
i = 0

for _, tr_img, tr_htmp, tr_ofyp, tr_ofxp, tr_csmp, tr_mkmp, tr_iszt in train_loader:
    print('%dth batch: '% (i))
    print(tr_img.shape)
    print(tr_ofyp.shape)
    print(tr_ofxp.shape)
    print(tr_csmp.shape)
    print(tr_mkmp.shape)
    print(torch.sum(tr_mkmp[:,0,::], dim=(1,2)).shape)
    print(len(tr_iszt))
    print(tr_iszt)
    print('### A Batch ###')
    print('\n\n')
    i += 1
    break



# %%
model = union_model(1)
# model = torch.load('60 epoch model.pth')
# input = torch.randn(2, 1, 512,512)
# input.to(device)
# flops, params = profile(model, inputs=(input, ))
model.to(device)
model.train()
model.float()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)


# %%

loss_1 = nn.MSELoss()
print('training on', device)
torch.autograd.set_detect_anomaly(True)
for t in range(epochs):
    train_loss, train_r_loss, train_oy_loss, train_ox_loss, train_c_loss = 0, 0, 0, 0, 0
    start_time = time.time()
    j, m = 0, 0
    for _, tr_img, tr_htmp, tr_ofyp, tr_ofxp, tr_csmp, tr_mkmp, tr_iszt in train_loader:
        j += 1
        loss_r, loss_oy, loss_ox, loss_c = 0, 0, 0, 0
        start = time.time()
        tr_img = tr_img.to(device)
        tr_htmp, tr_ofyp, tr_ofxp, tr_csmp, tr_mkmp = tr_htmp.to(device).float(), tr_ofyp.to(device).float(), tr_ofxp.to(device).float(), tr_csmp.to(device).float(), tr_mkmp.to(device).float()
        pr_htmp, pr_ofyp, pr_ofxp, pr_csmp = model(tr_img)

        maskmap=tr_mkmp.cpu().detach().numpy()
        n = tr_img.shape[0]
        m += n
        for i in range(11):
            w = torch.sum(tr_mkmp[:,i,:,:], dim = (1,2))
            loss_r += loss_1(pr_htmp[:,i,:,:], tr_htmp[:,i,:,:])
            for j in range(tr_csmp.shape[0]):
                loss_oy += float(1/(w[j])) *(Lossfuction_offy(tr_ofyp[j,i,:,:].squeeze()[np.squeeze(maskmap[j,i,:,:]==1)], pr_ofyp[j,i,:,:].squeeze()[[np.squeeze(maskmap[j,i,:,:]==1)]]))
                loss_ox += float(1/(w[j])) *(Lossfuction_offx(tr_ofxp[j,i,:,:].squeeze()[np.squeeze(maskmap[j,i,:,:]==1)], pr_ofxp[j,i,:,:].squeeze()[[np.squeeze(maskmap[j,i,:,:]==1)]]))
            loss_c +=  Lossfuction4(tr_csmp, pr_csmp, maskmap, i, w)
            # 交叉熵的计算方式 ———— 此处不用再采用插入空矩阵的方式
        #print(_)
        loss = ((loss_r + 2*loss_oy + 2*loss_ox + 0.001*loss_c) / 11)
        
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        train_loss += loss.cpu().item() 
        train_r_loss += loss_r.cpu().item() 
        train_oy_loss += loss_oy.cpu().item()
        train_ox_loss += loss_ox.cpu().item()
        train_c_loss += loss_c.cpu().item()
    
    if (t+1)%5 == 0:
        torch.save(model, '%d epoch model.pth'%(t+1))
        print('model saved!')
        #evaluate(model, test_loader)
    print('epoch:%d>>train_loss: %.2f, r_loss: %.4f\t off_y_loss: %.4f\t off_x_loss: %.4f\t c_loss: %.4f\t time: %d s'
            %(t+1,  train_loss/m, train_r_loss/m, train_oy_loss/m, train_ox_loss/m, train_c_loss/m, time.time()-start_time))







# %%
