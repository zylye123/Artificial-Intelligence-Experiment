# %%
import os
from glob import glob
from torch.utils.data import Dataset,DataLoader
from PIL import Image
train_pth = os.path.join(os.getcwd(), 'train/data')
train_img = glob(os.path.join(train_pth, '*.jpg'))
# %%
class MyDataset(Dataset):
    def __init__(self, pth, _heatmap, _offymap, _offxmap, _clssmap, _maskmap, is_zhuiti, transform=None):
        self.pth = pth
        
        self._heatmap = _heatmap  
        self._offymap = _offymap
        self._offxmap = _offxmap
        self._clssmap = _clssmap
        self._maskmap = _maskmap
        self.is_zhuiti = is_zhuiti

        self.transform = transform
    
    def __len__(self):
        return len(self.pth)

    def __getitem__(self, idx):
        pth = self.pth[idx]
        _ = pth
        img = Image.open(pth)
        
        _heatmap = self._heatmap[idx]  # 对应向量点乘得到相关程度相对较高的部分
        _offymap = self._offymap[idx]
        _offxmap = self._offxmap[idx]
        _clssmap = self._clssmap[idx]
        _maskmap = self._maskmap[idx]
        is_zhuiti = self.is_zhuiti[idx]

        # _heatmap = _heatmap.unsqueeze(0)
        # _offymap = _offymap.unsqueeze(0)
        # _offxmap = _offxmap.unsqueeze(0)
        # _clssmap = _clssmap.unsqueeze(0)
        # is_zhuiti = is_zhuiti.unsqueeze(0)

        if self.transform:
            img = self.transform(img)
        
        return _, img, _heatmap, _offymap, _offxmap, _clssmap, _maskmap, is_zhuiti



# %%

# %%
