import torch.utils.data as data
import os.path
import numpy as np
import random
import torch
random.seed(0)

class PairedImages_w_nameList(data.Dataset):
    '''
    can act as supervised or un-supervised based on flists
    '''
    def __init__(self, root1, root2, flist1, flist2, transform1=None, transform2=None, do_aug=False):
        self.root1 = root1
        self.root2 = root2
        self.flist1 = flist1
        self.flist2 = flist2
        self.transform1 = transform1
        self.transform2 = transform2
        self.do_aug = do_aug
    def __getitem__(self, index):
        impath1 = self.flist1[index]
        img1 = np.load(os.path.join(self.root1, impath1))
        impath2 = self.flist2[index]
        img2 = np.load(os.path.join(self.root2, impath2))
        if self.transform1 is not None:
            img1 = self.transform1(img1)
            img2 = self.transform2(img2)
        if self.do_aug:
            p1 = random.random()
            if p1<0.5:
                img1, img2 = torch.fliplr(img1), torch.fliplr(img2)
            p2 = random.random()
            if p2<0.5:
                img1, img2 = torch.flipud(img1), torch.flipud(img2)
        return img1, img2
    def __len__(self):
        return len(self.flist1)