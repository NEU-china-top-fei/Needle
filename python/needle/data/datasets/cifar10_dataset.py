import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

def unpickle(file):
    with open(file ,"rb") as f:
        ret=pickle.load(f,encoding="bytes")
    return ret
class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        temp=None
        self.p=p
        self.transforms=transforms
        if not train:
            temp=unpickle(base_folder+"/test_batch")
            self.X=temp[b"data"]/255
            self.y=temp[b"labels"]
        else:
            temp_x=[]
            temp_y=[]
            for i in range(5):
                path=base_folder+"/data_batch_"+str(i+1)
                ret=unpickle(path)
                #print(ret.keys())
                temp_x.append(ret[b"data"])
                temp_y.append(ret[b"labels"])
            self.X=np.concatenate(temp_x,axis=0)/255
            self.y=np.concatenate(temp_y,axis=0)
        
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        s=(3,32,32)
        if not isinstance(index,int):
            s=(len(index),3,32,32)
        return self.apply_transforms(self.X[index]).reshape(s),self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.y)
        ### END YOUR SOLUTION
