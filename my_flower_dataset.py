from typing import Any,Dict
import os
from torch.utils.data import Dataset
from PIL import Image
from scipy.io import loadmat
class Flowerdata(Dataset):
    def __init__(self,img_dir,transform=None)->None:
        super().__init__() #Maintain the consistency
        self.img_dir=img_dir
        self.img_infos=[] #This is a dictionary to store img path and class_id
        '''[{"path": path, "label": cls_id},...]'''
        self._get_img_info()
        self.transform=transform
        
    def __getitem__(self,index)->Any:
        img_info:Dict=self.img_infos[index]
        img_path=img_info["path"]
        label_id=img_info["label"]
        img=Image.open(img_path).convert("RGB")
        
        #Transform
        if self.transform is not None:
            img=self.transform(img)
            
        return img,label_id
            
            
    def __len__(self):
        return len(self.img_infos)
    
    def _get_img_info(self):
        """To get all the information from the mat file and the img file
                Then store them in dirctory form in the infos list"""
        label_file=os.path.join(os.path.dirname(self.img_dir),"imagelabels.mat")
        assert os.path.exists(label_file) #To check if the file exist
        
        label_array=loadmat(label_file)["labels"][0]
        label_array-=1 #Class start from zero
        
        for img_name in os.listdir(self.img_dir):
            path=os.path.join(self.img_dir,img_name)
            if not img_name[6:11].isdigit():
                continue
            img_id=int(img_name[6:11])
            cls_id=int(label_array[img_id-1])
            self.img_infos.append({"path":path,"label":cls_id})   