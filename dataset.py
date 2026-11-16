import os
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

class KITTI(data.Dataset):
    def __init__(self, root="data/kitti", seq="00", n_neg=10, resize_shape=(800, 128), max_frames=3000, pose_cols=(3, 7)):
        super().__init__()
        self.root = root
        self.n_neg = n_neg
        self.distThr = 5
        self.resize_shape = resize_shape
        
        pose_path = os.path.join(root, "poses", seq+".txt")
        self.poses = np.loadtxt(pose_path)
        self.poses = np.array([self.poses[:,pose_cols[0]], self.poses[:,pose_cols[1]]]).transpose()
        
        self.pose_q = []
        self.pose_db = []
        self.img_q = []
        self.img_db = []
        self.img_sam_q = []
        self.img_sam_db = []
        for i in range(min(max_frames, len(self.poses))):
            self.img_q.append('sequences/'+seq+'/image_2/'+f"{i:06d}"+'.png')
            self.img_sam_q.append('sequences/'+seq+'/imgSAM/'+f"{i:06d}"+'.png')
            self.pose_q.append(self.poses[i])
        for i in range(min(max_frames, len(self.poses))):
            self.img_db.append('sequences/' + seq + "/lidar_D50t/" + f"{i:06d}" + '.tiff')
            self.img_sam_db.append('sequences/'+seq+'/lidarSAM_t/'+f"{i:06d}"+'.tiff')
            self.pose_db.append(self.poses[i])
            
            
        
        self.pose_q  = np.array(self.pose_q)
        self.pose_db = np.array(self.pose_db)
        
    def __getitem__(self, index):
        raise NotImplementedError("Use hcmc_kitti.KITTIMultiPosGeoNeg for HCMC sampling")

    def __len__(self):
        return len(self.img_q)
