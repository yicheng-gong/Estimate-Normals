# package used
import python.NormalEstimatorHoughCNN as Estimator
import numpy as np
from tqdm import *
import torch
from torch.autograd import Variable
import os
import trimesh
import scipy
import matplotlib.pyplot as plt

class normal_Est:
    
    def __init__(self):
        pass

    def load_model(self,K,scale_number):
        model_file_name = "state_dict.pth"
        mean_file_name = "dataset_mean.npz"
        if scale_number == 1:
            Ks=np.array([K], dtype=int)
            import models.model_1s as model_1s
            traning_result_path = "training_result/model_1s"
            model = model_1s.load_model(os.path.join(traning_result_path, model_file_name))
            mean = np.load(os.path.join(traning_result_path, mean_file_name))["arr_0"]
        elif scale_number == 3:
            Ks=np.array([K,K/2,K*2], dtype=int)
            import models.model_3s as model_3s
            traning_result_path = "training_result/model_3s"
            model = model_3s.load_model(os.path.join(traning_result_path, model_file_name))
            mean = np.load(os.path.join(traning_result_path, mean_file_name))["arr_0"]
        elif scale_number == 5:
            Ks=np.array([K,K/4,K/2,K*2,K*4], dtype=int)
            import models.model_5s as model_5s
            traning_result_path = "training_result/model_5s"
            model = model_5s.load_model(os.path.join(traning_result_path, model_file_name))
            mean = np.load(os.path.join(traning_result_path, mean_file_name))["arr_0"]
            
        return Ks, model, mean

    def test_save_path(self,scale_number):
        if scale_number == 1:
            test_result_path = "test_result/model_1s"
        elif scale_number == 3:
            test_result_path = "test_result/model_3s"
        elif scale_number == 5:
            test_result_path = "test_result/model_5s"
        if not os.path.exists(test_result_path):
            os.makedirs(test_result_path)
        
        
        return test_result_path
            
        
    def est_normal(self,input_file_name, sample_num = 0, K = 256, scale_number = 1, batch_size = 256):
        
        
        
        USE_CUDA = True
        input_file_path = "sources/meshes"
        input_file = os.path.join(input_file_path, input_file_name + ".obj")
        if not os.path.exists(input_file):
            print("no file exists! Check the path!")
            
        # mesh load
        input_mesh = trimesh.load(input_file)
        input_mesh_norm = input_mesh.vertex_normals
        input_mesh_points = input_mesh.sample(sample_num)
        input_points = np.zeros([input_mesh.vertices.shape[0] + input_mesh_points.shape[0], 6])
        input_points[:,:3] = np.r_[input_mesh.vertices, input_mesh_points]
        input_points[:input_mesh.vertices.shape[0], 3:] = input_mesh_norm
        
        K_number = "_K" + str(K)
        batch_size_number = "_bs" + str(batch_size)
        # store as .xyz file
        xyz_file_path = "sources/xyzfiles"
        xyz_file_name = input_file_name + K_number + batch_size_number + ".xyz"
        xyz_file = os.path.join(xyz_file_path, xyz_file_name)
        np.savetxt(xyz_file, input_points, delimiter=' ', fmt='%f')

        # create the estimator
        estimator = Estimator.NormalEstimatorHoughCNN()

        # load the file
        estimator.loadXYZ(xyz_file)

        Ks, model, mean = self.load_model(K,scale_number)

        # set the neighborhood size
        estimator.setKs(Ks)

        # initialize
        estimator.initialize()

        # choose device
        if torch.cuda.is_available():
            USE_CUDA = True
        else:
            USE_CUDA = False

        # convert model to cuda if needed
        if USE_CUDA:
            model.cuda()
        model.eval()
        # iterate over the batches
        with torch.no_grad():
            for pt_id in tqdm(range(0,estimator.getPCSize(), batch_size)):
                bs = batch_size
                batch = estimator.getBatch(pt_id, bs) - mean[None,:,:,:]
                batch_th = torch.Tensor(batch)
                if USE_CUDA:
                    batch_th = batch_th.cuda()
                estimations = model.forward(batch_th)
                estimations = estimations.cpu().data.numpy()
                estimator.setBatch(pt_id,bs,estimations.astype(float))

        # save the estimator
        save_path = self.test_save_path(scale_number)
        save_path = os.path.join(save_path,xyz_file_name)
        estimator.saveXYZ(save_path)
        
    def evaluate(self, file_name, scale_number = 1, K = 256, batch_size = 256):
        # the origin data path
        origin_file_path = "sources/xyzfiles"
        origin_file = os.path.join(origin_file_path, file_name + ".xyz")
        if not os.path.exists(origin_file):
            print("No file exists! Check the path!")

        K_number = "_K" + str(K)
        batch_size_number = "_bs" + str(batch_size)
        # the evaluate data path
        eva_file_path = self.test_save_path(scale_number)
        eva_file = os.path.join(eva_file_path, file_name + K_number + batch_size_number + ".xyz")
        if not os.path.exists(eva_file):
            print("No file exists! Check the path!")
            
        # load data
        origin_points = np.loadtxt(origin_file)
        eva_points = np.loadtxt(eva_file)
        origin_points = origin_points[origin_points[:,3] != 0]
        eva_points = eva_points[:origin_points.shape[0],:]

        # split norm point
        origin_norm = origin_points[:, 3:]
        eva_norm = eva_points[:, 3:]

        # compute angle
        cos_theta = np.sum(origin_norm * eva_norm, axis=1) / (np.linalg.norm(origin_norm, axis=1) * np.linalg.norm(eva_norm, axis=1))
        theta = np.arccos(cos_theta)
        theta[theta>np.pi/2] = np.pi - theta[theta>np.pi/2]

        # compute prob
        angle = []
        prob = []
        for i in range(90):
            less_count = theta[theta < np.deg2rad(i)].shape[0]
            angle.append(i)
            prob.append(less_count/theta.shape[0])
        
        return angle, prob


