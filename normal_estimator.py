# package used
import normal_estimator_cpp.NormalEstimatorHoughCNN as Estimator
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

    def load_model(self, K, scale_number, use_paper_model):
        model_file_name = "model.pth"
        mean_file_name = "mean.npz"
        if scale_number == 1:
            Ks=np.array([K], dtype=int)
            import cnn_models.model_1s as model_1s
            if use_paper_model:
                model_path = "models_in_paper/model_1s_boulch_SGP2016"
            else:
                model_path = "models_reproduced/model_1s"
            model = model_1s.load_model(os.path.join(model_path, model_file_name))
            mean = np.load(os.path.join(model_path, mean_file_name))["arr_0"]
        elif scale_number == 3:
            Ks=np.array([K,K/2,K*2], dtype=int)
            import cnn_models.model_3s as model_3s
            if use_paper_model:
                model_path = "models_in_paper/model_3s_boulch_SGP2016"
            else:
                model_path = "models_reproduced/model_3s"
            model = model_3s.load_model(os.path.join(model_path, model_file_name))
            mean = np.load(os.path.join(model_path, mean_file_name))["arr_0"]
        elif scale_number == 5:
            Ks=np.array([K,K/4,K/2,K*2,K*4], dtype=int)
            import cnn_models.model_5s as model_5s
            if use_paper_model:
                model_path = "models_in_paper/model_5s_boulch_SGP2016"
            else:
                model_path = "models_reproduced/model_5s"
            model = model_5s.load_model(os.path.join(model_path, model_file_name))
            mean = np.load(os.path.join(model_path, mean_file_name))["arr_0"]
            
        return Ks, model, mean

    def estimate_result_path(self, scale_number, use_paper_model):
        if scale_number == 1:
            if use_paper_model:
                est_result_path = "estimate_results/paper_model_1s"
            else:
                est_result_path = "estimate_results/model_1s"
        elif scale_number == 3:
            if use_paper_model:
                est_result_path = "estimate_results/paper_model_3s"
            else:
                est_result_path = "estimate_results/model_3s"
        elif scale_number == 5:
            if use_paper_model:
                est_result_path = "estimate_results/paper_model_5s"
            else:
                est_result_path = "estimate_results/model_5s"
        if not os.path.exists(est_result_path):
            os.makedirs(est_result_path)
        
        return est_result_path
    
    def add_gaussian_noise(self,points,noise_scale):
        # points: numpy array with shape (N, 3), representing a point cloud

        # calculate the mean of each coordinate
        mean_x = np.mean(points[:,0])
        mean_y = np.mean(points[:,1])
        mean_z = np.mean(points[:,2])

        # calculate the standard deviation of the Gaussian noise for each coordinate
        std_x = np.abs(mean_x) * noise_scale
        std_y = np.abs(mean_y) * noise_scale
        std_z = np.abs(mean_z) * noise_scale

        # generate Gaussian noise for each coordinate
        noise_x = np.random.normal(0, std_x, size=points.shape[0])
        noise_y = np.random.normal(0, std_y, size=points.shape[0])
        noise_z = np.random.normal(0, std_z, size=points.shape[0])

        # add noise to each coordinate
        noisy_points = points.copy()
        noisy_points[:,0] += noise_x
        noisy_points[:,1] += noise_y
        noisy_points[:,2] += noise_z

        return noisy_points
            
        
    def est_normal(self, input_file_name, noise_scale = 0, sample_num = 0, K = 100, scale_number = 1, batch_size = 256, use_paper_model = False):
        
        input_file_path = "sources/meshes"
        input_file = os.path.join(input_file_path, input_file_name + ".obj")
        if not os.path.exists(input_file):
            print("no file exists! Check the path!")
            
        # mesh load
        input_mesh = trimesh.load(input_file)
        input_mesh_norm = input_mesh.vertex_normals
        input_mesh_points = input_mesh.sample(sample_num)
        noisy_vertices = self.add_gaussian_noise(input_mesh.vertices,noise_scale)
        input_points = np.zeros([input_mesh.vertices.shape[0] + input_mesh_points.shape[0], 6])
        input_points[:,:3] = np.r_[noisy_vertices, input_mesh_points]
        input_points[:noisy_vertices.shape[0], 3:] = input_mesh_norm
        
        # store as .xyz file
        K_number = "_K" + str(K)
        batch_size_number = "_bs" + str(batch_size)
        noise_scale_number = "_ns" + str(noise_scale)
        xyz_file_path = "sources/xyzfiles"
        if not os.path.exists(xyz_file_path):
            os.makedirs(xyz_file_path)
        xyz_file_name = input_file_name + K_number + batch_size_number + noise_scale_number +".xyz"
        xyz_file = os.path.join(xyz_file_path, xyz_file_name)
        np.savetxt(xyz_file, input_points, delimiter=' ', fmt='%f')

        # create the estimator
        estimator = Estimator.NormalEstimatorHoughCNN()

        # load the file
        estimator.loadXYZ(xyz_file)

        Ks, model, mean = self.load_model(K, scale_number, use_paper_model)

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
        save_path = self.estimate_result_path(scale_number, use_paper_model)
        save_path = os.path.join(save_path,xyz_file_name)
        estimator.saveXYZ(save_path)
        
    def evaluate(self, file_name, noise_scale = 0, scale_number = 1, K = 100, batch_size = 256, use_paper_model = False):
        # the origin data path
        K_number = "_K" + str(K)
        batch_size_number = "_bs" + str(batch_size)
        noise_scale_number = "_ns" + str(noise_scale)
        origin_file_path = "sources/xyzfiles"
        origin_file = os.path.join(origin_file_path, file_name + K_number + batch_size_number + noise_scale_number + ".xyz")
        if not os.path.exists(origin_file):
            print("No file exists! Check the path!")

        # the evaluate data path
        eva_file_path = self.estimate_result_path(scale_number,use_paper_model)
        eva_file = os.path.join(eva_file_path, file_name + K_number + batch_size_number + noise_scale_number + ".xyz")
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

        # compute RMS
        RMS = np.linalg.norm(np.rad2deg(theta))/theta.shape[0]
        
        # compute prob
        angle = []
        prob = []
        for i in range(91):
            less_count = theta[theta < np.deg2rad(i)].shape[0]
            angle.append(i)
            prob.append(less_count/theta.shape[0])
        
        # compute prob of angle devation less than 5 and 10 degree
        
        
        return RMS, angle, prob


