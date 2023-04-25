# package used
import normal_estimator_cpp.NormalEstimatorHoughCNN as Estimator
import numpy as np
from tqdm import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import os
import trimesh
import scipy
import matplotlib.pyplot as plt
import pickle

class normal_Est:
    
    # init parameter
    set_scale_number = 1
    set_batch_size = 256
    set_K = 100
    set_noise_scale = 0
    use_paper_model = False
    density_sensitive = False
    use_ResNet = False
    set_dataset_size = 100000
    
    # choose device
    if torch.cuda.is_available():
        USE_CUDA = True
        print("using device: CUDA")
    else:
        USE_CUDA = False
        print("using device: CPU")
    
    def __init__(self):
        pass
    
    def parameter_init(self):
        self.set_scale_number = 1
        self.set_batch_size = 256
        self.set_K = 100
        self.set_noise_scale = 0
        self.use_paper_model = False
        self.density_sensitive = False
        self.use_ResNet = False
        self.set_dataset_size = 100000
        
    def dataset_create(self, dataset_path, estimator):
        print("creating dataset")
        count = 0
        dataset = np.zeros((self.set_dataset_size, self.set_scale_number, 33,33))
        targets = np.zeros((self.set_dataset_size, 2))
        for i in tqdm(range(0, self.set_dataset_size, self.set_batch_size), ncols=80):
            nbr, batch, batch_targets = estimator.generateTrainAccRandomCorner(self.set_batch_size)
            if count+nbr > self.set_dataset_size:
                nbr = self.set_dataset_size - count
            dataset[count:count+nbr] = batch[0:nbr]
            targets[count:count+nbr] = batch_targets[0:nbr]
            count += nbr
            if(count >= self.set_dataset_size):
                break
            
        mean = dataset.mean(axis=0)
        print(mean.shape)
        dataset = {"input":dataset, "targets":targets, "mean":mean}
        print("  saving")
        pickle.dump(dataset, open(dataset_path, "wb" ))
        print("-->done")
    
    def generate_file_name(self):
        K_name = "_K" + str(self.set_K)
        scale_number_name = "_sn" + str(self.set_scale_number)
        batch_size_name = "_bs" + str(self.set_batch_size)
        if self.density_sensitive:
            density_sensitve_name = "_ds1"
        else:
            density_sensitve_name = "_ds0"
            
        fn_ = K_name + scale_number_name + batch_size_name + density_sensitve_name
        
        return fn_
    
    def model_init(self):
        if self.set_scale_number == 1:
            Ks=np.array([self.set_K], dtype=int)
            if self.use_ResNet:
                import cnn_models.ResNet.model_1s as modelCNN
                ResNet_name = "_ResNet"
            else:
                import cnn_models.LeNet.model_1s as modelCNN
                ResNet_name = "_LeNet"
            if self.use_paper_model:
                model_path = "models_in_paper/model_1s_boulch_SGP2016"
            else:
                model_path = "models_reproduced/model_1s" + ResNet_name
        elif self.set_scale_number == 3:
            Ks=np.array([self.set_K, self.set_K/2, self.set_K*2], dtype=int)
            if self.use_ResNet:
                import cnn_models.ResNet.model_3s as modelCNN
                ResNet_name = "_ResNet"
            else:
                import cnn_models.LeNet.model_3s as modelCNN
                ResNet_name = "_LeNet"
            if self.use_paper_model:
                model_path = "models_in_paper/model_3s_boulch_SGP2016"
            else:
                model_path = "models_reproduced/model_3s"+ ResNet_name
        elif self.set_scale_number == 5:
            Ks=np.array([self.set_K, self.set_K/4, self.set_K/2, self.set_K*2, self.set_K*4], dtype=int)
            if self.use_ResNet:
                import cnn_models.ResNet.model_5s as modelCNN
                ResNet_name = "_ResNet"
            else:
                import cnn_models.LeNet.model_5s as modelCNN
                ResNet_name = "_LeNet"
            if self.use_paper_model:
                model_path = "models_in_paper/model_5s_boulch_SGP2016"
            else:
                model_path = "models_reproduced/model_5s" + ResNet_name
        
        return Ks, model_path, modelCNN

    def estimate_result_path(self):
        if self.set_scale_number == 1:
            if self.use_paper_model:
                est_result_path = "estimate_results/paper_model_1s"
            else:
                est_result_path = "estimate_results/model_1s"
        elif self.set_scale_number == 3:
            if self.use_paper_model:
                est_result_path = "estimate_results/paper_model_3s"
            else:
                est_result_path = "estimate_results/model_3s"
        elif self.set_scale_number == 5:
            if self.use_paper_model:
                est_result_path = "estimate_results/paper_model_5s"
            else:
                est_result_path = "estimate_results/model_5s"
        if not os.path.exists(est_result_path):
            os.makedirs(est_result_path)
        
        return est_result_path
    
    def add_gaussian_noise(self, points):
        # points: numpy array with shape (N, 3), representing a point cloud

        mean_distance = np.linalg.norm(np.mean(points,axis = 0))
        std = self.set_noise_scale * mean_distance
        noise = np.random.normal(0,std,points.shape)
        noisy_points = points + noise
        return noisy_points
            
    def model_training(self): 
        # training parameter
        drop_learning_rate = 0.5
        learning_rate = 0.1
        epoch_max = 40
        decrease_step = 4
        
        # faster computation times
        torch.backends.cudnn.benchmark = True
        
        # create the estimator
        estimator = Estimator.NormalEstimatorHoughCNN()
        
        # Ks computing and model selecting    
        Ks, model_path, modelCNN = self.model_init()
        estimator.setKs(Ks)
        estimator.setDensitySensitive(self.density_sensitive)
        net = modelCNN.create_model()
        
        # training result store
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_result_name = "model" + self.generate_file_name() + ".pth"
        model_result_path = os.path.join(model_path, model_result_name)
        
        
        if not os.path.exists(model_result_path):
            
            # configure dataset
            dataset_path = "dataset"
            dataset_name = "dataset.p"
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            dataset_path = os.path.join(dataset_path, dataset_name)
            self.dataset_create(dataset_path, estimator)
                
            # dataset loading
            print("loading the model")
            dataset = pickle.load(open(dataset_path, "rb" ))
            dataset["input"] -= dataset["mean"][None,:,:,:]
            input_data = torch.from_numpy(dataset["input"]).float()
            target_data = torch.from_numpy(dataset["targets"]).float()
            ds = torch.utils.data.TensorDataset(input_data, target_data)
            ds_loader = torch.utils.data.DataLoader(ds, self.set_batch_size, shuffle=True)
            
            np.savez(os.path.join(model_path, "mean" + self.generate_file_name() +".npz"), dataset["mean"])
            
            # create optimizer
            print("Creating optimizer")
            criterion = nn.MSELoss()
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=0.9)
            
            # apply in gpu
            if self.USE_CUDA:
                net.cuda()
                criterion.cuda()
            
            # start train
            print("Training")
            for epoch in range(epoch_max):

                if(epoch%decrease_step==0 and epoch>0):
                    learning_rate *= drop_learning_rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate

                total_loss = 0
                count = 0

                t = tqdm(ds_loader, ncols=80)
                for data in t:

                    # set optimizer gradients to zero
                    optimizer.zero_grad()

                    # create variables
                    batch = Variable(data[0])
                    batch_target = Variable(data[1])
                    if(self.USE_CUDA):
                        batch = batch.cuda()
                        batch_target = batch_target.cuda()

                    # forward backward
                    output = net.forward(batch)
                    error = criterion(output, batch_target)
                    error.backward()
                    optimizer.step()

                    count += batch.size(0)
                    total_loss += error.item()

                    t.set_postfix(Bloss= error.item()/batch.size(0), loss= total_loss/count)

            # save the model
            torch.save(net.state_dict(), model_result_path)
            print("training finished")
    
    def est_normal(self, input_file_name):
        
        input_file_path = "sources/meshes"
        input_file = os.path.join(input_file_path, input_file_name + ".obj")
        if not os.path.exists(input_file):
            print("no file exists! Check the path!")
            
        # mesh load
        input_mesh = trimesh.load(input_file)
        input_mesh_norm = input_mesh.vertex_normals.copy()
        input_mesh_norm[input_mesh_norm[:, 2] < 0] = input_mesh_norm[input_mesh_norm[:, 2] < 0] * -1
        input_points = np.zeros([input_mesh.vertices.shape[0], 6])
        input_points[:, :3] = input_mesh.vertices.copy()
        # input_points[:,:3] = self.add_gaussian_noise(input_points[:,:3])
        input_points[:, 3:] = input_mesh_norm
        
        # store as .xyz file
        xyz_file_path = "sources/xyzfiles"
        if not os.path.exists(xyz_file_path):
            os.makedirs(xyz_file_path)
        xyz_file_name = input_file_name +".xyz"
        xyz_file = os.path.join(xyz_file_path, xyz_file_name)
        np.savetxt(xyz_file, input_points, delimiter=' ', fmt='%f')

        # create the estimator
        estimator = Estimator.NormalEstimatorHoughCNN()
        estimator.setDensitySensitive(self.density_sensitive)

        # load the file
        estimator.loadXYZ(xyz_file)

        # Ks, model, mean = self.load_model()
        Ks, model_path, modelCNN = self.model_init()
        if self.use_paper_model:
            model_result_name = "model.pth"
            model_result_path = os.path.join(model_path, model_result_name)
            model = modelCNN.load_model(model_result_path)
            mean = np.load(os.path.join(model_path, "mean.npz"))["arr_0"]
        else:
            model_result_name = "model" + self.generate_file_name() + ".pth"
            model_result_path = os.path.join(model_path, model_result_name)
            model = modelCNN.load_model(model_result_path)
            mean = np.load(os.path.join(model_path, "mean" + self.generate_file_name() +".npz"))["arr_0"]

        # set the neighborhood size
        estimator.setKs(Ks)

        # initialize
        estimator.initialize()

        # convert model to cuda if needed
        if self.USE_CUDA:
            model.cuda()
        model.eval()
        # iterate over the batches
        with torch.no_grad():
            for pt_id in tqdm(range(0,estimator.getPCSize(), self.set_batch_size)):
                bs = self.set_batch_size
                batch = estimator.getBatch(pt_id, bs) - mean[None,:,:,:]
                batch_th = torch.Tensor(batch)
                if self.USE_CUDA:
                    batch_th = batch_th.cuda()
                estimations = model.forward(batch_th)
                estimations = estimations.cpu().data.numpy()
                estimator.setBatch(pt_id,bs,estimations.astype(float))

        # save the estimator
        save_path = self.estimate_result_path()
        save_path = os.path.join(save_path,xyz_file_name)
        estimator.saveXYZ(save_path)
        
    def evaluate(self, file_name):
        # the origin data path
        origin_file_path = "sources/xyzfiles"
        origin_file = os.path.join(origin_file_path, file_name + ".xyz")
        if not os.path.exists(origin_file):
            print("No file exists! Check the path!")

        # the evaluate data path
        eva_file_path = self.estimate_result_path()
        eva_file = os.path.join(eva_file_path, file_name + ".xyz")
        if not os.path.exists(eva_file):
            print("No file exists! Check the path!")
            
        # load data
        origin_points = np.loadtxt(origin_file)
        eva_points = np.loadtxt(eva_file)

        # split norm point
        origin_norm = origin_points[:, 3:]
        eva_norm = eva_points[:, 3:]
        
        # compute angle
        cos_theta = np.sum(origin_norm * eva_norm, axis=1) / (np.linalg.norm(origin_norm, axis=1) * np.linalg.norm(eva_norm, axis=1))
        theta = np.arccos(cos_theta)
        # theta[theta>np.pi/2] = np.pi - theta[theta>np.pi/2]

        # compute RMS
        RMS = np.linalg.norm(np.rad2deg(theta))/np.sqrt(theta.shape[0])
        
        # compute prob
        angle = []
        prob = []
        for i in range(91):
            less_count = theta[theta < np.deg2rad(i)].shape[0]
            angle.append(i)
            prob.append(less_count/theta.shape[0])
        
        return RMS, angle, prob


