# package used
import python.NormalEstimatorHoughCNN as Estimator
import numpy as np
from tqdm import *
import torch
from torch.autograd import Variable
import os
import trimesh
import scipy


def load_model(K,scale_number):
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

def save_test_result(scale_number,output_filename):
    if scale_number == 1:
        test_result_path = "test_result/model_1s"
    elif scale_number == 3:
        test_result_path = "test_result/model_3s"
    elif scale_number == 5:
        test_result_path = "test_result/model_5s"
    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)
    save_path = os.path.join(test_result_path,output_filename)
    
    return save_path
        
    
if __name__ == "__main__":
    
    K = 16
    scale_number = 1
    batch_size = 256
    USE_CUDA = True
    input_file_path = "sources/meshes"
    input_file_name = "dragon"
    input_file = os.path.join(input_file_path, input_file_name + ".obj")
    if not os.path.exists(input_file):
        print("no file exists! Check the path!")
        
    # mesh load
    input_mesh = trimesh.load(input_file)
    input_mesh_norm = input_mesh.vertex_normals
    input_mesh_points = input_mesh.sample(0)
    input_points = np.zeros([input_mesh.vertices.shape[0] + input_mesh_points.shape[0], 6])
    input_points[:,:3] = np.r_[input_mesh.vertices, input_mesh_points]
    input_points[:input_mesh.vertices.shape[0], 3:] = input_mesh_norm
    
    # store as .xyz file
    xyz_file_path = "sources/xyzfiles"
    xyz_file_name = input_file_name + ".xyz"
    xyz_file = os.path.join(xyz_file_path, xyz_file_name)
    np.savetxt(xyz_file, input_points, delimiter=' ', fmt='%f')

    # create the estimator
    estimator = Estimator.NormalEstimatorHoughCNN()

    # load the file
    estimator.loadXYZ(xyz_file)

    Ks, model, mean = load_model(K,scale_number)

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
    save_path = save_test_result(scale_number, xyz_file_name)
    estimator.saveXYZ(save_path)

