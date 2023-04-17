# Deep Learning for Robust Normal Estimation in Unstructured Point Clouds
# Copyright (c) 2016 Alexande Boulch and Renaud Marlet
#
# This program is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street,
# Fifth Floor, Boston, MA 02110-1301  USA
#
# PLEASE ACKNOWLEDGE THE ORIGINAL AUTHORS AND PUBLICATION:
# "Deep Learning for Robust Normal Estimation in Unstructured Point Clouds "
# by Alexandre Boulch and Renaud Marlet, Symposium of Geometry Processing 2016,
# Computer Graphics Forum

# from python.lib.python.NormalEstimatorHoughCNN import NormEst as Estimator
import python.NormalEstimatorHoughCNN as Estimator
import numpy as np
from tqdm import *
import torch
from torch.autograd import Variable
import os


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
    
    K = 100
    scale_number = 1
    batch_size = 512
    USE_CUDA = True
    input_filename = "test/cube_100k.xyz"
    if not os.path.exists(input_filename):
        print("no file exists! Check the path!")
    
    output_filename = "out.xyz"

    # create the estimator
    estimator = Estimator.NormalEstimatorHoughCNN()

    # load the file
    estimator.loadXYZ(input_filename)
    print(estimator.getPCSize())

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
            estimator.setBatch(pt_id,bs,estimations.astype(np.float64))

    # save the estimator
    save_path = save_test_result(scale_number,output_filename)
    estimator.saveXYZ(save_path)
