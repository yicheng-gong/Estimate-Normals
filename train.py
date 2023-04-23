# package used
import normal_estimator_cpp.NormalEstimatorHoughCNN as Estimator
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import pickle
import os

def dataset_create(dataset_path, dataset_name, dataset_size, scale_number, batch_size, estimator):
    print("creating dataset")
    count = 0
    dataset = np.zeros((dataset_size, scale_number, 33,33))
    targets = np.zeros((dataset_size, 2))
    for i in tqdm(range(0,dataset_size, batch_size), ncols=80):
        nbr, batch, batch_targets = estimator.generateTrainAccRandomCorner(batch_size)
        if count+nbr > dataset_size:
            nbr = dataset_size - count
        dataset[count:count+nbr] = batch[0:nbr]
        targets[count:count+nbr] = batch_targets[0:nbr]
        count += nbr
        if(count >= dataset_size):
            break
        
    # save the dataset
    mean = dataset.mean(axis=0)
    print(mean.shape)
    dataset = {"input":dataset, "targets":targets, "mean":mean}
    print("  saving")
    pickle.dump(dataset, open(dataset_path, "wb" ))
    print("-->done")
    

if __name__ == "__main__":
    
    # faster computation times
    torch.backends.cudnn.benchmark = True

    # training model parameter
    K = 128
    scale_number = 5
    batch_size = 256

    # choose device
    if torch.cuda.is_available():
        USE_CUDA = True
    else:
        USE_CUDA = False
    
    # create the estimator
    estimator = Estimator.NormalEstimatorHoughCNN()
    # Ks computing and model selecting
    if scale_number == 1:
        Ks=np.array([K], dtype=int)
        import cnn_models.model_1s as model_def
        model_result_path = "model_1s"
    elif scale_number == 3:
        Ks=np.array([K,K/2,K*2], dtype=int)
        import cnn_models.model_3s as model_def
        model_result_path = "model_3s"
    elif scale_number == 5:
        Ks=np.array([K,K/4,K/2,K*2,K*4], dtype=int)
        import cnn_models.model_5s as model_def
        model_result_path = "model_5s"
    estimator.setKs(Ks)
    net = model_def.create_model()

    # configure dataset
    dataset_size = 100000
    dataset_path = "dataset"
    dataset_name = "dataset_sn" + str(scale_number) + "_bs" + str(batch_size) + "_ds" + str(dataset_size) + ".p"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    dataset_path = os.path.join(dataset_path, dataset_name)
    if not os.path.exists(dataset_path):
        dataset_create(dataset_path, dataset_name, dataset_size, scale_number, batch_size, estimator)
        
    # dataset loading
    print("loading the model")
    dataset = pickle.load(open(dataset_path, "rb" ))
    dataset["input"] -= dataset["mean"][None,:,:,:]
    input_data = torch.from_numpy(dataset["input"]).float()
    target_data = torch.from_numpy(dataset["targets"]).float()
    ds = torch.utils.data.TensorDataset(input_data, target_data)
    ds_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    # training parameter
    drop_learning_rate = 0.5
    learning_rate = 0.1
    epoch_max = 40
    decrease_step = 4
    
    # create optimizer
    print("Creating optimizer")
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=0.9)
    
    # apply in gpu
    if USE_CUDA:
        net.cuda()
        criterion.cuda()
        
    # training result store
    result_path = "models_reproduced"
    result_path = os.path.join(result_path, model_result_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    np.savez(os.path.join(result_path, "mean"), dataset["mean"])
    f = open(os.path.join(result_path, "training_logs.txt"), "w")
    
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
            if(USE_CUDA):
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

        f.write(str(epoch)+" ")
        f.write(str(learning_rate)+" ")
        f.write(str(total_loss))
        f.write("\n")
        f.flush()

        # save the model
        torch.save(net.state_dict(), os.path.join(result_path, "model.pth"))

    f.close()
    
    print("training finished")
