# package used
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # the origin data path
    origin_file_path = "sources/xyzfiles"
    file_name = "dragon.xyz"
    origin_file = os.path.join(origin_file_path, file_name)
    if not os.path.exists(origin_file):
        print("No file exists! Check the path!")

    # the evaluate data path
    eva_file_path = "test_result"
    model_path = "model_1s"
    eva_file = os.path.join(eva_file_path, model_path, file_name)
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
    
    #plot the figure
    plt.plot(angle, prob)
    # plt.legend()
    plt.xlabel('angle')
    plt.ylabel('prob')
    # plt.title('Noise Effect')
    # plt.savefig('Q7_Results/fitness_score.pdf')
    plt.show()
        
