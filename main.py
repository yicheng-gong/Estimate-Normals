# package used
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from normal_estimator import normal_Est


if __name__ == "__main__":
    nest = normal_Est()
    input_file_name = "dragon"
    
    '''
    compare scale number
    '''
    # scale_number = 1
    nest.est_normal(input_file_name, scale_number = 1)
    angle_s1, prob_s1 = nest.evaluate(input_file_name, scale_number = 1)
    
    # scale_number = 3
    nest.est_normal(input_file_name, scale_number = 3)
    angle_s3, prob_s3 = nest.evaluate(input_file_name, scale_number = 3)
    
    # scale_number = 5
    nest.est_normal(input_file_name, scale_number = 5)
    angle_s5, prob_s5 = nest.evaluate(input_file_name, scale_number = 5)
    
    #plot the figure
    plt.plot(angle_s1, prob_s1, label = "scale_number = 1")
    plt.plot(angle_s3, prob_s3, label = "scale_number = 3")
    plt.plot(angle_s5, prob_s5, label = "scale_number = 5")
    plt.legend()
    plt.xlabel('angle')
    plt.ylabel('prob')
    plt.title('Scale Number')
    # plt.savefig('Q7_Results/fitness_score.pdf')
    plt.show()
        
