# package used
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from normal_estimator import normal_Est


if __name__ == "__main__":
    nest = normal_Est()
    
    
    '''
    training model vs paper model 
    '''
    input_file_name = "dragon"
    
    ''' scale_number = 1 '''
    # paper model
    nest.est_normal(input_file_name, scale_number = 1, use_paper_model = True)
    angle_p, prob_p = nest.evaluate(input_file_name, scale_number = 1, use_paper_model = True)
    # training model
    nest.est_normal(input_file_name, scale_number = 1)
    angle_s1, prob_s1 = nest.evaluate(input_file_name, scale_number = 1)
    # plot the figure
    figure_path = "evaluate_figures_plot"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    plt.plot(angle_p, prob_p, label = "model in paper")
    plt.plot(angle_s1, prob_s1, label = "model reproduced")
    plt.legend()
    plt.xlabel('angle')
    plt.ylabel('prob')
    plt.title('Scale Number = 1')
    plt.savefig(os.path.join(figure_path, 'rm_vs_pm_sn1.pdf'))
    plt.show()
    ''' scale_number = 1 '''
    
    ''' scale_number = 3 '''
    # paper model
    nest.est_normal(input_file_name, scale_number = 3, use_paper_model = True)
    angle_p, prob_p = nest.evaluate(input_file_name, scale_number = 3, use_paper_model = True)
    # training model
    nest.est_normal(input_file_name, scale_number = 3)
    angle_s3, prob_s3 = nest.evaluate(input_file_name, scale_number = 3)
    # plot the figure
    figure_path = "evaluate_figures_plot"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    plt.plot(angle_p, prob_p, label = "model in paper")
    plt.plot(angle_s3, prob_s3, label = "model reproduced")
    plt.legend()
    plt.xlabel('angle')
    plt.ylabel('prob')
    plt.title('Scale Number = 3')
    plt.savefig(os.path.join(figure_path, 'rm_vs_pm_sn3.pdf'))
    plt.show()
    ''' scale_number = 3 '''
    
    ''' scale_number = 5 '''
    # paper model
    nest.est_normal(input_file_name, scale_number = 5, use_paper_model = True)
    angle_p, prob_p = nest.evaluate(input_file_name, scale_number = 5, use_paper_model = True)
    # training model
    nest.est_normal(input_file_name, scale_number = 5)
    angle_s5, prob_s5 = nest.evaluate(input_file_name, scale_number = 5)
    # plot the figure
    figure_path = "evaluate_figures_plot"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    plt.plot(angle_p, prob_p, label = "model in paper")
    plt.plot(angle_s5, prob_s5, label = "model reproduced")
    plt.legend()
    plt.xlabel('angle')
    plt.ylabel('prob')
    plt.title('Scale Number = 5')
    plt.savefig(os.path.join(figure_path, 'rm_vs_pm_sn5.pdf'))
    plt.show()
    ''' scale_number = 5 '''
    
    
    '''
    compare scale number
    '''
    # plot the figure
    figure_path = "evaluate_figures_plot"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    plt.plot(angle_s1, prob_s1, label = "scale_number = 1")
    plt.plot(angle_s3, prob_s3, label = "scale_number = 3")
    plt.plot(angle_s5, prob_s5, label = "scale_number = 5")
    plt.legend()
    plt.xlabel('angle')
    plt.ylabel('prob')
    plt.title('Scale Number')
    plt.savefig(os.path.join(figure_path, 'scale_number.pdf'))
    plt.show()
        
