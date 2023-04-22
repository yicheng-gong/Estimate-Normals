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
    
    '''
    varying noise scale = 0, 20%, 40%, 60%, 80%, 100%, 120%, 140%, 160%, 180%, 200%. 
    '''
    
    # noise scale set
    NOISE = np.linspace(0,2,21)
    # root mean square error set
    RMS_p = []
    RMS_r = []
    # prob set of angle deviation less than 5 and 10 degree
    PL5_p = []
    PL10_p = []
    PL5_r = []
    PL10_r = []
    for n in NOISE:
        # paper model
        nest.est_normal(input_file_name, noise_scale = n, scale_number = 1, use_paper_model = True)
        rms_p,angle_p,prob_p = nest.evaluate(input_file_name, noise_scale = n, scale_number = 1, use_paper_model = True)
        RMS_p.append(rms_p)
        PL5_p.append(prob_p[6])
        PL10_p.append(prob_p[11])
        # training model
        nest.est_normal(input_file_name, noise_scale = n, scale_number = 1)
        rms_r,angle_r,prob_r = nest.evaluate(input_file_name, noise_scale = n, scale_number = 1)
        RMS_r.append(rms_r)
        PL5_r.append(prob_r[6])
        PL10_r.append(prob_r[11])
    
    # figure save path
    figure_path = "evaluate_figures_plot"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    # plot figure 1
    plt.plot(NOISE, RMS_p, label = "model in paper")
    plt.plot(NOISE, RMS_r, label = "model reproduced")
    plt.legend()
    plt.ylabel('RMS (deg)')
    plt.xlabel('Noise Scale (%)')
    plt.title('RMS')
    plt.savefig(os.path.join(figure_path, 'rnsv.pdf'))
    plt.show()
    
    # plot figure 2
    plt.plot(NOISE, PL5_p, label = "model in paper")
    plt.plot(NOISE, PL5_r, label = "model reproduced")
    plt.legend()
    plt.ylabel('Prob of angle deviation < 5 degree (%)')
    plt.xlabel('Noise Scale (%)')
    plt.title('5 degree deviation')
    plt.ylim(0,1)
    plt.savefig(os.path.join(figure_path, '5dd.pdf'))
    plt.show()
    
    # plot figure 3
    plt.plot(NOISE, PL10_p, label = "model in paper")
    plt.plot(NOISE, PL10_r, label = "model reproduced")
    plt.legend()
    plt.ylabel('Prob of angle deviation < 10 degree (%)')
    plt.xlabel('Noise Scale (%)')
    plt.ylim(0,1)
    plt.title('10 degree deviation')
    plt.savefig(os.path.join(figure_path, '10dd.pdf'))
    plt.show()
    
    
    
    # ''' scale_number = 1 '''
    # # paper model
    # nest.est_normal(input_file_name, scale_number = 1, use_paper_model = True)
    # angle_p, prob_p = nest.evaluate(input_file_name, scale_number = 1, use_paper_model = True)
    # # training model
    # nest.est_normal(input_file_name, scale_number = 1)
    # angle_s1, prob_s1 = nest.evaluate(input_file_name, scale_number = 1)
    # # plot the figure
    # figure_path = "evaluate_figures_plot"
    # if not os.path.exists(figure_path):
    #     os.makedirs(figure_path)
    # plt.plot(angle_p, prob_p, label = "model in paper")
    # plt.plot(angle_s1, prob_s1, label = "model reproduced")
    # plt.legend()
    # plt.xlabel('angle')
    # plt.ylabel('prob')
    # plt.title('Scale Number = 1')
    # plt.savefig(os.path.join(figure_path, 'rm_vs_pm_sn1.pdf'))
    # plt.show()
    # ''' scale_number = 1 '''
    
    # ''' scale_number = 3 '''
    # # paper model
    # nest.est_normal(input_file_name, scale_number = 3, use_paper_model = True)
    # angle_p, prob_p = nest.evaluate(input_file_name, scale_number = 3, use_paper_model = True)
    # # training model
    # nest.est_normal(input_file_name, scale_number = 3)
    # angle_s3, prob_s3 = nest.evaluate(input_file_name, scale_number = 3)
    # # plot the figure
    # figure_path = "evaluate_figures_plot"
    # if not os.path.exists(figure_path):
    #     os.makedirs(figure_path)
    # plt.plot(angle_p, prob_p, label = "model in paper")
    # plt.plot(angle_s3, prob_s3, label = "model reproduced")
    # plt.legend()
    # plt.xlabel('angle')
    # plt.ylabel('prob')
    # plt.title('Scale Number = 3')
    # plt.savefig(os.path.join(figure_path, 'rm_vs_pm_sn3.pdf'))
    # plt.show()
    # ''' scale_number = 3 '''
    
    # ''' scale_number = 5 '''
    # # paper model
    # nest.est_normal(input_file_name, scale_number = 5, use_paper_model = True)
    # angle_p, prob_p = nest.evaluate(input_file_name, scale_number = 5, use_paper_model = True)
    # # training model
    # nest.est_normal(input_file_name, scale_number = 5)
    # angle_s5, prob_s5 = nest.evaluate(input_file_name, scale_number = 5)
    # # plot the figure
    # figure_path = "evaluate_figures_plot"
    # if not os.path.exists(figure_path):
    #     os.makedirs(figure_path)
    # plt.plot(angle_p, prob_p, label = "model in paper")
    # plt.plot(angle_s5, prob_s5, label = "model reproduced")
    # plt.legend()
    # plt.xlabel('angle')
    # plt.ylabel('prob')
    # plt.title('Scale Number = 5')
    # plt.savefig(os.path.join(figure_path, 'rm_vs_pm_sn5.pdf'))
    # plt.show()
    # ''' scale_number = 5 '''
    
    
    # '''
    # compare scale number
    # '''
    # # plot the figure
    # figure_path = "evaluate_figures_plot"
    # if not os.path.exists(figure_path):
    #     os.makedirs(figure_path)
    # plt.plot(angle_s1, prob_s1, label = "scale_number = 1")
    # plt.plot(angle_s3, prob_s3, label = "scale_number = 3")
    # plt.plot(angle_s5, prob_s5, label = "scale_number = 5")
    # plt.legend()
    # plt.xlabel('angle')
    # plt.ylabel('prob')
    # plt.title('Scale Number')
    # plt.savefig(os.path.join(figure_path, 'scale_number.pdf'))
    # plt.show()
        
