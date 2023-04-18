# package used
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from test_and_eval import test_and_eval


if __name__ == "__main__":
    te = test_and_eval()
    input_file_name = "dragon"
    te.est_normal(input_file_name)
    te.evaluate(input_file_name)
    
        
