"""This script makes a parametric analysis. Given the IDF and the relative EPW files, it must be specify the type of
parameters to change and calling the parametric_analysis method of the class Prediction. For each simulation a
energy signature is done, in order to see the effect on the save energy capacity of the building. The output of
the simulatiom are saved as several plots. """
import sys
import os
sys.path.insert(1, '../')
from method_building import Optimal_config as opt

path = '/home/ict4db/Scrivania/ICT4BD_git-master/files/idf/originals/'
epw = '/home/ict4db/Scrivania/ICT4BD_git-master/files/epw/ITA_Torino.160590_IWEC.epw'


#new_path = '/home/ict4db/Scrivania/ICT4BD_git-master/files/idf/originals/Office_Off_corrected.idf'

for idf in os.listdir(path):

	new_path = path + idf
	Parametric = opt(new_path,epw) #will automatically run all the method in class Optimal config





















