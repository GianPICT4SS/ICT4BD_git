"""This script makes a parametric analysis. Given the IDF and the relative EPW files, it must be specify the type of
parameters to change and calling the parametric_analysis method of the class Prediction. For each simulation a
energy signature is done, in order to see the effect on the save energy capacity of the building. The output of
the simulatiom are saved as several plots. """

import os
import sys
sys.path.insert(1, '../')
from method_building import Optimal_config as opt

path = '../files/idf/originals/'
epw = '../files/epw/ITA_Torino.160590_IWEC.epw'

for idf in os.listdir(path):
  Parametric = opt(idf, epw)
  




















"""

path_On = '/home/gianp/PycharmProjects/ict4bd/ict4bd_git/files/Office_On_corrected.idf'

path_Off = '/home/gianp/PycharmProjects/ict4bd/ict4bd_git/files/Office_Off_corrected.idf'

path_w = '/home/gianp/PycharmProjects/ict4bd/ict4bd_git/files/ITA_Torino.160590_IWEC.epw'

building_on = ef.get_building(path_On)
building_off = ef.get_building(path_Off)

dinamic_parameter = {'class_name': 'WindowMaterial:SimpleGlazingSystem',
                     'object_name': 'Simple 1001',
                    'field_name': 'UFactor'}


# ====================================
# Parameters and objectives selection
# ====================================

insulation_walls = FieldSelector(class_name='Material',
                           object_name='SuperInsulating_01445',
                           field_name='Thickness')

insulation_roof = FieldSelector(class_name='Material',
                           object_name='SuperInsulating_00795',
                           field_name='Thickness')

#lights = FieldSelector(class_name='Lights', object_name='*', field_name='Watts per Zone Floor Area')


windows = FieldSelector(class_name='WindowMaterial:SimpleGlazingSystem',
                               object_name='Simple 1001', field_name='UFactor')

parameters = [Parameter(selector=x) for x in (insulation_walls, insulation_roof, windows)]

objectives = ['DistrictHeating:Facility', 'DistrictCooling:Facility']


df_samples = pd.DataFrame({
    'Thickness_roof': np.linspace(0.01, 0.99, 60),
    'Thickness_walls': np.linspace(0.01, 0.99, 60),
    'U-VALUE_windows': np.linspace(-5, -0.3, 60)*-1
  })



problem = EPProblem(parameters, objectives)
#evaluator_on = EvaluatorEP(problem, building_on, epw_file=path_w, out_dir='out_dir', err_dir='err_dir')
#evaluator_off = EvaluatorEP(problem, building_off, epw_file=path_w, out_dir='out_dir', err_dir='err_dir')
#outputs_on = evaluator_on.df_apply(df_samples, keep_input=True)
#outputs_off = evaluator_off.df_apply(df_samples, keep_input=True)

#df_samples['Electricity:Facility-ON'] = outputs_on['Electricity:Facility']
#df_samples['DistrictHeating:Facility-ON'] = outputs_on['DistrictHeating:Facility']
#df_samples['DistrictCooling:Facility-ON'] = outputs_on['DistrictCooling:Facility']

#df_samples['Electricity:Facility-OFF'] = outputs_off['Electricity:Facility']
#df_samples['DistrictHeating:Facility-OFF'] = outputs_off['DistrictHeating:Facility']
#df_samples['DistrictCooling:Facility-OFF'] = outputs_off['DistrictCooling:Facility']



#heatmap(df_samples, df_samples.columns, name='corr_matr_60p')

#df_samples.to_csv('/home/gianp/PycharmProjects/ict4bd/ict4bd_git/files/outputs_60p_ch.csv')

parameters_ = expand_plist(
    {'SuperInsulating_01445':
        {'Thickness': (0.01, 0.99)},
     'SuperInsulating_00795':
         {'Thickness': (0.01, 0.99)},
     'Simple 1001':
        {'U-Factor': (0.3, 5)},

    }
)

"""
