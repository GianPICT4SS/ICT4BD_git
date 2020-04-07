import sys
sys.path.insert(1, '../')
from method_building import Prediction

learn = Prediction()
# =============================================
# Files
iddfile='/usr/local/EnergyPlus-9-0-1/Energy+.idd'
fname = '../../files/idf/Office_On_corrected.idf'
epw = '../../files/epw/ITA_Torino.160590_IWEC.epw'

learn.energy_signature()
