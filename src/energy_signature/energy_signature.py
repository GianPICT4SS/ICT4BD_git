import sys
sys.path.insert(1, '../')
from method_building import Prediction
from pathlib import Path

learn = Prediction()
# =============================================
# Files
iddfile='/usr/local/EnergyPlus-9-0-1/Energy+.idd'
fname = Path('../files/idf/optimal')
epw = Path('../files/epw/ITA_Torino.160590_IWEC.epw')

for idf in fname.iterdir():
    learn.energy_signature(iddfile=iddfile, idf_path=idf, epw_path=epw, name=idf.name)
