from src.method_building import energy_signature
# =============================================
# Files
iddfile='/usr/local/EnergyPlus-9-0-1/Energy+.idd'
fname = '../../files/idf/Office_On_corrected.idf'
epw = '../../files/epw/ITA_Torino.160590_IWEC.epw'

ls = energy_signature(iddfile=iddfile, fname=fname, epw=epw)
