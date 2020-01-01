import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pathf_0= '../../files/outputs/outputs_60p_ch_twr0.csv'

df_0 = pd.read_csv('../../files/outputs/outputs_60p_ch_twr0.csv')
df_1 = pd.read_csv('../../files/outputs/outputs_60p_ch_twr1.csv')
df_2 = pd.read_csv('../../files/outputs/outputs_60p_ch_twr2.csv')
df_3 = pd.read_csv('../../files/outputs/outputs_60p_ch_twr3.csv')
df_4 = pd.read_csv('../../files/outputs/outputs_60p_ch_twr4.csv')
df_5 = pd.read_csv('../../files/outputs/outputs_60p_ch_twr5.csv')

#parameters
# 0.4 m,
thickness_walls_0 = df_0['SuperInsulating_00795']
# Objectives
district_Heating_ON_0 = df_0['DistrictHeating:Facility']
district_Cooling_ON_0 = df_0['DistrictCooling:Facility']
# 0.6 m

thickness_walls_1 = df_1['SuperInsulating_00795']
# Objectives
district_Heating_ON_1 = df_1['DistrictHeating:Facility']
district_Cooling_ON_1 = df_1['DistrictCooling:Facility']
# 0.8
thickness_walls_2 = df_2['SuperInsulating_00795']
# Objectives
district_Heating_ON_2 = df_2['DistrictHeating:Facility']
district_Cooling_ON_2 = df_2['DistrictCooling:Facility']

# 1.0
thickness_walls_3 = df_3['SuperInsulating_00795']
# Objectives
district_Heating_ON_3 = df_3['DistrictHeating:Facility']
district_Cooling_ON_3 = df_3['DistrictCooling:Facility']
# 1.2
thickness_walls_4 = df_4['SuperInsulating_00795']
# Objectives
district_Heating_ON_4 = df_4['DistrictHeating:Facility']
district_Cooling_ON_4 = df_4['DistrictCooling:Facility']
# 1.4
thickness_walls_5 = df_5['SuperInsulating_00795']
# Objectives
district_Heating_ON_5 = df_5['DistrictHeating:Facility']
district_Cooling_ON_5 = df_5['DistrictCooling:Facility']




# Plots
# District cooling
fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))
fig.suptitle('Parametric Analysis (Thickness walls)')
ax1.plot(thickness_walls_0, district_Cooling_ON_0, label='[0.1 m / 0.4 W/m^2K]', color='brown')
ax1.plot(thickness_walls_1, district_Cooling_ON_1, label='[0.2 m / 0.9 W/m^2K]')
ax1.plot(thickness_walls_2, district_Cooling_ON_2, label='[0.3 m / 1.6 W/m^2K]')
ax1.plot(thickness_walls_3, district_Cooling_ON_3, label='[0.4 m / 2.5 W/m^2K]')
ax1.plot(thickness_walls_4, district_Cooling_ON_4, label='[0.5 m / 3.6 W/m^2K]')
ax1.plot(thickness_walls_5, district_Cooling_ON_5, label='[0.6 m / 4.9 W/m^2K]', color='blue')
ax1.set_xlabel('Thickness [m]')
ax1.set_ylabel('districtCooling [MWh]')
#ax1.legend(loc='upper right', bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))
ax1.grid(linestyle='--', linewidth=.4, which='both')


ax2.plot(thickness_walls_0, district_Heating_ON_0, label='[0.1 m / 0.4 W/m^2K]', color='brown')
ax2.plot(thickness_walls_1, district_Heating_ON_1, label='[0.2 m / 0.9 W/m^2K]')
ax2.plot(thickness_walls_2, district_Heating_ON_2, label='[0.3 m / 1.6 W/m^2K]')
ax2.plot(thickness_walls_3, district_Heating_ON_3, label='[0.4 m / 2.5 W/m^2K]')
ax2.plot(thickness_walls_4, district_Heating_ON_4, label='[0.5 m / 3.6 W/m^2K]')
ax2.plot(thickness_walls_5, district_Heating_ON_5, label='[0.6 m / 4.9 W/m^2K]', color='blue')
ax2.set_xlabel('Thickness [m]')
ax2.set_ylabel('districtHeating [MWh]')
ax2.legend(loc='upper left', bbox_to_anchor=(.8, -.5, 0.4, 0.4))
ax2.grid(linestyle='--', linewidth=.4, which='both')
plt.subplots_adjust(bottom=0.3, right=0.8, top=0.9, hspace=1)
plt.savefig(fname='../../plots/parametric_analysis_tw_on.png', dpi=500)
plt.close()
