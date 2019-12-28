from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm
from method_building import energy_signature

# =============================================
# Files
iddfile='/usr/local/EnergyPlus-9-0-1/Energy+.idd'
fname = '../files/idf/Office_On_corrected.idf'
epw = '../files/epw/ITA_Torino.160590_IWEC.epw'

df = energy_signature(iddfile=iddfile, fname=fname, epw=epw)
#df = df.set_index('date')
#df = df.set_index(pd.to_datetime(df.index))
df = df.resample('D').mean()
df['t_ext'] = df['Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)']
df['p_z1'] = df['BLOCCO1:ZONA1:Zone Ventilation Sensible Heat Loss Energy [J](Hourly)']
model = sm.OLS(df['p_z1'], sm.add_constant(df['t_ext']))
results = model.fit()

fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 7))
fig.suptitle("Energy Signature")
ax1.plot(df['t_ext'], results.predict(), 'r')
ax1.scatter(df['t_ext'], df['p_z1'])
ax1.set_xlabel('Temperature [C]')
ax1.set_ylabel('Heat Loss Energy [J]')
ax1.set_title('DAY resample')
ax1.grid(linestyle='--', linewidth=.4, which='both')

df = df.resample('W').mean()
df['t_ext'] = df['Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)']
df['p_z1'] = df['BLOCCO1:ZONA1:Zone Ventilation Sensible Heat Loss Energy [J](Hourly)']
model = sm.OLS(df['p_z1'], sm.add_constant(df['t_ext']))
results = model.fit()

ax2.plot(df['t_ext'], results.predict(), 'r')
ax2.scatter(df['t_ext'], df['p_z1'])
ax2.set_xlabel('Temperature [C]')
ax2.set_ylabel('Heat Loss Energy [J]')
ax2.set_title('WEEK resample')
ax2.grid(linestyle='--', linewidth=.4, which='both')
plt.subplots_adjust(bottom=0.4, right=0.8, top=0.9, hspace=1)
plt.savefig(fname='../plots/energy_signature_ex.png', dpi=400)
plt.close()