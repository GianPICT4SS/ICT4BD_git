from time import time
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 15]
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.inspection import plot_partial_dependence
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import statsmodels.api as sm

import eppy
from eppy.modeleditor import IDF
from geomeppy import IDF as geom_IDF
from besos import eppy_funcs as ef
from besos import sampling
from besos.evaluator import EvaluatorEP
from besos.parameters import RangeParameter, FieldSelector, FilterSelector, Parameter, expand_plist, wwr, CategoryParameter, GenericSelector
from besos.problem import EPProblem

import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s : %(message)s',
                    datefmt='%d/%m/%Y %H:%M ',
                    level=logging.INFO)

logger = logging.getLogger(__name__)



class Prediction():


    @classmethod
    def create_time_steps(cls, length):
        return list(range(-length, 0))

    @classmethod
    def show_plot(cls, plot_data, delta, title):
        labels = ['History', 'True Future', 'Model Prediction']
        marker = ['.-', 'rx', 'go']
        time_steps = cls.create_time_steps(plot_data[0].shape[0])
        if delta:
            future = delta
        else:
            future = 0

        plt.title(title)
        for i, x in enumerate(plot_data):
            if i:
                plt.plot(future, plot_data[i], marker[i], markersize=10,
                         label=labels[i])
            else:
                plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
        plt.legend()
        plt.xlim([time_steps[0], (future + 5) * 2])
        plt.xlabel('Time-Step')
        return plt

    @classmethod
    def multivariate_data(cls, dataset, target, start_index, end_index, history_size,
                          target_size, step, single_step=False):
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i - history_size, i, step)
            data.append(dataset[indices])
            if single_step:
                labels.append(target[i + target_size])
            else:
                labels.append(target[i:i + target_size])

        return np.array(data), np.array(labels)

    @classmethod
    def multi_step_plot(cls, history, true_future, prediction, model='LSTM'):
        plt.figure(figsize=(12, 6))
        num_in = cls.create_time_steps(len(history))
        num_out = len(true_future)

        plt.plot(num_in, np.array(history[:, 1]), label='History')
        plt.plot(np.arange(num_out), np.array(true_future), 'bo',
                 label='True Future')
        if prediction.any():
            plt.plot(np.arange(num_out), np.array(prediction), 'ro',
                     label='Predicted Future')
        plt.legend(loc='upper left')
        plt.xlabel('Timestemp')
        plt.ylabel('Temperature [C°]')
        plt.title(f'Mean Indoor Temperature Prediction {model}')

        plt.show()

    @classmethod
    def plot_train_history(cls, history, title):

        if 'accuracy' in history.history.keys():
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs = range(1, len(loss) + 1)

            fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 8))
            ax1.plot(epochs, acc, 'bo', label='Training acc')
            ax1.plot(epochs, val_acc, 'b', label='Validation acc')
            ax1.set_title('Training and validation accuracy')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(linestyle='--', linewidth=.4, which='both')

            ax2.plot(epochs, loss, 'bo', label='Training loss')
            ax2.plot(epochs, val_loss, 'b', label='Validation loss')
            ax2.set_title(title)
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(linestyle='--', linewidth=.4, which='both')
            plt.subplots_adjust(bottom=0.4, right=0.8, top=0.9, hspace=1)
            plt.savefig(fname=f'../../plots/prediction_model_error_{title}.png', dpi=400)
            plt.close()
        else:
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs = range(1, len(loss) + 1)
            fig, ax2 = plt.subplots(1, figsize=(8, 8))
            ax2.plot(epochs, loss, 'b.-', label='Training loss')
            ax2.plot(epochs, val_loss, 'r.-', label='Validation loss')
            ax2.set_title(title)
            plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            plt.axis([1, 20, 0, 0.05])
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Loss')
            ax2.legend(fontsize=14)
            ax2.grid(linestyle='--', linewidth=.4, which='both')
            plt.subplots_adjust(bottom=0.4, right=0.8, top=0.9, hspace=1)
            plt.savefig(fname=f'../../plots/prediction_model_error_{title}.png', dpi=400)
            plt.close()

    @classmethod
    def generator(cls, data, lookback, delay, min_index, max_index,
                  shuffle=False, batch_size=128, step=1):
        if max_index is None:
            max_index = len(data) - delay - 1
        i = min_index + lookback
        while 1:
            if shuffle:
                rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
            else:
                if i + batch_size >= max_index:
                    i = min_index + lookback
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)
            samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
            targets = np.zeros((len(rows),))
            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                samples[j] = data[indices]
                targets[j] = data[rows[j] + delay][1]
            yield samples, targets

        # Neural Network Multi-layer Perceptron regressor

    @classmethod
    def nnMR(cls, samples, output, test_size=0.2, random_state=123):

        train_in, test_in, train_out, test_out = train_test_split(samples, output, test_size=test_size,
                                                                      random_state=random_state)

        train_in_scaled = preprocessing.scale(train_in)
        train_out_scaled = preprocessing.scale(train_out)

        print("Training MLPRegressor...")
        tic = time()
        reg = MLPRegressor(solver='lbfgs', max_iter=350)  # lbfgs well suited for small datasets
        reg.fit(train_in_scaled, train_out_scaled)
        print("done in {:.3f}s".format(time() - tic))

        results = test_in.copy()
        results[output.name] = test_out
        results['Predicted'] = reg.predict(test_in)

        # Plots
        # The mean squared error
        print('===========================================================================')
        print(f"Summary Neural Network Multi-layer Perceptron regressor for {output.name}")
        print('===========================================================================')
        print(f"Mean Squared Error {output.name}: {mean_squared_error(results[output.name], results['Predicted'])}")
        print("===================================================================================================")
        # The coefficient of determination: 1 is perfect prediction
        print("R Square")
        print('Coefficient of determination: %.2f'
                  % r2_score(results[output.name], results['Predicted']))

        # Partial Dependence plots
        print('Computing partial dependence plots...')
        tic = time()
        # We don't compute the 2-way PDP (5, 1) here, because it is a lot slower
        # with the brute method.
        features = samples.columns
        plot_partial_dependence(reg, train_in, features,
                                    n_jobs=3, grid_resolution=20)
        print("done in {:.3f}s".format(time() - tic))
        fig = plt.gcf()
        fig.suptitle(f"Partial dependence of {output.name}, with MLPRegressor")
        fig.subplots_adjust(hspace=0.3)

        return results








    @classmethod
    def heatmap(cls, X, columns, name):
        if not hasattr(X, 'columns'):
            df = pd.DataFrame(X, columns=columns)
            corr = df.corr()
        else:
            corr = X.corr()

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        fig, ax = plt.subplots()

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        plt.title(name)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, square=True,
                linewidths=.4, cbar_kws={"shrink": .4}, annot=True)

        fname = name + '.png'
        plt.savefig('../../plots/' + fname)




    @classmethod
    def parametric_analysis(cls, dinamic_parameter, fixed_parameters, objectives, idf_path, epw_path, n_points=60):
        """
        this method allows to do a parametric analysis by varying a parameter while maintaining fixed others.

        parameter: dinamic_parameter: a dictionary containing information about the parameter that must be changed.
        {class_name: Material, object_name: Simply_1001, field_name: UFactor}
        fixed_parameters: a list of dictionary containing information about the parameters that must be fixed.
        The dictionary has the same structure of the dinamic_parameter dictionary.
        objectives: a list with the objectives,
        idf_path: a str with the path for idf file,
        epw_path: str path for epw file.
        n_points: the number of the sample points to be used in the simulation
        """


        building = ef.get_building(idf_path)
        #building_off = ef.get_building(path_Off)

        # ====================================
        # Parameters and objectives selection
        # ====================================

        ls_fp = []
        for a in fixed_parameters:
            ls_fp.append(FieldSelector(class_name=a['class_name'],
                                   object_name=a['object_name'],
                                   field_name=a['field_name']))


        d_par = FieldSelector(class_name=dinamic_parameter['class_name'],
                            object_name=dinamic_parameter['object_name'],
                            field_name=dinamic_parameter['field_name'])

        ls_fp.append(d_par)
        wwr_list = [0.15, 0.5, 0.9, 0.15, 0.5, 0.9]
        wwr_range = CategoryParameter(options=wwr_list)
        w_t_r = wwr(wwr_range)
        ls_fp.append(w_t_r)

        parameters = [Parameter(selector=x) for x in ls_fp]

        if dinamic_parameter['object_name'] == 'Simple 1001':
            logger.info(f"Parametric analysis for {dinamic_parameter['object_name']} will start.") # U-factor as dy par
            range_ufw = np.linspace(-5, -0.3, n_points)
            dict_ = {}
            for j in range(6):
                for i in range(len(fixed_parameters)):
                    #dict_[parameters[i].selector.object_name] = [((i+2)**2*0.1)]*n_points
                    dict_[parameters[i].selector.object_name] = [(j+1)*0.1]*n_points
                dict_[parameters[-1].selector.name] = wwr_list[j]*n_points  # add wwr_list associated with wwr par
                dict_[parameters[len(fixed_parameters)].selector.object_name] = range_ufw*-1
                df_samples = pd.DataFrame.from_dict(dict_)
                #df_samples[parameters[len(fixed_parameters)].selector.object_name] = range_ufw*-1  #take dy-par Ufactor
                problem = EPProblem(parameters, objectives)
                evaluator = EvaluatorEP(problem, building, epw_file=epw_path, out_dir='../../files/out_dir', err_dir='../../files/err_dir')
                outputs = evaluator.df_apply(df_samples, keep_input=True)
                logger.info(f"Simulation number {j} done.")
                iddfile = '/usr/local/EnergyPlus-9-0-1/Energy+.idd'
                fname = '../../files/out_dir/BESOS_Output/in.idf'
                epw = '../../files/epw/ITA_Torino.160590_IWEC.epw'
                cls.energy_signature(iddfile=iddfile, fname=fname, epw=epw, name=j)
                print('###############################')
                logger.info(f'Energy signature {j} done.')
                print('################################')
                df_samples['DistrictHeating:Facility'] = outputs['DistrictHeating:Facility']
                df_samples['DistrictCooling:Facility'] = outputs['DistrictCooling:Facility']
                df_samples['Electricity:Facility'] = outputs['Electricity:Facility']
                df_samples.to_csv('../../files/outputs/outputs_60p_ch_ufw' + str(j) + '.csv')
        elif dinamic_parameter['class_name'] == 'Material':
            logger.info(f"Parametric analysis for {dinamic_parameter['object_name']} will start.")
            range_t = np.linspace(0.01, 0.5, n_points)  # metri
            dict_ = {}
            for j in range(6):
                for i in range(len(fixed_parameters)):
                    if parameters[i].selector.object_name == 'Simple 1001':
                        dict_[parameters[i].selector.object_name] = [((j + 2) ** 2 * 0.1)] * n_points
                    else:
                        dict_[parameters[i].selector.object_name] = [(j + 1) * 0.1] * n_points
                dict_[parameters[-1].selector.name] = wwr_list[j]*n_points
                dict_[parameters[len(fixed_parameters)].selector.object_name] = range_t
                df_samples = pd.DataFrame.from_dict(dict_)
                problem = EPProblem(parameters, objectives)
                evaluator = EvaluatorEP(problem, building, epw=epw_path, out_dir='../../files/out_dir', err_dir='../../files/err_dir')
                outputs = evaluator.df_apply(df_samples, keep_input=True)
                logger.info(f"Simulation number {j} done.")
                iddfile = '/usr/local/EnergyPlus-9-0-1/Energy+.idd'
                fname = '../../files/out_dir/BESOS_Output/in.idf'
                epw = '../../files/epw/ITA_Torino.160590_IWEC.epw'
                cls.energy_signature(iddfile=iddfile, fname=fname, epw=epw, name=j)
                print('###############################')
                logger.info(f'Energy signature {j} done.')
                print('################################')
                df_samples['DistrictHeating:Facility'] = outputs['DistrictHeating:Facility']
                df_samples['DistrictCooling:Facility'] = outputs['DistrictCooling:Facility']
                df_samples['Electricity:Facility'] = outputs['Electricity:Facility']
                df_samples.to_csv(
                '../../files/outputs/outputs_60p_ch_twr' + str(j) + '.csv')

    @classmethod
    def create_csv(cls, df):

        cols = [
               'Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)',
               'Environment:Site Outdoor Air Barometric Pressure [Pa](Hourly)',
               'Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](Hourly)',
               'Environment:Site Solar Azimuth Angle [deg](Hourly)',
               'Environment:Site Solar Altitude Angle [deg](Hourly)',
               'Environment:Site Wind Speed [m/s](Hourly)',
               'Environment:Site Direct Solar Radiation Rate per Area [W/m2](Hourly)',
               'DistrictCooling:Facility [J](Hourly)',
               'DistrictHeating:Facility [J](Hourly)',
               'BLOCCO1:ZONA2:Zone Operative Temperature [C](Hourly)',
               'BLOCCO1:ZONA3:Zone Operative Temperature [C](Hourly)',
               'BLOCCO1:ZONA1:Zone Operative Temperature [C](Hourly)',
               'BLOCCO1:ZONA2:Zone Infiltration Sensible Heat Loss Energy [J](Hourly)',
               'BLOCCO1:ZONA2:Zone Infiltration Sensible Heat Gain Energy [J](Hourly)',
               'BLOCCO1:ZONA2:Zone Infiltration Air Change Rate [ach](Hourly)',
               'BLOCCO1:ZONA3:Zone Infiltration Sensible Heat Loss Energy [J](Hourly)',
               'BLOCCO1:ZONA3:Zone Infiltration Sensible Heat Gain Energy [J](Hourly)',
               'BLOCCO1:ZONA3:Zone Infiltration Air Change Rate [ach](Hourly)',
               'BLOCCO1:ZONA1:Zone Infiltration Sensible Heat Loss Energy [J](Hourly)',
               'BLOCCO1:ZONA1:Zone Infiltration Sensible Heat Gain Energy [J](Hourly)',
               'BLOCCO1:ZONA1:Zone Infiltration Air Change Rate [ach](Hourly)',
               'BLOCCO1:ZONA2:Zone Ventilation Sensible Heat Loss Energy [J](Hourly)',
               'BLOCCO1:ZONA2:Zone Ventilation Sensible Heat Gain Energy [J](Hourly)',
               'BLOCCO1:ZONA2:Zone Ventilation Air Change Rate [ach](Hourly)',
               'BLOCCO1:ZONA3:Zone Ventilation Sensible Heat Loss Energy [J](Hourly)',
               'BLOCCO1:ZONA3:Zone Ventilation Sensible Heat Gain Energy [J](Hourly)',
               'BLOCCO1:ZONA3:Zone Ventilation Air Change Rate [ach](Hourly)',
               'BLOCCO1:ZONA1:Zone Ventilation Sensible Heat Loss Energy [J](Hourly)',
               'BLOCCO1:ZONA1:Zone Ventilation Sensible Heat Gain Energy [J](Hourly)',
               'BLOCCO1:ZONA1:Zone Ventilation Air Change Rate [ach](Hourly)',
               'BLOCCO1:ZONA2:Zone Air Relative Humidity [%](Hourly)',
               'BLOCCO1:ZONA3:Zone Air Relative Humidity [%](Hourly)',
               'BLOCCO1:ZONA1:Zone Air Relative Humidity [%](Hourly)'
           ]


        ren = {
            'Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)':'Temp_ext[C]',
            'Environment:Site Outdoor Air Barometric Pressure [Pa](Hourly)':'Pr_ext[Pa]',
            'Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](Hourly)':'SolarRadiation[W/m2]',
            'Environment:Site Solar Azimuth Angle [deg](Hourly)':'AzimuthAngle[deg]',
            'Environment:Site Solar Altitude Angle [deg](Hourly)':'AltitudeAngle[deg]',
            'Environment:Site Wind Speed [m/s](Hourly)':'WindSpeed[m/s]',
            'Environment:Site Direct Solar Radiation Rate per Area [W/m2](Hourly)':'DirectSolarRadiation[W/m2]',
            'DistrictCooling:Facility [J](Hourly)': 'Cooling [J]',
            'DistrictHeating:Facility [J](Hourly)': 'Heating [J]',
            'BLOCCO1:ZONA2:Zone Operative Temperature [C](Hourly)':'Temp_in2[C]',
            'BLOCCO1:ZONA3:Zone Operative Temperature [C](Hourly)':'Temp_in3[C]',
            'BLOCCO1:ZONA1:Zone Operative Temperature [C](Hourly)':'Temp_in1[C]',
            'BLOCCO1:ZONA2:Zone Infiltration Sensible Heat Loss Energy [J](Hourly)':'InfHeatLoss_2[J]',
            'BLOCCO1:ZONA2:Zone Infiltration Sensible Heat Gain Energy [J](Hourly)':'InfHeatGain_2[J]',
            'BLOCCO1:ZONA2:Zone Infiltration Air Change Rate [ach](Hourly)':'InfAirChange_2[ach]',
            'BLOCCO1:ZONA3:Zone Infiltration Sensible Heat Loss Energy [J](Hourly)':'InfHeatLoss_3[J]',
            'BLOCCO1:ZONA3:Zone Infiltration Sensible Heat Gain Energy [J](Hourly)':'InfHeatGain_3[J]',
            'BLOCCO1:ZONA3:Zone Infiltration Air Change Rate [ach](Hourly)':'InfAirChange_3[ach]',
            'BLOCCO1:ZONA1:Zone Infiltration Sensible Heat Loss Energy [J](Hourly)':'InfHeatLoss_1[J]',
            'BLOCCO1:ZONA1:Zone Infiltration Sensible Heat Gain Energy [J](Hourly)':'InfHeatGain_1[J]',
            'BLOCCO1:ZONA1:Zone Infiltration Air Change Rate [ach](Hourly)':'InfAirChange_1[ach]',
            'BLOCCO1:ZONA2:Zone Ventilation Sensible Heat Loss Energy [J](Hourly)':'VentHeatLoss_2[J]',
            'BLOCCO1:ZONA2:Zone Ventilation Sensible Heat Gain Energy [J](Hourly)':'VentHeatGain_2[J]',
            'BLOCCO1:ZONA2:Zone Ventilation Air Change Rate [ach](Hourly)':'VentAirChange_2[ach]',
            'BLOCCO1:ZONA3:Zone Ventilation Sensible Heat Loss Energy [J](Hourly)':'VentHeatLoss_3[J]',
            'BLOCCO1:ZONA3:Zone Ventilation Sensible Heat Gain Energy [J](Hourly)':'VentHeatGain_3[J]',
            'BLOCCO1:ZONA3:Zone Ventilation Air Change Rate [ach](Hourly)':'VentAirChange_3[ach]',
            'BLOCCO1:ZONA1:Zone Ventilation Sensible Heat Loss Energy [J](Hourly)':'VentHeatLoss_1[J]',
            'BLOCCO1:ZONA1:Zone Ventilation Sensible Heat Gain Energy [J](Hourly)':'VentHeatGain_1[J]',
            'BLOCCO1:ZONA1:Zone Ventilation Air Change Rate [ach](Hourly)':'VentAirChange_1[ach]',
            'BLOCCO1:ZONA2:Zone Air Relative Humidity [%](Hourly)':'Humidity_2[%]',
            'BLOCCO1:ZONA3:Zone Air Relative Humidity [%](Hourly)':'Humidity_3[%]',
            'BLOCCO1:ZONA1:Zone Air Relative Humidity [%](Hourly)':'Humidity_1[%]'
        }

        df_ = pd.DataFrame(df, columns=cols)
        df_z = df_.rename(columns=ren)
        df_z['date'] = df['Date/Time'].astype('O')
        df_z['date'] = df_z['date'].map(lambda x: x if '24:00' not in x else x.replace('24:00', '00:00'))
        df_z = df_z.set_index('date')
        df_z = df_z.set_index(pd.to_datetime('2017/' + df_z.index))

        return df_z

    @classmethod
    def energy_signature(cls, iddfile, idf_path, epw_path, name):
        """

        :param iddfile: Energy+.idd file path
        :param idf_path: idf file path
        :param epw_path: weather file path
        :param name: flag name indicating which type of idf is used
        :return: None
        """


        try:
            IDF.setiddname(iddfile)
        except eppy.modeleditor.IDDAlreadySetError as e:
            pass
        idf = IDF(idf_path, epw_path)
        idf.run(readvars=True) # Eplus Simulation

        fname = 'eplusout_' + str(name) + '.csv'
        os.system(f'cp eplusout.csv ../eplus_simulation/eplus/{fname}')



        df1 = pd.read_csv(f'../eplus_simulation/eplus/{fname}')

        df = pd.DataFrame()
        # Temperature
        for i in df1.columns:
            if 'BLOCCO1:ZONA1' in i:
                if 'Zone Operative Temperature [C](Hourly)' in i:
                    df["Temp_in_1"] = df1[i]
            elif 'BLOCCO1:ZONA2' in i:
                if 'Zone Operative Temperature [C](Hourly)' in i:
                    df["Temp_in_2"] = df1[i]
            elif 'BLOCCO1:ZONA3' in i:
                if 'Zone Operative Temperature [C](Hourly)' in i:
                    df["Temp_in_3"] = df1[i]
        df['Temp_out'] = df1['Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)']
        # Power
        df['Cooling'] = df1['DistrictCooling:Facility [J](Hourly)']
        df['Heating'] = df1['DistrictHeating:Facility [J](Hourly)']
        df['Electricity'] = df1['Electricity:Facility [J](Hourly)']

        df['date'] = df1['Date/Time'].astype('O')
        df['date'] = df['date'].map(lambda x: x if '24:00' not in x else x.replace('24:00', '00:00'))
        df = df.set_index('date')
        # idx = df['date'].map(lambda x: x if '24:00' not in x else x.replace('24:00', '00:00'))
        # df.set_index(idx)
        df = df.set_index(pd.to_datetime('2018/' + df.index))
        df['Temp_in'] = df[['Temp_in_1', 'Temp_in_2', 'Temp_in_3']].astype(float).mean(1)
        df['deltaT'] = df['Temp_in'] - df['Temp_out']
        df.to_csv(f'../../files/outputs/en_sig_{name}.csv')

        # ============================================================
        # Energy Signature: HOURLY
        # ============================================================
        # HEATING
        heating_df = df.where(df['Heating']/3.6e6 > 0.2).dropna()
        heating_df = heating_df.resample('H').mean()
        heating_df = heating_df.dropna()
        model_H = sm.OLS(heating_df['Heating']/(3.6e6), sm.add_constant(heating_df['deltaT']))
        results_h = model_H.fit()
        # COOLING
        cool_df = df.where(df['Cooling']/3.6e6 > 0.5).dropna()
        cool_df = cool_df.resample('H').mean()
        cool_df = cool_df.dropna()
        model_C = sm.OLS(cool_df['Cooling']/(3.6e6), sm.add_constant(cool_df['deltaT']))
        results_c = model_C.fit()
        # Plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 10))
        fig.suptitle("Energy Signature")
        ax1.plot(heating_df['Temp_out'], results_h.predict(), label='Heating')
        ax1.plot(cool_df['Temp_out'], results_c.predict(), label='Cooling')
        ax1.scatter(heating_df['Temp_out'], heating_df['Heating']/(3.6e6))
        ax1.scatter(cool_df['Temp_out'], cool_df['Cooling']/(3.6e6))

        ax1.set_xlabel('T_out [C°]')
        ax1.set_ylabel('Energy Consumption [kWh]')
        ax1.set_title('Hourly resample')
        ax1.legend()
        ax1.grid(linestyle='--', linewidth=.4, which='both')

        # ============================================================
        # Energy Signature: DAILY
        # ============================================================
        # HEATING
        heating_df = df.where(df['Heating']/3.6e6 > 0.2).dropna()
        heating_df = heating_df.resample('D').mean()
        heating_df = heating_df.dropna()
        model_h = sm.OLS(heating_df['Heating']/3.6e6, sm.add_constant(heating_df['deltaT']))
        results_d_h = model_h.fit()

        # COOLING
        cool_df = df.where(df['Cooling']/3.6e6 > 0.5).dropna()
        cool_df = cool_df.resample('D').mean()
        cool_df = cool_df.dropna()
        model_c = sm.OLS(cool_df['Cooling']/(3.6e6), sm.add_constant(cool_df['deltaT']))
        results_d_c = model_c.fit()


        ax2.plot(heating_df['Temp_out'], results_d_h.predict(), label='Heating')
        ax2.plot(cool_df['Temp_out'], results_d_c.predict(), label='Cooling')
        ax2.scatter(heating_df['Temp_out'], heating_df['Heating']/(3.6e6))
        ax2.scatter(cool_df['Temp_out'], cool_df['Cooling']/(3.6e6))
        ax2.set_xlabel('T_out [C°]')
        ax2.set_ylabel('Energy Consumption [kWh]')
        ax2.set_title('DAY resample')
        ax2.legend()
        ax2.grid(linestyle='--', linewidth=.4, which='both')

        # ============================================================
        # Energy Signature: WEEK
        # ============================================================
        # HEATING
        heating_df = df.where(df['Heating']/3.6e6 > 0.2).dropna()
        heating_df = heating_df.resample('W').mean()
        heating_df = heating_df.dropna()
        model_h = sm.OLS(heating_df['Heating']/(3.6e6), sm.add_constant(heating_df['deltaT']))
        results_w_h = model_h.fit()

        # COOLING
        cool_df = df.where(df['Cooling']/3.6e6 > 0.5).dropna()
        cool_df = cool_df.resample('W').mean()
        cool_df = cool_df.dropna()
        model_c = sm.OLS(cool_df['Cooling']/(3.6e6), sm.add_constant(cool_df['deltaT']))
        results_w_c = model_c.fit()


        ax3.plot(heating_df['Temp_out'], results_w_h.predict(), label='Heating')
        ax3.plot(cool_df['Temp_out'], results_w_c.predict(), label='Cooling')
        ax3.scatter(heating_df['Temp_out'], heating_df['Heating']/(3.6e6))
        ax3.scatter(cool_df['Temp_out'], cool_df['Cooling']/(3.6e6))
        ax3.set_xlabel('T_out [C°]')
        ax3.set_ylabel('Energy Consumption [kWh]')
        ax3.set_title('WEEK resample')
        ax3.legend()
        ax3.grid(linestyle='--', linewidth=.4, which='both')
        plt.subplots_adjust(bottom=0.3, right=0.8, top=0.9, hspace=1)
        plt.savefig(fname=f'../../plots/energy_signature_{name}_tout.png', dpi=400)
        plt.close()





class Optimal_config :
    
    def __init__(self,idf,epw):
        self.building = ef.get_building(idf)
        self.name = os.path.basename(idf).replace('.idf','')
        self.epw = epw
        self.evaluation_output = '/home/ict4db/Scrivania/ICT4BD_git-master/files/outputs/'+self.name+'_DB.xlsx'
        #IDD_file = '/usr/local/EnergyPlus-9-0-1/Energy+.idd'
        IDD_file = '/home/ict4db/py3/lib/python3.6/site-packages/besos-1.3.3-py3.6.egg/Data/example_idd.idd'
        self.energy_evaluator(self.building)
        self.correlation(self.evaluation_output)
        self.plotting(self.evaluation_output)
        self.final_idf(idf,IDD_file)

    def energy_evaluator(self,building):
        
        list1=np.linspace(0.01,0.35,6,endpoint=False)
        list2=np.linspace(0.9,6,6,endpoint=False)
        #list1=np.linspace(0.01,0.35,2,endpoint=False) #test
        #list2=np.linspace(0.9,6,2,endpoint=False) #test
        list1 = np.round(list1,decimals=3)
        list2 = np.round(list2,decimals=2)
        wwr_list = [0.15,0.5,0.9]
        #wwr_list = [0.15] #test
        

        insul_range = CategoryParameter(options=list1)
        U_window_range = CategoryParameter(options=list2)
        wwr_range = CategoryParameter(options=wwr_list)
        #wwr_range = RangeParameter(0.15,0.9) #test

        insulation_wall = FieldSelector(class_name='Material', object_name='SuperInsulating_00795', field_name='Thickness')
        insulation_roof = FieldSelector(class_name='Material', object_name='SuperInsulating_01445', field_name='Thickness')
        glazing = FieldSelector(class_name='WindowMaterial:SimpleGlazingSystem',object_name ='Simple 1001',field_name ='U-Factor')
        
        
        #glazing = FieldSelector(class_name='WindowMaterial:SimpleGlazingSystem',object_name ='Simple 1001',field_name ='UFactor')
        
        insul_param_wall = Parameter(selector=insulation_wall,value_descriptor=insul_range,name ='Insulation Thickness wall')
        insul_param_roof = Parameter(selector=insulation_roof,value_descriptor=insul_range,name ='Insulation Thickness roof')
        U_window_param = Parameter(selector=glazing,value_descriptor=U_window_range,name ='windows-U-factor')
        windows_to_wall = wwr(wwr_range) #this function equivalent to fieldselctor+parameter functions 
        
    


        #CREATION OF THE PROBLEM
        parameters=[insul_param_wall]+[windows_to_wall]+[U_window_param]+[insul_param_roof]
        #parameters=[insul_param_wall]+[windows_to_wall]+[more_parameters]+[insul_param_roof] #test
        
        objectives = ['Electricity:Facility','DistrictHeating:Facility','DistrictCooling:Facility'] # these get made into `MeterReader` or `VariableReader`
        problem = EPProblem(parameters, objectives)
        samples = self.create_samples(wwr_list,list1,list1,list2)
       
        
        #CREATING EVALUATIONS
        evaluator = EvaluatorEP(problem, building, out_dir='OUTPUTS', err_dir='ERR_OUTPUTS' ,epw=self.epw) # evaluator = problem + building
        outputs = evaluator.df_apply(samples, keep_input=True)
        outputs = outputs.sort_values(by =['Insulation Thickness wall','windows-U-factor','Insulation Thickness roof'])
        
        outputs.to_excel(self.evaluation_output)
        #print(outputs)
        #print(outputs.describe())

    def create_samples(self,win_to_wall,wall_t,roof_t,u_wind):
        array = []
        for i in wall_t:
            for wwr in win_to_wall:
                for k in u_wind:
                    for j in roof_t:
                        tmp_row =[i,wwr,k,j]
                        array.append(tmp_row)

        df = pd.DataFrame(array)
        df.columns = ['Insulation Thickness wall','Window to Wall Ratio','windows-U-factor','Insulation Thickness roof']    
        return df

    def correlation (self,file):
        
        out = pd.read_excel(file)
        out= out.drop(columns='Unnamed: 0')
        corr = out.corr()
        labels = list(corr.keys())
        corr_matrix = corr.to_numpy()
        corr_matrix = np.around(corr_matrix,decimals=3)
        #print(corr_matrix)

        
        fig, ax = plt.subplots()
        im = ax.imshow(corr_matrix)

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, corr_matrix[i, j], ha="center", va="center", color="w")
        
        ax.set_title("heatmap of correlations")
        fig.tight_layout()
        plt.savefig('/home/ict4db/Scrivania/ICT4BD_git-master/plots/'+self.name+'_heatmap.png')

    def plotting(self,file):
        df = pd.read_excel(file)
        df = df.round(decimals={'Insulation Thickness wall':3,'windows-U-factor':1,'Insulation Thickness roof':3})
        list1=df['Insulation Thickness wall'].unique() #variation thickness
        #print (len(list1))
        list2=df['windows-U-factor'].unique() #variation u window
        #print(len(list2))
        list3= df['Insulation Thickness roof'].unique()
        #print (len(list3))

        #plot U-window and roof thickness FIXED and thickness wall varying (3 wwr configurations):
        for wwr in [0.15,0.5,0.9]:
            title = 'WWR = '+str(wwr*100)+'%'
            plt.figure()
            plt.title(title)
            for k in range(len(list1)):

                i = list2[k]
                j = list3[k]
                tmp_df = df.loc [(df['windows-U-factor']== i)&(df['Insulation Thickness roof']==j)]
                tmp_df = tmp_df.loc[tmp_df['Window to Wall Ratio']==wwr]
                tmp_df = tmp_df.sort_values(by=['Insulation Thickness wall'])
                label = 'U-wind='+str(i)+',roof='+str(j)
                x = list(tmp_df['Insulation Thickness wall'])
                y = list(tmp_df['DistrictHeating:Facility'])
                plt.plot(x,y,label=label)
            

            plt.legend()
            plt.xlabel('thickness [m]')
            plt.ylabel('Heating [kWh]')
            plt.savefig('/home/ict4db/Scrivania/ICT4BD_git-master/plots/'+'thick_wall_'+self.name+str(title)+'.png')
            plt.close()
        
        #plot U-window and wall thickness FIXED and thickness roof varying (3 wwr configurations):
        for wwr in [0.15,0.5,0.9]:
            title = 'WWR = '+str(wwr*100)+'%'
            plt.figure()
            plt.title(title)
            for k in range(len(list1)):

                i = list1[k]
                j = list2[k]
                tmp_df = df.loc [(df['Insulation Thickness wall']== i)&(df['windows-U-factor']==j)]
                tmp_df = tmp_df.loc[tmp_df['Window to Wall Ratio']==wwr]
                tmp_df = tmp_df.sort_values(by=['Insulation Thickness roof'])
                label = 'wall='+str(i)+',wind_U='+str(j)
                x = list(tmp_df['Insulation Thickness roof'])
                y = list(tmp_df['DistrictHeating:Facility'])
                plt.plot(x,y,label=label)
            plt.yscale('log',basey=10)
            plt.legend()
            plt.xlabel('roof thickness[m]')
            plt.ylabel('Heating [kWh]')
            plt.savefig('/home/ict4db/Scrivania/ICT4BD_git-master/plots/'+'thick_roof_'+self.name+str(title)+'.png')
            #plt.savefig('plots/thickness_roof/'+str(title)+'U_window.png')
            plt.close()


        #plot wall thickness and roof thickness FIXED and U-window varying (3 wwr configurations):
        for wwr in [0.15,0.5,0.9]:
            title = 'WWR = '+str(wwr*100)+'%'
            plt.figure()
            plt.title(title)
            for k in range(len(list1)):

                i = list1[k]
                j = list3[k]
                tmp_df = df.loc [(df['Insulation Thickness wall']== i)&(df['Insulation Thickness roof']==j)]
                tmp_df = tmp_df.loc[tmp_df['Window to Wall Ratio']==wwr]
                tmp_df = tmp_df.sort_values(by=['windows-U-factor'])
                print (tmp_df)
                print (tmp_df['windows-U-factor'])
                label = 'wall='+str(i)+',roof='+str(j)
                x = list(tmp_df['windows-U-factor'])
                y = list(tmp_df['DistrictHeating:Facility'])
                plt.plot(x,y,label=label)
            plt.legend()
            plt.xlabel('window U [W/m^2 K]')
            plt.ylabel('Heating [kWh]')
            plt.savefig('/home/ict4db/Scrivania/ICT4BD_git-master/plots/'+'U_wind_'+self.name+str(title)+'.png')
            #plt.savefig('plots/U_window/'+str(title)+'U_window.png')
            plt.close()


    def find_optimum(self,dataframe):
        #df = pd.read_excel(file)
        #renaming columns
        df = dataframe
        old_names = list(df.keys())
        new_name ={
                    'Insulation Thickness wall':'wall_thick',
                    'Window to Wall Ratio':'wwr',
                    'windows-U-factor':'wind_U',
                    'Insulation Thickness roof':'roof_thick',
                    'Electricity:Facility':'Electricity',
                    'DistrictHeating:Facility':'Heating',
                    'DistrictCooling:Facility':'Cooling',
                    }
        df = df.rename(columns=new_name)
        min_H = df[df.Heating == df.Heating.min()]
        min_E= df[df.Electricity == df.Electricity.min()]
        min_C = df[df.Cooling == df.Cooling.min()]


        #initialization
        best_E = 10000000000000000
        best_C = 10000000000000000
        best_H = 10000000000000000

        for index, row in df.iterrows():
            E = row['Electricity']
            C = row['Cooling']
            H = row['Heating']
            E_diff = best_E-E
            H_diff = best_H-H
            C_diff = best_C-C

            #if (E <= best_E and H < best_H and C < best_C):
            #if (H < best_H and C < best_C):
            if E_diff >= 100 and C_diff >= 500 and H_diff >= 500:
                best_index = index
                best_E = E
                best_C = C
                best_H = H
            else:
                continue
        #print(df.iloc[best_index,:])

        return best_index



    def final_idf(self,idf,idd):
        #per questa fase ricordarsi di installare pacchetto geomeppy in anaconda gia presente
        geom_IDF.setiddname(idd)
        idf1 = geom_IDF(idf)
        df = pd.read_excel(self.evaluation_output)
        #running the function to find the optimal values among all the simulations
        best_index = self.find_optimum(df)
        
        Material = idf1.idfobjects['MATERIAL']
        Window_Material = idf1.idfobjects['WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM']
        #MATERIAL:#'WindowMaterial:SimpleGlazingSystem'
        #ROOF: 'SuperInsulating_01445', POSITION: 2
        #WALL: 'SuperInsulating_00795', POSITION: 7
        #window material simple glazing...
        #WINDOW: 'Simple 1001', POSITION: 0
        
        #print(df.keys())
        print('==================================')
        print('best configuration:')
        print('thick_wall: ',df.loc[best_index,'Insulation Thickness wall'])
        print('thick_roof: ',df.loc[best_index,'Insulation Thickness roof'])
        print('U_value: ',df.loc[best_index,'windows-U-factor'])
        print('WWR: ',df.loc[best_index,'Window to Wall Ratio'])
        print('==================================')

        Material[2].Thickness = df.loc[best_index,'Insulation Thickness roof']
        Material[7].Thickness = df.loc[best_index,'Insulation Thickness wall']
        Window_Material[0].UFactor = df.loc[best_index,'windows-U-factor']
        
        #print (Window_Material[0].UFactor)


        #changing the wwr
        
        new_wwr = df.loc[best_index,'Window to Wall Ratio']
        old_fen = idf1.idfobjects['FENESTRATIONSURFACE:DETAILED']
        idf1.set_wwr(wwr=new_wwr)
        new_fen = idf1.idfobjects['FENESTRATIONSURFACE:DETAILED']
        
        

        #SAVE THE ULTIMATE IDF FILE chose the directory in saveas
        idf1.saveas('/home/ict4db/Scrivania/ICT4BD_git-master/files/idf/'+self.name+'_Optimal.idf')





if __name__ == '__main__':

    learn = Prediction()
    df = pd.read_csv('eplus_simulation/eplus/eplusout.csv')
    df = learn.create_csv(df)



