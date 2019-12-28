from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 15]
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.inspection import plot_partial_dependence
from sklearn.neural_network import MLPRegressor
import seaborn as sns

from eppy.modeleditor import IDF
from besos import eppy_funcs as ef
from besos.evaluator import EvaluatorEP
from besos.parameters import FieldSelector, Parameter, expand_plist
from besos.problem import EPProblem
import logging



logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s : %(message)s',
                    datefmt='%d/%m/%Y %H:%M ',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)





def heatmap(X, columns, name):

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
    plt.savefig('../plots/' + fname)


# Neural Network Multi-layer Perceptron regressor

def nnMR(samples, output, test_size=0.2, random_state=123):


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


def parametric_analysis(dinamic_parameter, fixed_parameters, objectives, idf_path, epw_path, n_points=60):
    """
    this method allows to do a parametric analysis by varying a parameter while maintaining fixed others.

    parameter: dinamic_parameter: a dictionary containing information about the parameter that must be changed.
    {class_name: Material, object_name: Simply_1001, field_name: UFactor}
    fixed_parameters: a list of dictionary containing information about the parameters that must be fixed.
    The dictionary has the same structure of the dinamic_parameter dictionary.
    objectives: a list lof the objectives,
    idf_path: a str with the path for idf file,
    weather_path: str path for epw file.
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

    parameters = [Parameter(selector=x) for x in ls_fp]

    if dinamic_parameter['object_name'] == 'Simple 1001':
        logger.info(f"Parametric analysis for {dinamic_parameter['object_name']} will start.")
        range_ufw = np.linspace(-5, -0.3, n_points)
        dict_ = {}
        for j in range(6):
            for i in range(len(fixed_parameters)):
                #dict_[parameters[i].selector.object_name] = [((i+2)**2*0.1)]*n_points
                dict_[parameters[i].selector.object_name] = [(j+1)*0.1]*n_points
            df_samples = pd.DataFrame.from_dict(dict_)
            df_samples[parameters[len(fixed_parameters)].selector.object_name] = range_ufw*-1
            problem = EPProblem(parameters, objectives)
            evaluator = EvaluatorEP(problem, building, epw_file=epw_path, out_dir='../out_dir', err_dir='../err_dir')
            outputs = evaluator.df_apply(df_samples, keep_input=True)
            logger.info(f"Simulation number {j} done.")
            df_samples['DistrictHeating:Facility'] = outputs['DistrictHeating:Facility']
            df_samples['DistrictCooling:Facility'] = outputs['DistrictCooling:Facility']
            df_samples['Electricity:Facility'] = outputs['Electricity:Facility']
            df_samples.to_csv('../files/outputs/outputs_60p_ch_ufw' + str(j) + '.csv')
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
            df_samples = pd.DataFrame.from_dict(dict_)
            df_samples[parameters[len(fixed_parameters)].selector.object_name] = range_t
            problem = EPProblem(parameters, objectives)
            evaluator = EvaluatorEP(problem, building, epw_file=epw_path, out_dir='../out_dir', err_dir='../err_dir')
            outputs = evaluator.df_apply(df_samples, keep_input=True)
            logger.info(f"Simulation number {j} done.")
            df_samples['DistrictHeating:Facility'] = outputs['DistrictHeating:Facility']
            df_samples['DistrictCooling:Facility'] = outputs['DistrictCooling:Facility']
            df_samples['Electricity:Facility'] = outputs['Electricity:Facility']
            df_samples.to_csv(
                '../files/outputs/outputs_60p_ch_twr' + str(j) + '.csv')


def create_csv(df):

    df1 = pd.DataFrame()
    for i in df.columns:
        if '(Hourly)' in i:
            if 'Zone Operative Temperature' in i or 'Environment:Site Outdoor Air Drybulb Temperature' in i:
                df1[i] = df[i]
            elif 'Air Barometric Pressure' in i or 'Site Diffuse Solar Radiation' in i:
                df1[i] = df[i]
            elif 'Site Solar Azimuth' in i or 'Site Solar Altitude' in i:
                df1[i] = df[i]
            elif 'Zone People Sensible Heating' in i or 'Zone Mean Air Temperature' in i or 'Zone Mean Radiant Temperature' in i:
                df1[i] = df[i]
            elif 'Zone Ventilation Sensible Heat' in i or 'Zone Infiltration Sensible Heat' in i:
                df1[i] = df[i]
            elif 'Air Change Rate' in i or 'Relative Humidity' in i:
                df1[i] = df[i]

    df1['date'] = df['Date/Time'].astype('O')
    df1['date'] = df1['date'].map(lambda x: x if '24:00' not in x else x.replace('24:00', '00:00'))
    df1 = df1.set_index('date')
    df1 = df1.set_index(pd.to_datetime('2018/' + df1.index))
    return df1


def energy_signature(iddfile, fname, epw):

    IDF.setiddname(iddfile)
    idf = IDF(fname, epw)
    idf.run(readvars=True)

    df1 = pd.read_csv('eplusout.csv')

    df = pd.DataFrame()

    # Temperature
    for i in df1.columns:
        if 'Zone Operative Temperature [C](Hourly)' in i or 'Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)' in i:
            df[i] = df1[i]
        elif 'Zone Ventilation Sensible Heat Loss Energy [J](Hourly)' in i:
            df[i] = df1[i]

    df['date'] = df1['Date/Time'].astype('O')
    df['date'] = df['date'].map(lambda x: x if '24:00' not in x else x.replace('24:00', '00:00'))
    df = df.set_index('date')
    # idx = df['date'].map(lambda x: x if '24:00' not in x else x.replace('24:00', '00:00'))
    # df.set_index(idx)
    df = df.set_index(pd.to_datetime('2018/' + df.index))

    df.to_csv('../files/outputs/en_sig.csv')
    return df

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
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