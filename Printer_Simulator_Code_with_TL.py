import time
import numpy as np
import pandas as pd
import pywt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Predictor function using Decision Tree Regressor and Grid Search for hyperparameter tuning
def Predictor(train_X, train_y, param_grid, random_state=42, cv=6):
    """
    Train a Decision Tree Regressor model using Grid Search for hyperparameter optimization.
    Args:
    - train_X: Training data features
    - train_y: Training data labels
    - param_grid: Grid of hyperparameters for tuning
    - random_state: Seed for random number generator
    - cv: Number of cross-validation folds
    Returns:
    - Trained Decision Tree Regressor model
    """
    # Instantiate and fit the grid search model
    grid_search = GridSearchCV(estimator=DecisionTreeRegressor(random_state=random_state),
                               param_grid=param_grid, cv=cv)
    grid_search.fit(train_X, train_y)

    # Retrieve the best parameters and retrain the model
    best_params = grid_search.best_params_
    best_estimator = DecisionTreeRegressor(**best_params, random_state=random_state)
    best_estimator.fit(train_X, train_y)

    return best_estimator

# Function for reconstructing data using wavelet transformation
def reconstruct_data(inputs, threshold):
    """
    Reconstruct data using wavelet transformation for noise reduction.
    Args:
    - inputs: Input data to be processed
    - threshold: Threshold values for the wavelet transformation
    Returns:
    - Reconstructed data after applying wavelet transformation
    """
    # Decomposition and denoising using wavelet
    input_dimension = inputs.shape[-1]
    coeffs = pywt.wavedecn(inputs, wavelet='db1', level=2, mode='symmetric', axes=[0, 1, 2])

    # Process detail coefficients at each decomposition level
    denoised_detail = []
    for dcoeff in coeffs[1:]:
        denoised_detail_level = dict()
        for key in dcoeff:
            dcoeff_dimension = [pywt.threshold(dcoeff[key][:, :, :, dimension], value=threshold[dimension],
                                               mode='hard', substitute=0) for dimension in range(input_dimension)]
            denoised_detail_level[key] = np.stack(dcoeff_dimension, 3)
        denoised_detail.append(denoised_detail_level)

    # Reconstruction using wavelet
    coeffs[1:] = denoised_detail
    true_signals = pywt.waverecn(coeffs, 'db1', mode='symmetric', axes=[0, 1, 2])

    return true_signals

# Main execution block
if __name__ == '__main__':
    start_time = time.time()

    # Define input parameters
    Printing_Pressure = 60  # Printing Pressure
    Printing_Speed = 30  # Printing Speed
    Separation_Speed = 3.5 # Separation Speed
    Cleaning_Age = 3   # Cleaning Age
    Cleaning_Type = 'Wet'
    Direction = 'F'  # Direction
    Board_type = 'EKRA'  # Board type
    Printer = 'EKRA'  # Printer type

    # Import the dataset
    SpiData = pd.read_csv('C:/Users/Rahul22/Downloads/BU_Printer_Simulator_Updated/Printer_Simulator_With_TL/PrinterSimulatorDataset.csv')

    # Prepare output DataFrame
    Output = pd.DataFrame()
    Output['PadID'] = SpiData[:4500]['PadID']
    Output['Rotation'] = SpiData[:4500]['Rotation']
    Output['AR'] = SpiData[:4500]['AR']
    Output['ASR'] = SpiData[:4500]['ASR']
    Output['Printing Speed'] = Printing_Speed
    Output['Printing Pressure'] = Printing_Pressure
    Output['Separation Speed'] = Separation_Speed
    Output['Cleaning Age'] = Cleaning_Age
    Output['Cleaning Type'] = Cleaning_Type
    Output['Direction'] = Direction

    # Define input and output columns for the model
    input_columns = ['Printing Speed', 'Printing Pressure', 'Separation Speed', 'AR', 'ASR', 'Cleaning Age']
    output_columns = ['Volume(%)', 'OffsetX(um)', 'OffsetY(um)']

    # Process volume data
    input_data = np.array(SpiData[output_columns]).reshape(624, 90, 50, len(output_columns))
    TrueSignal = reconstruct_data(input_data, threshold=[11.02, 7.02, 7.52])
    TrueSignal = TrueSignal.reshape(len(SpiData), len(output_columns))

    # Calculate true and noise values
    SpiData[['True Volume', 'True OffsetX', 'True OffsetY']] = TrueSignal
    SpiData[['Noise Volume', 'Noise OffsetX', 'Noise OffsetY']] = SpiData[output_columns] - TrueSignal

    # Generate simulated data with random noise
    Aperture = SpiData["AR"].unique()
    SpiData[['Simulated Volume', 'Simulated OffsetX', 'Simulated OffsetY']] = TrueSignal
    for aperture in Aperture:
        b_aperture = SpiData['AR'] == aperture
        apertureData = SpiData.loc[b_aperture]
        for name in ['Volume', 'OffsetX', 'OffsetY']:
            noise_samples = np.random.normal(np.mean(apertureData[f'Noise {name}']),
                                             np.std(apertureData[f'Noise {name}']), len(apertureData))
            SpiData.loc[b_aperture, f'Simulated {name}'] += noise_samples


    # Set up grid search parameters for different outputs
    param_grid = {
        'Volume': {
            'avg': {'max_depth': [8, 10, 15, 25, 40, 50, 60],
                    'min_samples_split': [50, 70, 80, 100],
                    'max_features': ['log2', 'sqrt']},
            'std': {'max_depth': [5, 8, 10, 20, 30],
                    'min_samples_split': [30, 40, 50, 70]}
        },
        'OffsetX': {
            'avg': {'max_depth': [10, 20, 30],
                    'min_samples_split': [40, 60, 80],
                    'max_features': ['auto', 'sqrt']},
            'std': {'max_depth': [5, 10, 15],
                    'min_samples_split': [20, 30, 50],
                    'max_features': ['auto', 'log2']}
        },
        'OffsetY': {
            'avg': {'max_depth': [10, 15, 20, 25],
                    'min_samples_split': [30, 50, 70],
                    'max_features': ['sqrt', 'log2']},
            'std': {'max_depth': [7, 10, 13],
                    'min_samples_split': [25, 40, 55],
                    'max_features': ['auto', 'sqrt']}
        }
    }

    # Define aggregation functions
    aggregation = {
        'Simulated Volume': ['mean', 'std'],
        'Simulated OffsetX': ['mean', 'std'],
        'Simulated OffsetY': ['mean', 'std']
    }

    # Conditional processing based on Board Type and Printer Type
    if Board_type == 'MOM4' and Printer == 'EKRA':
        # Processing specific to MOM4 and EKRA
        SimulatedDataVolume = SpiData.loc[(SpiData['Cleaning Type'] == Cleaning_Type) & (SpiData['Direction'] == Direction)]
        summarizedData = SimulatedDataVolume.groupby(
            ['PCB ID', 'Printing Speed', 'Printing Pressure', 'Separation Speed', 'AR', 'ASR', 'Cleaning Age', 'Rotation'],
            as_index=False).agg(aggregation)

        # Iterate through each rotation and perform predictions
        Rotation = summarizedData['Rotation'].unique()
        for rotation in Rotation:
            summarizedData_rotation = summarizedData.loc[summarizedData['Rotation'] == rotation]
            Output_rotation = Output[Output['Rotation'] == rotation]

            train_X = summarizedData_rotation[input_columns].values
            test_X = Output_rotation[input_columns].values

            # Predict values for each output column
            for name in ['Volume', 'OffsetX', 'OffsetY']:
                train_avg = summarizedData_rotation[('Simulated %s' % name, 'mean')].values
                train_std = summarizedData_rotation[('Simulated %s' % name, 'std')].values

                predicted_avg = Predictor(train_X, train_avg, param_grid[name]['avg']).predict(test_X)
                predicted_std = Predictor(train_X, train_std, param_grid[name]['std']).predict(test_X)

                Output.loc[Output.Rotation == rotation, name] = predicted_avg + predicted_std * np.random.randn(len(predicted_avg))
        # Save the output to a CSV file
        Output.to_csv('Simulator Output.csv', index=0)

    else:
        # Data processing for other board types and printers
        # Aggregate data and process
        # Load your simulation data and real data
        aggregation = {'Simulated Volume': ['mean', 'std']}
        SimulatedDataVolume = SpiData.groupby(
            ['PCB ID', 'Printing Speed', 'Printing Pressure', 'Separation Speed', 'AR', 'ASR', 'Cleaning Age', 'Direction'],
            as_index=False).agg(aggregation)
            
        # Convert the multi-index columns in summarizedData to a single-level index
        SimulatedDataVolume.columns = ['_'.join(filter(None, col)).strip() for col in SimulatedDataVolume.columns.values]

        # Rename only specific columns
        column_rename_map = {
            'Simulated Volume_mean': 'VolAVG',
            'Simulated Volume_std': 'VolSTD'
        }

        SimulatedDataVolume.rename(columns=column_rename_map, inplace=True)
         # Define the input and output columns for simulation data
        X_sim_columns = ['Printing Speed', 'Printing Pressure', 'Separation Speed', 'AR', 'ASR', 'Cleaning Age', 'Direction']
        Y_sim_columns = ['VolAVG', 'VolSTD']

        # Define the input and output columns for real data (assuming they have the same columns)
        X_real_columns = ['Printing Speed', 'Printing Pressure', 'Separation Speed', 'AR', 'ASR', 'Cleaning Age', 'Direction']
        Y_real_columns = ['VolAVG', 'VolSTD']

        # Load your simulation data and real data
        simulated_data = SimulatedDataVolume
        # Replace 'F' with 1 and 'B' with 2 in the direction column
        simulated_data["Direction"] = simulated_data[X_sim_columns[-1]].map({'F': 1, 'B': 2}) 
        new_training_data=pd.read_csv('C:/Users/Rahul22/Downloads/BU_Printer_Simulator_Updated/Printer_Simulator_With_TL/NewJobFileTrainingData.csv')
        volname='Volume(%)'

        def newagg(dataframe):
            aggregations={volname:[('VolAVG','mean'),('VolSTD','std'),('Count','count')]}
            SummariedData=dataframe.groupby(["PCB ID","Printing Speed", "Printing Pressure","Separation Speed","AR","ASR","Cleaning Age","Direction"],as_index=False).agg(aggregations) 
            RenameCol=list(SummariedData.columns.get_level_values(1))
            RenameCol[:8]=["PCB ID","Printing Speed", "Printing Pressure","Separation Speed","AR","ASR","Cleaning Age","Direction"] 
            SummariedData.columns =RenameCol
            return(SummariedData)
        real_data_training=newagg(new_training_data)
        real_data_training['Direction'] = real_data_training[X_sim_columns[-1]].map({'F': 1, 'B': 2}) 
       

        # Normalize the input and output data separately
        scaler_X = StandardScaler()

        X_sim_scaled = scaler_X.fit_transform(simulated_data[X_sim_columns].values)
        Y_sim_scaled = simulated_data[Y_sim_columns].values

        X_real_scaled = scaler_X.transform(real_data_training[X_real_columns].values)
        Y_real_scaled = real_data_training[Y_real_columns].values


        X_real_train = X_real_scaled
        Y_real_train = Y_real_scaled
        Y_real_columns1 = ['Actual VolAVG', 'Actual VolSTD']

        # Create a neural network model
        def create_model(input_shape, output_shape):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(output_shape, activation='linear')  # Output layer with linear activation
            ])
            return model

        # Create and compile the model for pre-training
        input_shape = (X_sim_scaled.shape[1],)  # Assuming X_sim_scaled is your simulated data features
        output_shape = Y_sim_scaled.shape[1]  # Assuming Y_sim_scaled is your simulated data labels
        model = create_model(input_shape, output_shape)
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Pre-train the model on simulation data
        history_pretrain = model.fit(X_sim_scaled, Y_sim_scaled, epochs=100, verbose=0)  

        # Fine-tune the model on real data
        history_finetune = model.fit(X_real_train, Y_real_train, epochs=700, verbose=0)  
        
        Output = pd.DataFrame()
        Number_of_unique_AR_and_ASR=real_data_training[['AR','ASR','Count']].drop_duplicates()
        Output['Index']=range(0,len(Number_of_unique_AR_and_ASR))
        Output['Printing Speed'] = Printing_Speed
        Output['Printing Pressure'] = Printing_Pressure
        Output['Separation Speed'] = Separation_Speed
        Output['AR'] = Number_of_unique_AR_and_ASR['AR']
        Output['ASR'] = Number_of_unique_AR_and_ASR['ASR']
        Output['Cleaning Age'] = Cleaning_Age
        Output['Direction'] = Direction
        Output['Direction'] = Output[X_sim_columns[-1]].map({'F': 1, 'B': 2}) 
        Output=Output.drop(columns=['Index'])
        Y_pred = model.predict(scaler_X.transform(Output.values))
        # Assign predictions to separate columns in Output
        Output['VolAVG'] = Y_pred[:, 0]
        Output['VolSTD'] = Y_pred[:, 1]
        Output['Count'] = Number_of_unique_AR_and_ASR['Count']
        
        # Function to generate data and expand rows for generating results on pad level
        def expand_row(row):
            avg = row['VolAVG']
            std = row['VolSTD']
            count = int(row['Count'])  
            
            # Generate 'count' number of data points
            generated_data = avg + std * np.random.randn(count)
            
            # Create a new dataframe from the generated data, keeping other columns the same
            expanded_data = pd.DataFrame({col: np.repeat(row[col], count) for col in Output.columns})
            expanded_data['Volume(%)'] = generated_data  # This is your new column
            return expanded_data

        # Apply the function to each row and concatenate the result into a new dataframe
        expanded_data = pd.concat([expand_row(row) for _, row in Output.iterrows()])
        expanded_data.drop(['VolAVG','VolSTD','Count'], axis=1, inplace=True)
        print(expanded_data)
        # Save the processed data
        expanded_data.to_csv('Simulator Output.csv', index=0)

    print('time : ', time.time() - start_time)


