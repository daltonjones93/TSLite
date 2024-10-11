# TSLite

## Description


This project provides a simple unified API for timeseries data analysis, feature engineering, machine learning model training and optimization and model performance evaluation.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

Follow these steps to set up the project on your local machine:

1. **Clone the repository**:
   Open your terminal or command prompt and run the following command to clone the repository:
   ```bash
   git clone https://github.com/username/repository.git
2. **Install dependencies**:
   From within the project directory run
   ```bash
   pip install -r requirements.txt


## Usage

To get started using the repo 

1. **Download Data, Perform Analysis**:
    ```python
    from dataset_timeseries import TimeseriesData
    ts = TimeseriesData('data.csv', target_col = 'CAISO_system_load', time_index = 'interval_start_time')
    ts.pairplot()
    ts.plot_seasonal_decomposition()


2. **Timeseries Feature Engineering**:
    ```python
    ts.create_hour_feature(convert_to_cyclic = True)
    ts.create_fft_feature(window = 24 * 5)
    t.fill_missing_values()
    t.scale_features()
    n_steps_predict = 24
    past_states = 96
    #create training dataset (for recurrent models like transformer or LSTM)
    xTrain,xVal,xTest,yTrain,yVal,yTest = t.return_training_data_recurrent(past_states,n_steps_predict)
    input_size = xTrain.shape[-1]

3. **Model Loading and Training**:
    ```python
    #load model from json config
    from model import MLModel
    import os
    from constants import CONFIG_DIR
    #specify config file
    json_config = os.path.join(CONFIG_DIR,"lstm_config.json")
    model = MLModel(json_config)

    #alternately load model using specified parameters
    from models.pytorch_models import pytorchModel
    kwargs = {'input_size' : input_size, 'output_size':n_steps_predict,"hidden_size":64,'n_layers':2,'batch_size':64,
            'lr':.001,'epochs':2}
    model = pytorchModel(model_type = "transformer",**kwargs)

    #Train model
    model.fit(xTrain,yTrain)

    #make predictions
    model.predict(xTest,yTest)

4. **Optimize Hyperparameters**
    ```python
    from models.pytorch_models import grid_search

    param_grid = {
        'input_size': [12],              # Fixed input size
        'output_size': [24],              # Fixed output size (for example, binary classification)
        'hidden_size': [4, 8],         # Try hidden sizes of 32 and 64
        'n_layers': [1, 2],              # Try 1 or 2 LSTM layers
        'lr': [0.001, 0.0001],           # Try learning rates of 0.001 and 0.0001
        'epochs': [1],              # Train for 10 or 20 epochs
        'batch_size': [4]           # Try batch sizes of 32 and 64
    }

    #this function outputs best model parameters to a model specific json file

    best_params, best_score = grid_search(xTrain, yTrain,
                                          xVal, yVal, param_grid,
                                          model_type = 'transformer')

5. **Compare Performance**
    ```python
    # Initialize the comparer
    from eval import TimeSeriesModelComparer

    comparer = TimeSeriesModelComparer()

    # Add models
    kwargs = {'input_size' : input_size, 'output_size':n_steps_predict,"hidden_size":64,'n_layers':2,'batch_size':64,'lr':.001,'epochs':2}

    transformer_model = pytorchModel(model_type = "transformer",**kwargs)
    lstm_model = pytorchModel(model_type = "lstm",**kwargs)
    

    transformer_model.fit(xTrain,yTrain)
    lstm_model.fit(xTrain,yTrain)
    
    comparer.add_model('Transformer', transformer_model)
    comparer.add_model('LSTM', lstm_model)

    # Make predictions
    comparer.predict(xTest)

    # Evaluate models
    evaluation_results = comparer.evaluate(yTest)
    print("\nEvaluation Results:")
    print(evaluation_results)

    # Plot results
    comparer.plot_results(yTest.flatten(),max_size = n_steps_predict)




