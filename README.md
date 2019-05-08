# Automated Stock Trading Simulation

## What is this place?
I have created this project as part of my Bachelor Thesis "Machine Learning for Automated Stock Trading".
It aims at evaluating different approaches of predicting financial time series data.
More precisely, this project comprises a framework for statistically comparing results of automated stock trading simulations using different predictors.
Among these predictors are the two widely used classification algorithms k-Nearest Neighbors and Random Forest together with extensions that for the one part enable them to base their decisions on some confidence score, and for the other part enable them to handle concept drift.
For more details on these concepts and the approach implemented here please consider my Bachelor Thesis, which is included in this repository as well.

## Setup
Before using the simulation the required dependencies have to be installed.
This can be done using the following command: "python src/setup.py"
The setup script will install a few dependencies via pip, including the contained library for fast nearest-neighbors search that will be used by SAM k-NN.

## Running a Simulation
Running src/thesis.py will download price data, run a trading simulation, persist the prediction results and display the results of the comparative analysis.
It will run the simulation on data between June, 3rd, 2002 and May, 31st, 2017 using the freely available S&P 500 stock index data from Quandl.
This project comes with several implementations of the AbstractPredictor interface.
These implementations use different algorithms for prediction:
* k-NN 
* k-NN (confidence-aware)
* SAM k-NN (drift-handling)
* Random Forest
* Random Forest (confidence-aware)
* Adaptive Random Forest (drift-handling)
 
They also include a few baseline strategies for comparison:
* Buy and Hold
* Constant Prediction
* Majority Prediction

SP500Data.py can be executed directly to get some insights about the data.
TechnicalIndicators.py can be executed directly to get a feel for the used features.
stats.py can be executed directly for playing around with the Wilcoxon's Signed Rank Test.
