Random Forest Model deployment in Amazon Sagemaker

In this notebook, I show how I trained and deployed a Random Forest machine learning model using AWS SageMaker. 
This notebook was created and run in an Amazon Sagemaker notebook instance. 
This demonstration is from a project I worked on titled "Machine Learning Approach to Simulating CO2 Fluxes in Cropping Systems".
This notebook only deals with the training, testing, and deployment aspect of the machine learning pipeline.
Data cleaning and manipulation was done in a separate notebook, thus cleaned data was uploaded and used in this notebook.


Resources needed
A dataset: The dataset used in this project was the GRACEnet dataset.The main purpose of the GRACEnet database is to aggregate information from many studies so that methods for quantifying GHG emissions and other environmental impacts of cropped and grazed systems can be developed, and to provide scientific evidence for carbon trading programs that can help reduce GHG emissions

An algorithm: I used the Random Forest algorithm in scikit-learn provided by Amazon SageMaker to train the model using the GRACEnet dataset to predict the CO2 flux in the cropping systems.

Resources from Amazon SageMaker
A few resources needed for storing your data and running the code in Amazon SageMaker:

An Amazon Simple Storage Service (Amazon S3) bucket to store the training data and the model artifacts that Amazon SageMaker creates when it trains the model.

An Amazon SageMaker notebook instance to prepare and process data and to train and deploy a machine learning model.

A Jupyter notebook to use with the notebook instance to prepare your training data and train and deploy the model.

Detailed description of this work is available in our paper: Adjuik, T.A., Davis, S.C. Machine Learning Approach to Simulate Soil CO2 Fluxes under Cropping Systems. Agronomy 2022, 12, 197, doi: https://doi.org/10.3390/agronomy12010197
