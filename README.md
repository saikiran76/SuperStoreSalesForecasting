# SuperStore Sales-Forecasting
This repository contains a time series analysis and forecasting project for Superstore sales data. The goal of this project is to analyze historical sales data, build time series forecasting models, and make future sales predictions for the Furniture and Office Supplies categories. This README file provides an overview of the project, its structure, and key findings.

Project Overview:
In this project, we perform time series analysis and forecasting on sales data from the Superstore. Specifically, we focus on the Furniture and Office Supplies categories to understand their sales patterns and make predictions for future sales.

Data Preprocessing:
We start by loading the Superstore sales data, filtering it for the Furniture and Office Supplies categories, and performing necessary data preprocessing steps. These steps include:

Removing unnecessary columns from the dataset.
Sorting the data by order date.
Handling missing values.
Aggregating sales data by date.
Time Series Analysis
We conduct a thorough time series analysis on the Furniture category. Here are some key insights:

The data spans four years, from [start date] to [end date].
There is a clear seasonality pattern, with sales being low at the beginning of the year and high at the end.
There is an upward trend in sales within each year, with a few low months in the middle of the year.
Time Series Forecasting
To make future sales predictions, we employ the SARIMA (Seasonal Autoregressive Integrated Moving Average) model. We go through the parameter selection process to determine the best model parameters. After selecting the optimal parameters, we train the SARIMA model on historical data and use it to make forecasts.

Model Evaluation:
We evaluate the accuracy of our forecasts using metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE). These metrics help us assess the quality of our model's predictions.

Visualization:
We visualize the historical sales data, forecasts, and model diagnostics to better understand the results. This includes plotting time series data, decomposition of the time series, and comparing observed vs. predicted sales.

Time Series Comparison:
In addition to forecasting for each category separately, we compare the sales patterns of Furniture and Office Supplies. We identify when Office Supplies sales first surpassed Furniture sales, providing insights into the relative performance of these two categories.

Forecasting with Prophet:
We use Facebook Prophet, a forecasting tool, to build models for both Furniture and Office Supplies categories. Prophet is known for its ability to handle time series data with multiple seasonalities and holidays. We make forecasts for the future and visualize the results.

Visualizing Trends and Patterns:
We visualize the trends and patterns in sales data for both categories using Prophet. This includes examining yearly, weekly, and daily trends to gain insights into how sales vary over time.

Conclusion:
This project provides a comprehensive analysis of Superstore sales data for the Furniture and Office Supplies categories. It demonstrates the use of time series forecasting techniques, including SARIMA and Prophet, to make accurate predictions and gain insights into sales trends. The results can help stakeholders make informed decisions regarding inventory management, marketing strategies, and resource allocation for these product categories.

For more details, code, and visualizations, please refer to the project files in this repository. Checkout the Jupyter notebook :)
