# M5ForecastingCompetition-Submission
Top 4% in final results for accuracy, 9% for Uncertainty. Submission for M5 Forecasting Competition of Walmart Data. 
https://www.kaggle.com/c/m5-forecasting-accuracy
https://www.kaggle.com/c/m5-forecasting-uncertainty/

Accuray involved forecasting sales for over 30 000 time series, scored using weighted mean squared scaled error which is affected by: 
the cost of the product and a one step naive forecast benchmark

Submission was also created for the uncertainty competition - creating the 1%.5%.33%,50%,67%,95%,99% intervals for every prediction. 
Scored using scaled pinball loss for each quantile. 

From this competition I learned:
Uncertainty intervals are as equally important as good point predictions, extremely useful to organizations avoiding stockouts
Significantly less tools were available to make good quantile predictions, and only 909 submissions were made compared to 5558 for point forecasts (accuracy competition)
My submission was only in the top 9% in the uncertainty competition (ranking 74th) with room for improvement. 

From the discussions of top submissions I decided to investigate:
  Innovation State Space Model (based on Exponential Smoothing) 
  Quantile Gradient Boosting regression
  Seasonal ARIMA models

So far quantile gradient boosting regression has been the most promising on some backtests and a new project involving real business data from a local boutique. 

