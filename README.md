# M5ForecastingCompetition-Submission
Top 4% in final results for accuracy, 9% for Uncertainty. Submission for M5 Forecasting Competition of Walmart Data.   
https://www.kaggle.com/c/m5-forecasting-accuracy  
https://www.kaggle.com/c/m5-forecasting-uncertainty/    

Data provided by Walmart of the daily sales and price of over 30 000 products over 5 years. Two different sets of predictions were required:    

**Accuracy competition** - point forecasts - scored using weighted mean squared scaled error  
**Uncertainty competition** - interval forecasts - creating the 1%.5%.33%,50%,67%,95%,99% intervals for every prediction, scored using scaled pinball loss for each quantile  

### Files:  
*submission_joshli_m5acccuracy.ipynb* is the full submission notebook, originally made on kaggle.com using the data [here](https://www.kaggle.com/c/m5-forecasting-accuracy/data)

*Uncertainty Stream* contains all the files needed to produce the uncertainty interval listed above (including the point forecasts downloaded from submission_joshli_m5acccuracy.ipynb)


### From this competition I learned:  
- Uncertainty intervals are as equally important as good point predictions, extremely useful to organizations avoiding stockouts  
- Significantly less tools were available to make good quantile predictions, and only 909 submissions were made compared to 5558 for point forecasts (accuracy competition)  My submission was ranking 74th (top 9%) with room significant for improvement. 

### From the discussions of top submissions I decided to investigate:  
  Innovation State Space Model (based on Exponential Smoothing)   
  Quantile Gradient Boosting regression  
  Seasonal ARIMA models  

So far quantile gradient boosting regression has been the most promising on some backtests and a new project involving real business data from a local boutique. 

