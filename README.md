# Stock Trading Signal Detection

by Andy Karnik


## Problem Statement

The stock market -- closely followed, not always understood.  Long positions, short sells.  Rate of return, volatility.  Investments made by 401k's, pension funds, value investors and day traders.  It surrounds us and binds us, as Yoda would say (if he were an investor).

The stock market is complicated, as there are many players in the market with varying trading strategies, all with the goal of investing their money and producing a positive rate of return.  However, since the stock market is characterized by short-term price swings, it is difficult to know when to buy, when to sell, and when to hold.  And even still, profit is not always achieved.




## Solution Statement

To address this problem, we seek to implement a stock trading model based on research performed by Pei-Chann Chang and Jheng-Long Wu (see diagram below).

![](doc/img/model_framework.png)




## Historical Stock Data

We will use historical data from S&P 500 stocks, which spans five years.  The data includes the following fields:

- Date
- Opening price
- Intraday high
- Intraday low
- Closing price
- Volume
- Ticker




## Trading Signal Generation

Trading signals (indicators of whether to buy, sell, or hold stock), are generated using piecewise linear representation (also known as PLR, see diagrams below).  By varying the segmentation threshold value, we can determine the number of peaks and valleys in PLR that maximize the profit for a given stock.

These trading signals will be used to train our machine learning model, which will be used to predict future trading signals.

#### Piecewise Linear Representation

![](doc/img/plr.png)

#### Trading Signal

![](doc/img/trading_signal_2.png)






## Critical Feature Engineering and Extraction

In order to engineer critical features, the technical calculations below are performed on the historical stock data.  These features will be used to train our machine learning model.

- Moving Average
- Exponential Moving Average
- Momentum
- Rate of Change
- Average True Range
- Bollinger Bands
- Pivot Points, Supports and Resistances
- Stochastic oscillator %K
- Stochastic oscillator %D
- Trix
- Average Directional Movement Index
- MACD, MACD Signal and MACD difference
- Mass Index
- KST Oscillator
- Relative Strength Index
- True Strength Index
- Accumulation/Distribution
- Chaikin Oscillator
- Money Flow Index and Ratio
- On-balance Volume
- Force Index
- Coppock Curve
- Standard Deviation






## Modeling

In order to model our data, we will create a gridsearch pipeline that consists of the following steps to determine which parameters yield the best R^2 scores:

- Principal component analysis (PCA), to determine the best number of principal components
- Kernel PCA (KPCA), to determine the best kernel and polynomial degrees (only applicable to the polynomial kernel)
- Support Vector Regression (SVR), to determine the best kernel, polynomial degrees, penalty parameter C, and epsilon hyperparameter

Once found, the best model will be used to predict future trading signals.

