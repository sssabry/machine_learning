import pandas as pd
import quandl 
import numpy as np
aapl = quandl.get("WIKI/AAPL", start_date="2006-10-01", end_date="2012-01-01")

# initial lookback periods
short = 20 #days
long = 50 #days

signals = pd.DataFrame(index=aapl.index)
signals['signal'] = 0.0

# calc moving averages
signals['short_mavg'] = aapl['Close'].rolling(window=short, min_periods=1, center=False).mean()
signals['long_mavg'] = aapl['Close'].rolling(window=long, min_periods=1, center=False).mean()

# overwrite 'signal' with 1 if condition is true (short has crossed over)
signals['signal'][short:] = np.where(signals['short_mavg'][short:] > signals['long_mavg'][short:], 
                                     1.0,
                                     0.0)

# generate trading orders by taking the difference
signals['positions'] = signals['signal'].diff()

# note: where positions is 1.0, is where u transition into buying
#print(signals.head(short + 5))

initial_cash = 10000
positions = pd.DataFrame(index=aapl.index)
positions['AAPL'] = 0.0
positions['AAPL'] = 100 * signals['signal'] # buy if 1, dont if 0

print(positions.head(short+5))


portfolio = pd.DataFrame(index=aapl.index)
portfolio['positions'] = positions['AAPL'] # store shares bought/sold
portfolio['holdings'] = portfolio['positions'] * aapl['Adj. Close'] # calc market value

# cumulatively decrement initial cash to calc cash available at each
portfolio['cash'] = initial_cash - (positions['AAPL'] * aapl['Adj. Close']).cumsum() 

portfolio['total'] = portfolio['cash'] + portfolio['holdings']
portfolio['returns'] = portfolio['total'].pct_change()

print(portfolio.head(short+5))
