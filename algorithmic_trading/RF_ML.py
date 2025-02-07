import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def get_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    data['SMA_10'] = data['Close'].rolling(window=10).mean() # 10 day simple moving avg
    data['SMA_50'] = data['Close'].rolling(window=50).mean() # 50 day simple moving avg
    data['RSI'] = rsi_calc(data['Close'], 14) # RSI <3
    data['MACD'] = macd_calc(data['Close']) # moving avg convergence/divergence
    data['ATR'] = atr_calc(data)

    data['Price_Change'] = data['Close'].pct_change() 
    data.dropna(inplace=True)
    return data

def rsi_calc(series, period): # RSI using rolling means of gains
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd_calc(series, fast_period=12, slow_period=26, signal_period=9): # macd & signal using exponental fast/slow MAs
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd - signal

def atr_calc(data, period=14):
    data['H-L'] = data['High'] - data['Low']
    data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
    data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
    data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    return data['TR'].rolling(window=period).mean()

def train_random_forest(data):
    features = ['SMA_10', 'SMA_50', 'RSI', 'MACD', 'ATR']
    data['Target'] = np.where(data['Price_Change'].shift(-1) > 0, 1, 0)
    
    X = data[features]
    y = data['Target']
    # splitting data into training/testing sets:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    param_dist = {
        'n_estimators': randint(50, 500),
        'max_depth': randint(10, 50),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 10)
    }
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=100,
        cv=5,
        n_jobs=-1, # use all cores
        scoring='accuracy')
    
    random_search.fit(X_train, y_train)

    print(f"Best Parameters: {random_search.best_params_}")
    model = random_search.best_estimator_
    
    # Debugging prediction model:
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions)
    classification = classification_report(y_test, predictions)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Model Confusion Matrix: {confusion}")
    print(f"Model Classification Report: {classification}")


    # application to entire dataset
    data['Prediction'] = model.predict(X)
    return model, data

# backtesting !!
class RandomForestStrategy(bt.SignalStrategy):
    def __init__(self, model, data):
        self.datafeed = data
        self.model = model
        # additional thresholds for model confidence: 
        self.buy_signal_threshold = 0.55  
        self.sell_signal_threshold = 0.45 
        self.risk_per_trade = 0.02 # risk 2% of portfolio per trade

    # one hold at a time, buys if up & none held, sells if held and down
    def next(self):
        prediction_prob = self.datafeed.Prediction.iloc[0]
        size = int(self.broker.get_cash() * self.risk_per_trade / self.data.close[0])

        if prediction_prob >= self.buy_signal_threshold and not self.position:
            self.buy(size=size)
        elif prediction_prob <= self.sell_signal_threshold and self.position:
            self.sell(size=self.position.size)

def run_backtest(data):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(RandomForestStrategy, model=None, data=data)
    
    bt_data = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(bt_data)
    
    cerebro.broker.set_cash(100000)
    cerebro.broker.setcommission(commission=0.001)
    
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())
    # for local: cerebro.plot()

def run_alternative_backtest(dataset):
    for symbol in dataset:
        data = get_data(symbol, start=start_date, end=end_date)
        model, data = train_random_forest(data)
        run_backtest(data)


# Main Execution
symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-01-01'

data = get_data(symbol, start=start_date, end=end_date)
model, data = train_random_forest(data)
run_backtest(data)
