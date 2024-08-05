import backtrader as bt
import yfinance as yf
import datetime

class moving_avg_crossover(bt.SignalStrategy):
    def __init__(self):
        self.short_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=50)
        self.long_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=200)
        
        self.signal_add(bt.SIGNAL_LONG, self.data.close > self.short_ma)
        self.signal_add(bt.SIGNAL_SHORT, self.data.close < self.long_ma)

    def next(self):
        if self.data.close[0] > self.short_ma[0] and not self.position:
            self.buy()
        elif self.data.close[0] < self.long_ma[0] and self.position:
            self.sell()

def run_backtest():
    cerebro = bt.Cerebro() 
    cerebro.addstrategy(moving_avg_crossover)  

    # historical data YF
    data = bt.feeds.PandasData(dataname=yf.download('AAPL', start='2022-01-01', end='2023-01-01'))

    cerebro.adddata(data) 
    cerebro.broker.set_cash(100000)  # initial cash
    cerebro.broker.setcommission(commission=0.001)  # commission @ 0.1%
    
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()  # Run the backtest
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    cerebro.plot()

if __name__ == '__main__':
    run_backtest()
