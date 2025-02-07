## Intro:
This is a running repository in which I'm pushing various algorithmic trading strategies I've explored. 
My main focus has been to analyze different standard strategies, and their potential integrations with Machine Learning models like Random Forest, 
and the impact this then has on their effectivity and accuracy.

This is a continuous project, mainly documenting my own exploriations 

(Note: not yet fully updated, I've just started pushing files from various explorations I've done in the past!)

## Setup
---

To get started run:
'''
    pip install -r requirements.txt
'''

## Strategy Notes:
each have dynamic position sizing & very basid risk managemtn

### momentum or trend trading:
_aka you believe this stock will continue in its current pattern_
- *Moving avg crossover* - when price moves from one side of the running avg to another (crosses over) it represents a change in momentum/trend, this is the point where we decide to enter or exit the market
    ** most basic 'hello world' type of quant strat **
- *Dual moving avg crossover* - when short term avg crosses a long term avg, signal is used to identify that the trend is shifting towards the short term avg 
    - buy signal generated when short term avg > long term avg 
    - sell signal when short term < longterm avg
- *Turtle trading* - buy futures @ 20 day high & sell on a 20 day low

### reversion strats:
_aka you believe the current quantity pattern will eventually reverse_
- *mean reversion strat* [you believe it will go  back to the mean, so you can exploit this when it deviates]
- *pairs trading strat* if theres a proven high correlation between 2 stocks, you can signal trade events if 1/2 moves out of correlation with the other
    correlation decreased 
    -> higher priced stock in short (sold) cuz it will return to mean, eventually so selling now gets u profit
    -> lower priced stock in long (bought) cuz it will return to normal which increases the price

### extras:
- *forecasting strat* [tries to predict future direction based on historical factors]
- *high frequency trading strat* [exploit sub-millisecond using FPGAs <333]

## Next to implement:
_Exponential moving average: (EMA)_
    based on short term & long term moving average crossovers w/ additional confirmation from MACD/RSI indexes -- has dynamic position sizing & (very) basic risk management

_Bollinger bands strategy: (BB)_
    generates buy/sell signals when price moves outside bands w/ MACD/RISI index confirmation


## Notes
