#!/usr/bin/env python
# coding: utf-8

# In[299]:


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


# In[300]:


tickers_10 = ['IDCC','NVDA','ISRG','HIVE','DOX','LIN','SBUX','COST','UFPI','ACP']
benchmark_ticker = 'SPY' 


# In[301]:


stock_data = {ticker: yf.download(ticker, start='2014-01-01', end='2024-01-01')['Adj Close'] for ticker in tickers_10}
benchmark_data = yf.download(benchmark_ticker, start='2014-01-01', end='2024-01-01')['Adj Close']


# In[302]:


df = pd.DataFrame(stock_data)
df['Benchmark'] = benchmark_data
df


# In[303]:


returns_df = df.pct_change().dropna()


# In[304]:


plt.figure(figsize=(20, 12))
for ticker in tickers_10:
    df[ticker].plot(label=ticker)
plt.title('Adjusted Close Prices', fontsize=18)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Adjusted Close Price USD ($)', fontsize=20)
plt.legend(loc='upper left')
plt.show()


# In[305]:


plt.figure(figsize=(14, 7))
for ticker in tickers_10:
    returns_df[ticker].plot(label=ticker)
plt.title('Daily Returns')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
plt.show()


# In[306]:


plt.figure(figsize=(12, 14))


color_list = ['blue', 'green', 'red', 'purple', 'orange',  
               'teal', 'darkgoldenrod', 'darkcyan', 'magenta', 'gold']  

for i, ticker in enumerate(tickers_10, 1):
    plt.subplot(5, 2, i)
    returns_df[ticker].hist(bins=50, alpha=0.7, color=color_list[i % len(color_list)]) 
    plt.title(f'{ticker} Daily Returns')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[419]:


plt.figure(figsize=(5, 10))
scatter_matrix = sns.pairplot(returns_df)
plt.suptitle('Scatter Matrix of Daily Returns', y=1.02)
plt.show()


# In[308]:


from BorutaShap import BorutaShap
import lightgbm as lgb


# In[309]:


X = df.drop(columns=['Benchmark']) 
y = df['Benchmark']


model = lgb.LGBMRegressor()
feat_selector = BorutaShap(model=model, importance_measure='shap', classification=False)

feat_selector.fit(X=X, y=y, n_trials=100, sample=False, verbose=True)

important_features = feat_selector.accepted

print("Selected Features:", important_features)

feat_selector.plot(which_features='all',
                      X_size=12, figsize=(20,8),
                      y_scale='log')


# # Momentum Strategy

# In[310]:


def momentum_strategy(returns, lookback_period = 2516, top_n=3):
    #print(top_n)
    filtered_df_date = df.loc[df.index[-lookback_period]:]
    benchmark_returns = filtered_df_date['Benchmark'].pct_change().dropna()  
    asset_returns = filtered_df_date.drop('Benchmark', axis=1).pct_change().dropna()
    
    relative_strength = asset_returns.apply(lambda x: (1 + x).cumprod() / (1 + benchmark_returns).cumprod(), axis=0)
    relative_strength_rank = relative_strength.iloc[-1].rank(ascending=False) 

    top_assets = relative_strength_rank[relative_strength_rank <= top_n].index.tolist()
    return top_assets

top_assets = momentum_strategy(returns_df)


# In[311]:


lookback_period=90
top_n=3
Volatile_top_assets=momentum_strategy(returns_df,lookback_period,top_n)
print( "top_assets",Volatile_top_assets)


# In[312]:


plt.figure(figsize=(10, 6))
for asset in top_assets:

    plt.plot(returns_df[asset][-lookback_period:], label=asset)

plt.title('Top {} Assets Based on Momentum Strategy'.format(top_n))
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.show()


# In[313]:


filtered_df = df[top_assets]


# In[314]:


top_n=3

plt.figure(figsize=(12, 6))
for ticker in top_assets:
    plt.plot(filtered_df[ticker], label=ticker)


equal_weights = np.ones(len(top_assets)) / len(top_assets)
portfolio_returns = (returns_df[top_assets] * equal_weights).sum(axis=1)
plt.plot((1 + portfolio_returns).cumprod(), label='Equal-Weighted Portfolio', linestyle='dashed', linewidth=2, color='black')

plt.xlabel("Date", fontsize=12)
plt.ylabel("Adjusted Close Price", fontsize=12)
plt.title(f"Price Chart of Top {top_n} Assets Based on Relative Strength", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# # Backtesting Strategy

# In[315]:


mu = expected_returns.mean_historical_return(filtered_df)
S = risk_models.sample_cov(filtered_df)
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)


# In[316]:


def backtest_strategy(returns_df, selected_assets):
    portfolio_returns = returns_df[selected_assets].mean(axis=1)
    cumulative_returns = (portfolio_returns + 1).cumprod() - 1
    return cumulative_returns

cumulative_returns = backtest_strategy(returns_df, top_assets)


# In[318]:


cumulative_returns.plot(figsize=(10, 6), title='Cumulative Returns of Momentum Strategy')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Cumulative Returns", fontsize=12)
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# #  Optimization

# In[319]:


def optimize_strategy(returns_df, parameter_grid={'lookback_period': [365*2, 365*5, 365*6], 'top_n': [9,6,3,7,2,4,1,5,10]}):
    best_cumulative_returns = None
    best_parameters = None
    best_assets=None
    
    for lookback_period in parameter_grid['lookback_period']:
        for top_n in parameter_grid['top_n']: 
            #print(str(lookback_period)+"--"+str(top_n))
            selected_assets = momentum_strategy(returns_df, lookback_period=lookback_period, top_n=top_n)
            cumulative_returns = backtest_strategy(returns_df, selected_assets)
            #print (cumulative_returns)
            if best_cumulative_returns is None or cumulative_returns.iloc[-1] > best_cumulative_returns.iloc[-1]:
                best_cumulative_returns = cumulative_returns
                best_parameters = {'lookback_period': lookback_period, 'top_n': top_n}
                best_assets=selected_assets
    
    return best_cumulative_returns, best_parameters,best_assets

best_returns,best_assets, best_params = optimize_strategy(returns_df)
print("Best best_assets:", best_params)
print("Best  parameters:", best_assets )
best_returns.plot(figsize=(10, 6), title='Optimized Cumulative Returns of Momentum Strategy Used')
plt.grid(True)


# In[ ]:





# In[ ]:





# # Covid-19 Data and their comparison of recovery time considering time and cost factors

# In[320]:


def calculate_drawdown_recovery(stock_data, tickers, start_date, end_date):
    results = {}
    for ticker in tickers:
        data = stock_data[ticker].loc[start_date:end_date]
        drawdown = (data - data.cummax()) / data.cummax()
        max_drawdown_date = drawdown.idxmin()
        recovery_start = max_drawdown_date
        recovery_end = data.loc[max_drawdown_date:].idxmax()
        recovery_time = (recovery_end - recovery_start).days
        results[ticker] = {
            'Max Drawdown Date': max_drawdown_date,
            'Recovery Start': recovery_start,
            'Recovery End': recovery_end,
            'Recovery Time (days)': recovery_time
        }
    return results


# In[321]:


impact_start_date = '2020-02-01'
impact_end_date = '2020-03-31'

recovery_start_date = '2020-04-01'
recovery_end_date = '2021-12-31'


# In[322]:


selected_features_df = df[important_features + ['Benchmark']]


# In[323]:


recovery_results = calculate_drawdown_recovery(selected_features_df, tickers_10, impact_start_date, recovery_end_date)


# In[324]:


recovery_df = pd.DataFrame(recovery_results).T
print(recovery_df)


# In[325]:


plt.figure(figsize=(10, 6))
for ticker in tickers_10:
    data = selected_features_df[ticker]
    drawdown = (data - data.cummax()) / data.cummax()
    plt.plot(data.index, drawdown, label=ticker)

plt.title('Drawdown During COVID-19 Impact and Recovery')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.legend()
plt.show()


# In[326]:


plt.figure(figsize=(14, 7))
for ticker in tickers_10:
    data = selected_features_df[ticker]
    plt.plot(data.index, data, label=ticker)

    max_dd_date = recovery_df.loc[ticker, 'Max Drawdown Date']
    recovery_start = recovery_df.loc[ticker, 'Recovery Start']
    recovery_end = recovery_df.loc[ticker, 'Recovery End']

    plt.axvline(x=max_dd_date, color='r', linestyle='--', alpha=0.7)
    plt.axvline(x=recovery_start, color='g', linestyle='--', alpha=0.7)
    plt.axvline(x=recovery_end, color='b', linestyle='--', alpha=0.7)

    plt.text(max_dd_date, data[max_dd_date], f'Max DD\n{max_dd_date.date()}', color='r', fontsize=8)
    plt.text(recovery_start, data[recovery_start], f'Recovery Start\n{recovery_start.date()}', color='g', fontsize=8)
    plt.text(recovery_end, data[recovery_end], f'Recovery End\n{recovery_end.date()}', color='b', fontsize=8)

plt.title('Stock Prices During COVID-19 Impact and Recovery')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()


# In[327]:


for ticker in tickers_10:
    data = selected_features_df[ticker]
    drawdown = (data - data.cummax()) / data.cummax()

    plt.figure(figsize=(10, 7))


    plt.subplot(2, 1, 1)
    plt.plot(data.index, drawdown, label='Drawdown')
    max_dd_date = recovery_df.loc[ticker, 'Max Drawdown Date']
    recovery_start = recovery_df.loc[ticker, 'Recovery Start']
    recovery_end = recovery_df.loc[ticker, 'Recovery End']
    plt.axvline(x=max_dd_date, color='r', linestyle='--', alpha=0.7)
    plt.axvline(x=recovery_start, color='g', linestyle='--', alpha=0.7)
    plt.axvline(x=recovery_end, color='b', linestyle='--', alpha=0.7)
    plt.text(max_dd_date, drawdown[max_dd_date], f'Max DD\n{max_dd_date.date()}', color='r', fontsize=8)
    plt.text(recovery_start, drawdown[recovery_start], f'Recovery Start\n{recovery_start.date()}', color='g', fontsize=8)
    plt.text(recovery_end, drawdown[recovery_end], f'Recovery End\n{recovery_end.date()}', color='b', fontsize=8)
    plt.title(f'{ticker} Drawdown During COVID-19 Impact and Recovery')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.show()


# In[328]:


for ticker in tickers_10:
    data = selected_features_df[ticker]
    drawdown = (data - data.cummax()) / data.cummax()

    plt.figure(figsize=(10, 7))

    plt.subplot(2, 1, 2)
    plt.plot(data.index, data, label='Stock Price')
    plt.axvline(x=max_dd_date, color='r', linestyle='--', alpha=0.7)
    plt.axvline(x=recovery_start, color='g', linestyle='--', alpha=0.7)
    plt.axvline(x=recovery_end, color='b', linestyle='--', alpha=0.7)
    plt.text(max_dd_date, data[max_dd_date], f'Max DD\n{max_dd_date.date()}', color='r', fontsize=8)
    plt.text(recovery_start, data[recovery_start], f'Recovery Start\n{recovery_start.date()}', color='g', fontsize=8)
    plt.text(recovery_end, data[recovery_end], f'Recovery End\n{recovery_end.date()}', color='b', fontsize=8)
    plt.title(f'{ticker} Stock Prices During COVID-19 Impact and Recovery')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    plt.tight_layout()
    plt.show()


# In[ ]:





# In[ ]:





# # Portfolio

# In[396]:


import riskfolio as rp


# # Mean Risk Portfolios

# # 1. Calculate the portfolio that maximizes Mean/CVaR ratio.

# In[402]:


import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.4%}'.format


start = '2014-01-01'
end = '2024-01-01'


# In[404]:


assets = ['IDCC','NVDA','ISRG','HIVE','DOX','LIN','SBUX','COST','UFPI','ACP']
assets.sort()


# In[405]:


data = yf.download(assets, start = start, end = end)
data = data.loc[:,('Adj Close', slice(None))]
data.columns = assets


# In[406]:


Y = data[assets].pct_change().dropna()

display(Y.head())


# In[408]:


import riskfolio as rp


port = rp.Portfolio(returns=Y)

method_mu='hist' 
method_cov='hist' 

port.assets_stats(method_mu=method_mu,
                  method_cov=method_cov,
                  d=0.94)

model = 'Classic' 

rm = 'MV'
obj = 'Sharpe'
hist = True 
rf = 0
l = 0

w = port.optimization(model=model,
                      rm=rm,
                      obj=obj,
                      rf=rf,
                      l=l,
                      hist=hist)

display(w)


# # 2 Plot portfolio composition

# In[409]:


ax = rp.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
                 height=6, width=10, ax=None)


# # 3 Calculate efficient frontier
# 

# In[410]:


points = 50

frontier = port.efficient_frontier(model=model,
                                   rm=rm,
                                   points=points,
                                   rf=rf,
                                   hist=hist)

display(frontier.T.head())


# In[415]:


label = 'Max Risk Adjusted Return Portfolio' 
mu = port.mu
cov = port.cov
returns = port.returns 

ax = rp.plot_frontier(w_frontier=frontier,mu=mu,cov=cov,returns=returns,rm=rm,rf=rf,alpha=0.05, cmap='viridis',w=w,
                      label=label,marker='*',s=16,c='r',height=6,width=10,ax=None)


# In[416]:


ax = rp.plot_frontier_area(w_frontier=frontier,
                           cmap="tab20",
                           n_colors=20,
                           height=6,
                           width=10,
                           ax=None)


# In[ ]:





# In[ ]:




