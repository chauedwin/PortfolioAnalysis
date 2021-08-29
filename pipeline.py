import yfinance as yf
import numpy as np
import scipy.stats as stats
import pandas as pd
from datetime import datetime

class Portfolio():
    
    '''
    Class for Portfolio Analysis
    
    An ETL pipeline to download, clean, and process data before computing portfolio weights. 
    
    Parameters:
    -------------------------
    tickers:    list
                a list of stock/asset tickers to be included in the portfolio
    **kwargs:   additional parameters utilized by the yfinance download function 
    
    Methods:
    -------------------------
    download()
        
        returns:    dataframe of historical OHLC price data for assets and S&P500  
        
    clean_data(df = stock_data, minmonths = 0)
        df:         dataframe of OHCL prices for assets and S&P500
        minmonths:  minimum cutoff for number of observations per asset 
        
        returns:    dataframe of the cleaned data 
        
    compute_returns(df = cleaned_stock_data)
        df:         dataframe of cleaned/validated asset and S&P500 data
        
        returns:    numpy array of monthly percent returns for each asset without S&P500
                    numpy array of monthly percent returns of S&P500
                    list of names of assets from df 
                    
    compute_weights(returns = array_returns, mktreturn = sp500_returns, labels = asset_names)
        returns:    numpy array of monthly returns, with each column representing a single asset
        mktreturn:  returns of the S&P500
        labels:     list of names of the assets included in returns
        
        returns:    a Series of weights of assets in the final portfolio
    '''
    
    def __init__(self, tickers, **kwargs):
        self.tickers = tickers
        self.value = kwargs.get('value', None)
        self.stockdata = None
        self.sim = None
        self.weights = None
        self.numstocks = None
        
        # initialize yfinance params
        self.start = kwargs.get('start', None)
        self.end = kwargs.get('end', None)
        self.period = kwargs.get('period', '1mo')
        self.interval = kwargs.get('interval', '1mo')
        self.group_by = kwargs.get('group_by', 'ticker')
        self.auto_adjust = kwargs.get('auto_adjust', True)
        self.prepost = kwargs.get('prepost', False)
        self.threads = kwargs.get('threads', True)
        self.proxy = kwargs.get('proxy', None)
        
    
    def download(self, **kwargs):  
        df = yf.download(tickers = ['^GSPC'] + self.tickers, start = self.start, end = self.end, period = self.period, interval = self.interval, group_by = self.group_by, auto_adjust = self.auto_adjust, prepost = self.prepost, threads = self.threads, proxy = self.proxy)
        
        # unstack dataframe for easier indexing/slicing
        self.stockdata = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
        
        
    def clean_data(self, minmonths = 0, **kwargs):     
        # filter out assets with less observations than minmonths
        #minmonths = kwargs.get('minmonths', 0)
        df = self.stockdata.groupby('Ticker').filter(lambda x: len(x) > minmonths)
        
        # data includes current date, remove to avoid skewed data
        recent = df.index[-1] - pd.DateOffset(day = 1)
        self.stockdata = df.loc[df.index <= recent].copy()
        
    def compute_weights(self, **kwargs):  
        start = kwargs.get('start', min(self.stockdata.index))
        end = kwargs.get('end', max(self.stockdata.index))
        returns, mktreturn, labels = self.compute_returns(start = start, end = end)
        self.sim = self.reg_params(returns, mktreturn, labels)
        sim_cutoff = self.cut(self.sim, mktreturn)
        z = (sim_cutoff['beta'] / sim_cutoff['eps']) * (sim_cutoff['excess'] - sim_cutoff['C'])
        self.weights = z.sort_values(ascending = False) / z.sum()
            
        
    def compute_returns(self, **kwargs):
        start = kwargs.get('start', min(self.stockdata.index))
        end = kwargs.get('end', max(self.stockdata.index))
        df = self.stockdata.loc[(self.stockdata.index >= start) & (self.stockdata.index <= end)]

        # pivot and drop columns with all NaNs, pandas ignores partial NaNs
        pivoted = df.pivot(columns = 'Ticker', values = 'Open').dropna(axis=1, how='all')
        
        # extract and drop market from data
        spprices = pivoted['^GSPC'].copy().to_numpy()
        pivoted = pivoted.drop(columns = ['^GSPC'])
        
        # convert to numpy for easier computing
        prices = pivoted.to_numpy()
        
        # compute percent returns by subtracting previous day from current and dividing by previous
        returnarr = (prices[1:, :] - prices[:(prices.shape[0] - 1), :]) / prices[:(prices.shape[0] - 1), :]
        spreturns = (spprices[1:] - spprices[:(len(spprices) - 1)]) / spprices[:(len(spprices) - 1)]
        
        return(returnarr, spreturns, pivoted.columns.values)
    
    
    def reg_params(self, returns, mktreturn, labels, **kwargs):
        # compute alphas and betas by regression a stock's mean return on the market mean return
        alphas = np.zeros(returns.shape[1])
        betas = np.zeros(returns.shape[1])
        unsyserr = np.zeros(returns.shape[1])
        for i in np.arange(returns.shape[1]):
            treturn = returns[:,i]
            tnonan = treturn[np.logical_not(np.isnan(treturn))]
            mktmatch = mktreturn[(len(mktreturn) - len(tnonan)):]
            
            betas[i], alphas[i], r, p, se = stats.linregress(mktmatch, tnonan)
            unsyserr[i] = np.sum((tnonan - alphas[i] - betas[i]*mktmatch)**2) / (len(mktmatch) - 2)
            
        params = pd.DataFrame(data = {'alpha': alphas, 'beta': betas, 'eps': unsyserr, 'rmean': returns.mean(axis=0)}, index = labels)
        
        return(params)
        
        
    def cut(self, sim_params, mktreturn, **kwargs):
        
        sim_params['excess'] = sim_params['rmean'] / sim_params['beta']
        sim_params = sim_params.sort_values(by=['excess'], ascending = False)
        sim_params = sim_params.loc[(sim_params['excess'] > 0) & (sim_params['beta'] > 0)]
        
        # compute C values and cutoff
        num = sim_params['rmean'] * sim_params['beta'] / sim_params['eps']
        den = sim_params['beta']**2 / sim_params['eps']
        sim_params['C'] = mktreturn.var() * num.cumsum() / (1 + mktreturn.var() * den.cumsum())
        
        return(sim_params.loc[sim_params['C'] < sim_params['excess']])
    
    
    def plot_value(self, ticker, **kwargs):
        plt.plot(self.stockdata.loc[self.stockdata['Ticker'] == ticker, 'Open'])
        plt.show()
        
        
    def compute_numstocks(self, **kwargs):
        today_date = datetime.today().strftime('%Y-%m-%d')
        latest_raw = yf.download(tickers = self.weights.index.values.tolist(), start = today_date, period = self.period, interval = self.interval, group_by = self.group_by, auto_adjust = self.auto_adjust, prepost = self.prepost, threads = self.threads, proxy = self.proxy)
        latest = latest_raw.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
        latest_prices = latest.set_index('Ticker', drop=True).loc[self.weights.index.values, 'Close']
        self.numstocks = self.value*self.weights/latest_prices
    
    
if __name__ == '__main__':
    ticks = pd.read_csv("nasdaq_tech.csv")['Symbol'].tolist()
    #ticks = pd.read_csv("constituents_csv.csv")['Symbol'].tolist()
    #ticks = ['AAPL', 'MSFT', 'TSLA']
    p = Portfolio(tickers = ticks, value = 20000, start = '2016-06-01')
    p.download()
    p.clean_data(minmonths = 36)
    p.compute_weights()
    print(p.weights)
    p.compute_numstocks()
    print(p.numstocks)
        