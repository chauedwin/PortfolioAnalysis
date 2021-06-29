import yfinance as yf
import numpy as np
import scipy.stats as stats
import pandas as pd

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
        
        # initialize yfinance params
        self.start = kwargs.get('start', None)
        self.end = kwargs.get('end', None)
        self.period = kwargs.get('period', '1mo')
        self.interval = kwargs.get('interval', '1mo')
        self.group_by = kwargs.get('group_by', 'ticker')
        self.auto_adjust = kwargs.get('auto_adjust', True)
        self.prepost = kwargs.get('prepost', 'False')
        self.threads = kwargs.get('threads', True)
        self.proxy = kwargs.get('proxy', None)
        
    
    def download(self, **kwargs):
        
        try:
            data = yf.download(tickers = ['^GSPC'] + self.tickers, start = self.start, end = self.end, period = self.period, interval = self.interval, group_by = self.group_by, auto_adjust = self.auto_adjust, prepost = self.prepost, threads = self.threads, proxy = self.proxy)
        
        except Error as e:
            print(e)
            
        return(data)
    
    
    def clean_data(self, df, **kwargs):
        minmonths = kwargs.get('minmonths', 0)
        df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
        
        # filter out assets with less observations than minmonths
        df = df.groupby('Ticker').filter(lambda x: len(x) > minmonths)
        df = df.sort_values(by=['Date', 'Ticker'])
        
        # data includes current date, remove to avoid skewed data
        recent = df.index[-1] - pd.DateOffset(day = 1)
        df = df.loc[df.index <= recent].copy()
       
        return(df)
     
    def compute_returns(self, df, **kwargs):
        # drop columns with all NaNs, pandas ignores partial NaNs
        pivoted = df.pivot(columns = 'Ticker', values = 'Open').dropna(axis=1, how='all')
        
        # extract and drop market from data
        spprices = pivoted['^GSPC'].to_numpy()
        pivoted = pivoted.drop(columns = ['^GSPC'])
        
        # convert to numpy for easier computing
        prices = pivoted.to_numpy()
        
        # compute percent returns by subtracting previous day from current and dividing by previous
        returnarr = (prices[1:, :] - prices[:(prices.shape[0] - 1), :]) / prices[:(prices.shape[0] - 1), :]
        spreturns = (spprices[1:] - spprices[:(len(spprices) - 1)]) / spprices[:(len(spprices) - 1)]
        
        return(returnarr, spreturns, pivoted.columns.values)
    
    def compute_weights(self, returns, mktreturn, labels, **kwargs):
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
            
        simdf = pd.DataFrame(data = {'alpha': alphas, 'beta': betas, 'eps': unsyserr, 'rmean': returns.mean(axis=0)}, index = labels)
        simdf['excess'] = simdf['rmean'] / simdf['beta']
        simdf = simdf.sort_values(by=['excess'], ascending = False)
        simdf = simdf.loc[(simdf['excess'] > 0) & (simdf['beta'] > 0)]
        
        # compute C values and cutoff
        num = simdf['rmean'] * simdf['beta'] / simdf['eps']
        den = simdf['beta']**2 / simdf['eps']
        simdf['C'] = mktreturn.var() * num.cumsum() / (1 + mktreturn.var() * den.cumsum())
        
        cutoff = simdf.loc[simdf['C'] < simdf['excess']]
        z = (cutoff['beta'] / cutoff['eps']) * (cutoff['excess'] - cutoff['C'])
        
        return(z.sort_values(ascending = False) / z.sum())
    
if __name__ == '__main__':
    ticks = pd.read_csv("nasdaq_screener.csv")['Symbol'].tolist()
    p = Portfolio(tickers = ticks, start = '2016-06-01')
    stockdata_raw = p.download()
    stockdata = p.clean_data(stockdata_raw, minmonths = 36)
    print(stockdata)
    stockreturns, marketreturn, names = p.compute_returns(stockdata)
    print(stockreturns, marketreturn, names)
    x = p.compute_weights(stockreturns, marketreturn, names)
    print(x)
        