"""
Resources:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1314585
http://www.blacklitterman.org/
http://www.andreisimonov.com/NES/Litterman_Nomura.pdf
https://corporate.morningstar.com/ib/documents/MethodologyDocuments/IBBAssociates/BlackLitterman.pdf
http://www.stat.berkeley.edu/~nolan/vigre/reports/Black-Litterman
http://www.quantandfinancial.com/2013/08/portfolio-optimization-ii-black.html
https://github.com/omartinsky/QuantAndFinancial/blob/master/black_litterman/black_litterman.ipynb
http://www.blacklitterman.org/code/hl_py.html
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=334304

Notes:
  - The (lambda * Sigma)^-1 * mu is the solution to the uncontstrained markowitz optimization problem
  - Common values of tau are 1, 0.01-0.05, 1/# return observations?

Todos:
 - Try computing lambda by using the return/variance of the S&P500 / some other market index
 - Try computing the information ratio and other risk profile characteristics mentioned in the Idzorek paper
 - Try other estimates of the covariance matrix
 - Add ability to specify confidence levels as mentioned in the Idzorek paper
 - Figure out why posterior covariance is so small
"""

import cvxopt as opt
import datetime as dt
import numpy as np
import pandas as pd
import pandas_datareader.data as data
import re
import requests
from cvxopt import blas, solvers
from pandas_datareader.yahoo.quotes import _yahoo_codes
from bs4 import BeautifulSoup

# setup
_yahoo_codes.update({'Market Cap': 'j1'})
np.random.seed(123)
solvers.options['show_progress'] = False

# Various functions


def posterior_return_100_one_view(prior_returns, risk_multiplier, returns_cov, view_link_vector, view):
    """
    """
    factor1 = risk_multiplier * returns_cov @ view_link_vector
    factor2 = np.linalg.inv(view_link_vector.T @ (risk_multiplier * returns_cov) @ view_link_vector)
    factor3 = view - view_link_vector.T @ prior_returns

    return prior_returns + (factor1 @ factor2 @ factor3)


def create_view_matrices(symbols, views, weighting_method=None,
    market_caps=None):
    """
    Weighting methods are market_cap, equal, and None 
     """
    Q = np.matrix(np.empty((len(views), 1)))
    P = np.matrix(np.empty((len(views), len(symbols))))

    for index, view in enumerate(views):
        Q[index, :] = [view['amount']]

        P_row = []
        weights = {}

        # precompute some values
        if weighting_method == 'market_cap':
           pos_caps = [market_caps[symbols.index(sym)] for sym in view['pos_portfolio']]
           if 'neg_portfolio' in view:
              neg_caps = [market_caps[symbols.index(sym)] for sym in view['neg_portfolio']]

        for sym in symbols:
            if sym in view['pos_portfolio']:
               if weighting_method == 'market_cap':
                  weight = market_caps[symbols.index(sym)]/sum(pos_caps)
               elif weighting_method == 'equal':
                   weight = 1./len(view['pos_portfolio'])
               else:
                   weight = 1

            elif 'neg_portfolio' in view and sym in view['neg_portfolio']:
                if weighting_method == 'market_cap':
                    weight = -1 * market_caps[symbols.index(sym)]/sum(neg_caps)
                elif weighting_method == 'equal':
                     weight = -1./len(view['neg_portfolio'])
                else:
                     weight = -1

            else:
              weight = 0

            P_row.append(weight)

        P[index, :] = P_row
               
    return Q, P

        
def compute_unconstrained_optimization_weights(cov, mean, market_return, rfr, market_var, l=None):
    """
    """
    # risk aversion 
    if l is None:
       l = (market_return - rfr) / market_var

    return np.dot(np.linalg.inv(l*cov), mean)


def optimal_weights(returns, cov, rfr, method='sharpe', quiet=False):
    n = len(returns)

    N = 1000
    mus = [10**(8 * t/N - 4) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(cov)
    pbar = opt.matrix(returns)

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -1*pbar, G, h, A, b)['x'] 
                  for mu in mus]

    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar.T, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x.T, S*x)) for x in portfolios]

    sharps = [(ret - rfr)/risk for ret, risk in zip(returns, risks)]

    best_sharp = np.argmax(sharps)

    if not quiet:
       print("Best sharpe ratio:")
       print(sharps[best_sharp])

       print("Best risk-reward parameter:")
       print(mus[best_sharp])
    
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(mus[best_sharp] * S), -1*pbar, G, h, A, b)['x']

    best_ret = blas.dot(pbar.T, wt)
    best_risk = np.sqrt(blas.dot(wt.T, S*wt))

    if not quiet:
       print("Best return, best risk, best sharpe ratio:")
       print(best_ret, best_risk, (best_ret - rfr)/best_risk)

    return np.asarray(wt), returns, risks


def compute_view_variances(risk_multiplier, view_link_matrix,
    returns_covariance):
    """
    """
    return risk_multiplier * np.dot(view_link_matrix,
    np.dot(returns_covariance, view_link_matrix.T))


def compute_posterior_covariance(risk_multiplier, return_covariance,
    view_link_matrix, view_variances):
    """
    """
    term1 = np.linalg.inv(risk_multiplier*return_covariance)
    term2 = np.dot(view_link_matrix.T, np.dot(np.linalg.inv(view_variances), view_link_matrix))

    return np.linalg.inv(term1 + term2)


def compute_posterior_returns(risk_multiplier, return_covariance,
    view_link_matrix, view_variances, excess_returns, views):
    """
    """
    term1 = compute_posterior_covariance(risk_multiplier, return_covariance,
            view_link_matrix, view_variances)


    subterm1 = np.dot(np.linalg.inv(risk_multiplier * return_covariance),
    excess_returns)
    subterm2 = np.dot(view_link_matrix.T, np.dot(np.linalg.inv(view_variances), views))

    term2 = subterm1 + subterm2

    return np.dot(term1, term2)


def compute_portfolio_return(security_returns, weights):
    """
    """
    return np.mean(np.dot(returns, weights))


def compute_portfolio_variance(sigma, weights):
    """
    """
    return np.dot(weights.T, np.dot(sigma, weights))


def compute_excess_returns(market_return, market_weights, market_var, cov_mat, rfr, l = None):
    """
    """
    if l is None:
       l = (market_return - rfr) / market_var

    return l * np.dot(cov_mat, market_weights)


def get_risk_free_rate(month=3):
    """
    """
    r = requests.get('http://data.treasury.gov/feed.svc/DailyTreasuryYieldCurveRateData?$filter=month(NEW_DATE)%20eq%2012%20and%20year(NEW_DATE)%20eq%202016')

    soup = BeautifulSoup(r.text, 'html5lib')
    rfr = float(soup.find_all('content')[-1].find('d:bc_{}month'.format(month)).get_text())/100
    
    return rfr


def compute_covariance_matrix(returns):
    """
    """
    return np.cov(returns)


def get_returns(start, end, symbols=['GOOG', 'AMZN'], rule='BM'):
    """
    """
    daily_closes = [data.DataReader(symbol, 'yahoo', start, end)['Close'] for symbol in portfolio]

    returns = [dc.resample(rule).apply(lambda x: x[-1]).pct_change() for dc in daily_closes]

    return pd.concat(returns, axis=1).dropna().values


def get_market_caps(symbols=['GOOG', 'AMZN'],
                                 method='yahoo', debug=False):

    if method == 'yahoo':
        if debug:
            print(data.get_quote_yahoo(symbols)['Market Cap'])

        w_m = [float(mcap[:-1]) for mcap in
           data.get_quote_yahoo(symbols)['Market Cap'].values]

        if debug:
            print(w_m)

    elif method == 'etfdb':
        r = requests.get("http://etfdb.com/compare/market-cap/")
        soup = BeautifulSoup(r.text, 'html5lib')
        rows = soup.find_all('tr')

        aums = {}
        for row in rows:
            columns = row.find_all('td')
            if len(columns) > 0:
                sym = columns[0].get_text()
                aum = columns[2].get_text()

                if sym in symbols:
                    print("Found symbol {}, adding...".format(sym))
                    aums[sym] = float(aum[1:].replace(',', ''))

        w_m = [aums[sym] for sym in portfolio]

    return w_m
    
if __name__ == '__main__':
    # Compute the black-litterman portfolio
    portfolio = ['VTI', 'VTV', 'VOE', 'VBR', 'VEA', 'VWO']

    market_caps = get_market_caps(portfolio, method='etfdb')

    # declare views here
    views = [
          {
            'pos_portfolio': [
                                  'VTI',
                                  'VTV',
                                  'VOE',
                                  'VBR'
                                  ],
            'amount': 0.05,
            'neg_portfolio': [
                                  'VEA',
                                  'VWO'
            ]
          }
    ]

    weighting_method = 'market_cap'

    view_percents, view_link_matrix = create_view_matrices(portfolio, views, weighting_method=weighting_method, market_caps=market_caps)

    print("View percents")
    print(view_percents)

    print("View link matrix")
    print(view_link_matrix)



    start = dt.date(2014, 1, 1)
    end = dt.date.today()

    period = '3BM'
    rfr = get_risk_free_rate(3)
    print("Risk free rate:")
    print(rfr)

    total = sum(market_caps)

    w_m = np.asmatrix([cap/total for cap in market_caps]).T
    print("Market weights:")
    print(w_m)

    returns = get_returns(start, end, symbols=portfolio, rule=period)
    returns_cov = compute_covariance_matrix(returns.T)

    risk_multiplier = 1/len(returns)


    print("Risk multiplier")
    print(risk_multiplier)

    print('Returns')
    print(returns)

    print('Returns covariance')
    print(returns_cov)

    view_variances = compute_view_variances(risk_multiplier, view_link_matrix,
    returns_cov)

    print('View variances')
    print(view_variances)

    mean_returns = np.asmatrix(np.mean(returns,axis=0)).T
    print('Mean returns')
    print(mean_returns)

    w_mpt, _, _ = optimal_weights(mean_returns, returns_cov, rfr, method='sharpe')

    print('Optimal weights using historical data')
    print(w_mpt)

    market_return = compute_portfolio_return(returns, w_m)
    market_variance = compute_portfolio_variance(returns_cov, w_m).item((0, 0))

    print("Market return")
    print(market_return)

    print("Market variance")
    print(market_variance)

    excess_returns = np.matrix(compute_excess_returns(market_return, w_m, market_variance,returns_cov, rfr))

    print("Excess returns over rfr using CAPM")
    print(excess_returns)

    print("Prior returns (excess + rfr)")
    prior_returns = excess_returns + rfr
    print(prior_returns)

    print("Prior covariance")
    prior_cov = risk_multiplier * returns_cov
    print(prior_cov)

    w_excess, _, _ = optimal_weights(prior_returns, prior_cov, rfr, method='sharpe')

    print("Optimal weights for prior returns")
    print(w_excess)

    posterior_returns = compute_posterior_returns(risk_multiplier, returns_cov,
        view_link_matrix, view_variances, prior_returns, view_percents)

    posterior_cov = compute_posterior_covariance(risk_multiplier, returns_cov, view_link_matrix, view_variances)

    print("Posterior returns")
    print(posterior_returns)

    print("Posterior covariance")
    print(posterior_cov)

    w_s, returns, risks = optimal_weights(posterior_returns, posterior_cov,
    rfr, method='sharpe')

    print("Optimal view-adjusted weights")
    print(w_s)

    # print(compute_unconstrained_optimization_weights(posterior_cov, posterior_returns - rfr, market_return, rfr, market_variance, l=None))

    """
    Idzorek's procedure for specifying 0-100% confidence levels based on section 3 of https://corporate.morningstar.com/ib/documents/MethodologyDocuments/IBBAssociates/BlackLitterman.pdf
    """
    omega = np.matrix(np.empty((len(views), len(views))))
    confidences = np.matrix([[0.7]])

    for index, (view, view_link_vector, confidence) in enumerate(zip(view_percents, view_link_matrix, confidences)):
        if view_link_vector.sum() > 0.1:
            view = view[0,0] - rfr

        returns_k_100 = posterior_return_100_one_view(prior_returns, risk_multiplier, returns_cov, view_link_vector.T, view)

        w_k_100, _, _ = optimal_weights(returns_k_100, returns_cov, rfr, method='sharpe', quiet=True)

        # w_k_100 = compute_unconstrained_optimization_weights(returns_cov, returns_k_100 - rfr, market_return, rfr, market_variance, l=None)

        D_k_100 = w_k_100 - w_m

        C_k = (view_link_vector != 0)*confidence[0,0]

        tilt_k = np.multiply(D_k_100, C_k.T)

        w_k_pcnt = w_m + tilt_k

        # Grid search the uncertainty that minimises the sum of squared errors
        # between w_k and w_k_pcnt
        N = 100
        min_i = 0
        min_er = float('Inf')
        for i in range(N+1):
            print(i)
            omega_k = np.matrix([[10**(8*i/N - 7)]])
            posterior_ret_k = compute_posterior_returns(risk_multiplier, returns_cov, view_link_vector, omega_k, prior_returns, view)

            w_k, _, _ = optimal_weights(posterior_ret_k, returns_cov, rfr, method='sharpe', quiet=True)

            # w_k = compute_unconstrained_optimization_weights(returns_cov, posterior_ret_k - rfr, market_return, rfr, market_variance, l=None)

            error = np.square((w_k_pcnt - w_k)).sum()

            # note: the w_k > 0 constraint is optional
            if error < min_er and (w_k > 0).all():
                min_er = error
                min_i = i

        omega[index, index] = 10**(8*min_i/N - 7)

    print("View uncertainties")
    print(omega)

    posterior_returns_idz = compute_posterior_returns(risk_multiplier, returns_cov, view_link_matrix, omega, prior_returns, view_percents)

    posterior_cov_idz = compute_posterior_covariance(risk_multiplier, returns_cov, view_link_matrix, omega)

    print("Posterior returns")
    print(posterior_returns_idz)

    print("Posterior covariance")
    print(posterior_cov_idz)

    w_s, returns, risks = optimal_weights(posterior_returns_idz, posterior_cov_idz, rfr, method='sharpe')

    print(w_s)



    prices = pd.read_csv('prices.csv', index_col=0)

    rfr = 0.0015

    print(prices)

    returns = prices.pct_change().dropna().values

    print(returns)

    returns_cov = compute_covariance_matrix(returns.T)

    mean_returns = np.asmatrix(np.mean(returns,axis=0)).T
    print(mean_returns)
    print(returns_cov)

    w_mpt, _, _ = optimal_weights(mean_returns, returns_cov, rfr, method='sharpe')

    print(w_mpt)

