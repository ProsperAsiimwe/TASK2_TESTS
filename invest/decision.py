import json
import pandas as pd
import numpy as np
from invest.networks.invest_recommendation import investment_recommendation, create_invest_bn, learn_invest_cpts
from invest.networks.quality_evaluation import quality_network, create_quality_bn, learn_quality_cpts
from invest.networks.value_evaluation import value_network, create_value_bn, learn_value_cpts
from invest.preprocessing.simulation import simulate
from invest.store import Store
from invest.evaluation.validation import process_metrics, process_benchmark_metrics
import logging

companies_jcsev = json.load(open('data/jcsev.json'))['names']
companies_jgind = json.load(open('data/jgind.json'))['names']
companies = companies_jcsev + companies_jgind
companies_dict = {"JCSEV": companies_jcsev, "JGIND": companies_jgind}

def prepare_data_for_learning(df):
    required_columns = ['Price', 'PE/PEMarket', 'PE/PESector', 'PE/HistoricalPE', 'PE', 'ROEvsCOE', 'RelDE', 'CAGRvsInflation', 'ROE', 'ShareBeta']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logging.warning(f"Missing columns in dataframe: {missing_columns}")
        # Create placeholder columns with default values
        for col in missing_columns:
            df[col] = 'Medium'  # Use 'Medium' as a default value

    # Create the dataframes using the mapped column names
    value_df = df[['Price', 'PE/PEMarket', 'PE/PESector', 'PE/HistoricalPE', 'PE']].copy()
    quality_df = df[['Price', 'ROEvsCOE', 'RelDE', 'CAGRvsInflation', 'ROE']].copy()
    invest_df = df[['Price', 'PE', 'ROE', 'ShareBeta']].copy()

    # Rename the columns to match what the learning functions expect
    value_df.columns = ['FutureSharePerformance', 'PERelative_ShareMarket', 'PERelative_ShareSector', 
                        'ForwardPE_CurrentVsHistory', 'ValueRelativeToPrice']
    quality_df.columns = ['FutureSharePerformance', 'ROEvsCOE', 'RelDE', 'CAGRvsInflation', 'Quality']
    invest_df.columns = ['Performance', 'Value', 'Quality', 'Investable']

    # Discretize the data
    for df in [value_df, quality_df, invest_df]:
        for col in df.columns:
            if df[col].dtype != 'object':  # Only discretize numeric columns
                df[col] = pd.qcut(df[col], q=3, labels=['Low', 'Medium', 'High'])

    return {
        'value': value_df,
        'quality': quality_df,
        'invest': invest_df
    }

def learn_all_networks(data, method='MLE'):
    prepared_data = prepare_data_for_learning(data)
    value_bn = learn_value_cpts(create_value_bn(), prepared_data['value'], method)
    quality_bn = learn_quality_cpts(create_quality_bn(), prepared_data['quality'], method)
    invest_bn = learn_invest_cpts(create_invest_bn(), prepared_data['invest'], method)
    return {'value': value_bn, 'quality': quality_bn, 'invest': invest_bn}

def investment_portfolio(df, params, index_code, verbose=False, learned_bns=None):
    """
    Decides the shares for inclusion in an investment portfolio using INVEST
    Bayesian networks. Computes performance metrics for the IP and benchmark index.

    Parameters
    ----------
    df : pandas.DataFrame
        Fundamental and price data
    params : argparse.Namespace
        Command line arguments
    index_code: str,
        Johannesburg Stock Exchange sector index code
    verbose: bool, optional
        Print output to console
    learned_bns: dict, optional
        Dictionary of learned Bayesian networks

    Returns
    -------
    portfolio: dict
    """
    logging.info(f"Creating investment portfolio for {index_code}")
    
    if params.noise:
        df = simulate(df)
    
    prices_initial = {}
    prices_current = {}
    betas = {}
    investable_shares = {}

    store = Store(df, companies, companies_jcsev, companies_jgind,
                  params.margin_of_safety, params.beta, params.start, False)
    
    for year in range(params.start, params.end):
        year_data = store.df_main[store.df_main['Date'].dt.year == year]
        investable_shares[str(year)] = []
        prices_initial[str(year)] = []
        prices_current[str(year)] = []
        betas[str(year)] = []
        
        for company in companies_dict[index_code]:
            company_data = year_data[year_data['Name'] == company]
            if company_data.empty:
                continue

            if store.get_acceptable_stock(company):
                decision = investment_decision(store, company, params.extension, params.ablation,
                                               params.network, learned_bns)
                if decision == "Yes":
                    investable_shares[str(year)].append(company)
                    prices_initial[str(year)].append(company_data.iloc[0]['Price'])
                    prices_current[str(year)].append(company_data.iloc[-1]['Price'])
                    betas[str(year)].append(company_data.iloc[-1]["ShareBeta"])

    if verbose:
        print("\n{} {} - {}".format(index_code, params.start, params.end))
        print("-" * 50)
        print("\nInvestable Shares")
        for year in range(params.start, params.end):
            print(year, "IP." + index_code, len(investable_shares[str(year)]), investable_shares[str(year)])

    if all(len(investable_shares[str(year)]) == 0 for year in range(params.start, params.end)):
        print(f"Warning: No investable shares found for {index_code}")
        return {
            "ip": {
                "shares": investable_shares,
                "metrics": ([], 0, 0, 0, 0)  # empty annual returns, 0 for CR, AAR, TR, SR
            },
            "benchmark": {
                "metrics": process_benchmark_metrics(params.start, params.end, index_code, params.holding_period)
            }
        }

    ip_metrics = process_metrics(df, prices_initial, prices_current, betas, params.start, params.end, index_code)
    benchmark_metrics = process_benchmark_metrics(params.start, params.end, index_code, params.holding_period)

    portfolio = {
        "ip": {
            "shares": investable_shares,
            "metrics": ip_metrics
        },
        "benchmark": {
            "metrics": benchmark_metrics
        }
    }
    return portfolio

def investment_decision(store, company, extension=False, ablation=False, network='v', learned_bns=None):
    pe_relative_market = store.get_pe_relative_market(company)
    pe_relative_sector = store.get_pe_relative_sector(company)
    forward_pe = store.get_forward_pe(company)

    roe_vs_coe = store.get_roe_vs_coe(company)
    relative_debt_equity = store.get_relative_debt_equity(company)
    cagr_vs_inflation = store.get_cagr_vs_inflation(company)
    systematic_risk = store.get_systematic_risk(company)

    value_bn = learned_bns['value'] if learned_bns else None
    quality_bn = learned_bns['quality'] if learned_bns else None
    invest_bn = learned_bns['invest'] if learned_bns else None

    value_decision = value_network(pe_relative_market, pe_relative_sector, forward_pe, learned_bn=value_bn)
    quality_decision = quality_network(roe_vs_coe, relative_debt_equity, cagr_vs_inflation,
                                       systematic_risk, extension, learned_bn=quality_bn)
    
    print(f"Company: {company}")
    print(f"Value decision: {value_decision}")
    print(f"Quality decision: {quality_decision}")
    
    if ablation and network == 'v':
        decision = "Yes" if value_decision in ["Cheap", "FairValue"] else "No"
        print(f"Ablation (v) decision: {decision}")
        return decision
    if ablation and network == 'q':
        decision = "Yes" if quality_decision in ["High", "Medium"] else "No"
        print(f"Ablation (q) decision: {decision}")
        return decision
    
    final_decision = investment_recommendation(value_decision, quality_decision, learned_bn=invest_bn)
    print(f"Final decision: {final_decision}")
    
    if final_decision == "No":
        print("Reason: ", end="")
        if value_decision == "Expensive":
            print("Share is considered expensive.")
        elif quality_decision == "Low":
            print("Share quality is low.")
        else:
            print("Combination of value and quality not favorable for investment.")
    
    return final_decision

def walk_forward_validation(data, start_year, end_year, params, index_code, learning_method):
    results = []
    for year in range(start_year, end_year):
        train_data = data[data['Date'].dt.year < year]
        test_data = data[data['Date'].dt.year == year]
        
        # Create a Store object to calculate the derived columns
        store = Store(train_data, companies, companies_jcsev, companies_jgind,
                      params.margin_of_safety, params.beta, year, False)
        
        # Use the processed data from the Store object
        processed_train_data = store.df_main
        
        learned_bns = learn_all_networks(processed_train_data, method=learning_method)
        portfolio = investment_portfolio(test_data, params, index_code, learned_bns=learned_bns)
        results.append(portfolio)
    return results

