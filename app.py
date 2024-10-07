import argparse
import time
import art
import numpy as np
import pandas as pd
from invest.decision import investment_portfolio, walk_forward_validation
from invest.preprocessing.dataloader import load_data
import pyAgrum as gum

VERSION = 1.2

def main():
    start = time.time()
    df = load_data()
    df['Date'] = pd.to_datetime(df['Date'])
    
    results = {}
    methods = ['Original', 'MLE', 'MDL', 'BDeu']
    
    for method in methods:
        if method == 'Original':
            jgind_portfolio = investment_portfolio(df[df['Name'].isin(companies_jgind)], args, "JGIND", True)
            jcsev_portfolio = investment_portfolio(df[df['Name'].isin(companies_jcsev)], args, "JCSEV", True)
        else:
            jgind_portfolio = walk_forward_validation(df[df['Name'].isin(companies_jgind)], args.start, args.end, args, "JGIND", method)
            jcsev_portfolio = walk_forward_validation(df[df['Name'].isin(companies_jcsev)], args.start, args.end, args, "JCSEV", method)
        
        results[method] = {
            'JGIND': jgind_portfolio,
            'JCSEV': jcsev_portfolio
        }
    
    end = time.time()

    print_results(results)

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nExperiment Time: ""{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

def print_results(results):
    sectors = ['JGIND', 'JCSEV']
    methods = ['Original', 'MLE', 'MDL', 'BDeu']
    metrics = ['CR', 'AAR', 'TR', 'SR']

    for sector in sectors:
        print(f"\n{sector} Sector Results (2015-2018)")
        print("-" * 50)
        print(f"{'Method':<10} {'CR (%)':<10} {'AAR (%)':<10} {'TR':<10} {'SR':<10}")
        print("-" * 50)
        
        for method in methods:
            if method == 'Original':
                cr = results[method][sector]['ip']['metrics'][1] * 100
                aar = results[method][sector]['ip']['metrics'][2] * 100
                tr = results[method][sector]['ip']['metrics'][3]
                sr = results[method][sector]['ip']['metrics'][4]
            else:
                cr = np.mean([p['ip']['metrics'][1] for p in results[method][sector]]) * 100
                aar = np.mean([p['ip']['metrics'][2] for p in results[method][sector]]) * 100
                tr = np.mean([p['ip']['metrics'][3] for p in results[method][sector]])
                sr = np.mean([p['ip']['metrics'][4] for p in results[method][sector]])
            
            print(f"{method:<10} {cr:<10.2f} {aar:<10.2f} {tr:<10.2f} {sr:<10.2f}")
        
        print("-" * 50)
        benchmark_cr = results['Original'][sector]['benchmark']['metrics'][1] * 100
        benchmark_aar = results['Original'][sector]['benchmark']['metrics'][2] * 100
        benchmark_tr = results['Original'][sector]['benchmark']['metrics'][3]
        benchmark_sr = results['Original'][sector]['benchmark']['metrics'][4]
        print(f"{'Benchmark':<10} {benchmark_cr:<10.2f} {benchmark_aar:<10.2f} {benchmark_tr:<10.2f} {benchmark_sr:<10.2f}")
        print("\n")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Intelligent system for automated share evaluation',
                                     epilog='Version 1.2')
    parser.add_argument("--start", type=int, default=2015)
    parser.add_argument("--end", type=int, default=2018)
    parser.add_argument("--margin_of_safety", type=float, default=1.4)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--extension", type=str2bool, default=False)
    parser.add_argument("--noise", type=str2bool, default=False)
    parser.add_argument("--ablation", type=str2bool, default=False)
    parser.add_argument("--network", type=str, default='v')
    parser.add_argument("--holding_period", type=int, default=-1)
    args = parser.parse_args()

    # Load company lists
    import json
    companies_jcsev = json.load(open('data/jcsev.json'))['names']
    companies_jgind = json.load(open('data/jgind.json'))['names']

    print(art.text2art("INVEST"))
    print("Insaaf Dhansay & Kialan Pillay")
    print("© University of Cape Town 2021")
    print("Version {}".format(VERSION))
    print(gum.__version__)
    print("=" * 50)

    main()