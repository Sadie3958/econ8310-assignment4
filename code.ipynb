import pandas as pd
import pymc as pm
import numpy as np
import arviz as az

# Load the dataset given from Assignment 4 from GitHub so everything is reproducible
url = 'https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/cookie_cats.csv'
df = pd.read_csv(url)

# Split into control vs treatment groups
# reminder: gate_30 = original version, gate_40 = new version we are testing
group_30 = df[df['version'] == 'gate_30']
group_40 = df[df['version'] == 'gate_40']

def analyze_retention(column_name):
    print(f"\n--- Bayesian Analysis for {column_name} ---")
    
    # Pull out the actual retention data (0 = no return, 1 = returned)
    obs_30 = group_30[column_name].values
    obs_40 = group_40[column_name].values

    with pm.Model() as model:
        # Start with simple, neutral priors
        # Basically saying: before seeing data, all retention rates are equally possible
        p_30 = pm.Beta('p_30', alpha=1, beta=1)
        p_40 = pm.Beta('p_40', alpha=1, beta=1)

        # Model the observed data as Bernoulli (since retention is yes/no)
        retention_30 = pm.Bernoulli('retention_30', p=p_30, observed=obs_30)
        retention_40 = pm.Bernoulli('retention_40', p=p_40, observed=obs_40)

        # Measure the effect of the change (gate_40 - gate_30)
        # This tells us if the new version improved or hurt retention
        delta = pm.Deterministic('delta', p_40 - p_30)

        # Run MCMC to estimate the posterior distributions
        # Using enough samples to get stable results
        trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)

    # Check how often the new version performs worse (delta < 0)
    delta_samples = trace.posterior['delta'].values
    prob_worse = (delta_samples < 0).mean()
    
    print(f"Probability that gate_40 has LOWER retention than gate_30: {prob_worse * 100:.2f}%")
    
    # Print summary stats + credible intervals for interpretation to see extra info
    print(az.summary(trace, var_names=['p_30', 'p_40', 'delta']))
    
    return trace

# Run the model for 1-day retention
trace_1 = analyze_retention('retention_1')

# Run the model for 7-day retention
trace_7 = analyze_retention('retention_7')
