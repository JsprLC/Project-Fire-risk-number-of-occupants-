import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination
from scipy import integrate, stats
from itertools import product
import matplotlib.pyplot as plt


# ============================================================================
# Configuration
# ============================================================================
BUILDING_TYPE = 1  # 0: Low Risk, 1: Middle Risk, 2: High Risk
N_BINS = 5  # Number of bins for continuous variables

# Building parameters
idx_people = 44
idx_volume = 1865.1

# Risk factors
risk_factors_array = {
    'alpha': np.array([-9.011, -9.005, -9.422]),
    'beta':  np.array([0.434, 0.536, 0.669]),
    'gamma': np.array([0.018, 0.034, 0.030])
}

risk_level_map = {0: "Low Risk", 1: "Middle Risk", 2: "High Risk"}


# ============================================================================
# Helper Classes
# ============================================================================

class ContinuousToDiscreteCPD:
    """Helper class to discretize continuous distributions into TabularCPD using quantile binning"""
    
    def __init__(self, variable_name, n_bins, distribution, dist_params):
        """
        Args:
            variable_name: Name of the variable
            n_bins: Number of discrete bins
            distribution: scipy.stats distribution object
            dist_params: Dictionary of distribution parameters
        """
        self.variable_name = variable_name
        self.n_bins = n_bins
        self.distribution = distribution
        self.dist_params = dist_params
        
        # Calculate bins based on quantiles for better coverage
        quantiles = np.linspace(0.001, 0.999, n_bins + 1)
        self.bins = distribution.ppf(quantiles, **dist_params)
        
        # Calculate the mean value of each bin
        self.bin_means = self._calculate_bin_means()
        
    def _calculate_bin_means(self):
        """Calculate the mean value within each bin using numerical integration"""
        bin_means = []
        
        for i in range(self.n_bins):
            lower = self.bins[i]
            upper = self.bins[i + 1]
            
            # E[X | a < X < b] = ∫(x * f(x))dx / ∫f(x)dx from a to b
            def weighted_pdf(x):
                return x * self.distribution.pdf(x, **self.dist_params)
            
            numerator, _ = integrate.quad(weighted_pdf, lower, upper)
            denominator = (self.distribution.cdf(upper, **self.dist_params) - 
                          self.distribution.cdf(lower, **self.dist_params))
            
            if denominator < 1e-10:  # Avoid division by zero
                bin_mean = (lower + upper) / 2
            else:
                bin_mean = numerator / denominator
            
            bin_means.append(bin_mean)
        
        return np.array(bin_means)
    
    def get_probabilities(self):
        """Calculate discrete probabilities for each bin"""
        probs = []
        for i in range(self.n_bins):
            prob = (self.distribution.cdf(self.bins[i+1], **self.dist_params) - 
                   self.distribution.cdf(self.bins[i], **self.dist_params))
            probs.append(prob)
        
        # Normalize to ensure sum = 1.0
        probs = np.array(probs)
        probs = probs / probs.sum()
        return probs
    
    def discretize_value(self, continuous_value):
        """Convert a continuous value to discrete bin index"""
        return np.digitize(continuous_value, self.bins[1:-1])
    
    def continuous_from_bin(self, bin_index):
        """Convert a bin index back to a representative continuous value (bin mean)"""
        if 0 <= bin_index < self.n_bins:
            return self.bin_means[bin_index]
        else:
            raise ValueError(f"Bin index {bin_index} out of range [0, {self.n_bins})")
    
    def get_cpd(self):
        """Create TabularCPD for this variable"""
        probs = self.get_probabilities().reshape(-1, 1)
        return TabularCPD(
            variable=self.variable_name,
            variable_card=self.n_bins,
            values=probs
        )


class DeterministicVariableBinner:
    """Helper class to bin deterministic derived variables using quantile binning"""
    
    def __init__(self, variable_name, n_bins, all_possible_values):
        """
        Args:
            variable_name: Name of the variable
            n_bins: Number of bins
            all_possible_values: Array of all possible values this variable can take
        """
        self.variable_name = variable_name
        self.n_bins = n_bins
        
        # Create quantile-based bins
        quantiles = np.linspace(0, 1, n_bins + 1)
        self.bins = np.quantile(all_possible_values, quantiles)
        
        # Ensure unique bins (in case of repeated values)
        self.bins = np.unique(self.bins)
        if len(self.bins) < n_bins + 1:
            # If we have fewer unique bins than requested, adjust
            self.bins = np.linspace(min(all_possible_values), 
                                   max(all_possible_values), 
                                   n_bins + 1)
        
        # Calculate bin means
        self.bin_means = self._calculate_bin_means(all_possible_values)
        
    def _calculate_bin_means(self, all_values):
        """Calculate the empirical mean of values in each bin"""
        bin_means = []
        
        for i in range(len(self.bins) - 1):
            lower = self.bins[i]
            upper = self.bins[i + 1]
            
            # Find all values in this bin
            if i < len(self.bins) - 2:
                mask = (all_values >= lower) & (all_values < upper)
            else:
                # Last bin includes upper bound
                mask = (all_values >= lower) & (all_values <= upper)
            
            values_in_bin = all_values[mask]
            
            if len(values_in_bin) > 0:
                bin_mean = np.mean(values_in_bin)
            else:
                # Fallback to midpoint if no values in bin
                bin_mean = (lower + upper) / 2
            
            bin_means.append(bin_mean)
        
        return np.array(bin_means)
    
    def discretize_value(self, continuous_value):
        """Convert a continuous value to discrete bin index"""
        return np.digitize(continuous_value, self.bins[1:-1])
    
    def continuous_from_bin(self, bin_index):
        """Convert a bin index back to a representative continuous value (bin mean)"""
        if 0 <= bin_index < len(self.bin_means):
            return self.bin_means[bin_index]
        else:
            raise ValueError(f"Bin index {bin_index} out of range [0, {len(self.bin_means)})")


def create_deterministic_cpd(child, parents, parent_cards, func):
    """
    Create a CPD for deterministic relationships.
    
    Args:
        child: Name of child variable
        parents: List of parent variable names
        parent_cards: List of cardinalities for each parent
        func: Deterministic function that takes parent values and returns child value
    
    Returns:
        TabularCPD object
    """
    # Generate all parent combinations
    parent_combos = list(product(*[range(card) for card in parent_cards]))
    n_combos = len(parent_combos)
    
    # Calculate child cardinality
    child_vals = [func(*combo) for combo in parent_combos]
    child_card = max(child_vals) + 1
    
    # Initialize CPD table (all zeros)
    cpd_values = np.zeros((child_card, n_combos))
    
    # Fill in deterministic probabilities
    for i, parent_vals in enumerate(parent_combos):
        child_val = func(*parent_vals)
        cpd_values[child_val, i] = 1.0
    
    return TabularCPD(child, child_card, cpd_values,
                     evidence=parents, evidence_card=parent_cards)


# ============================================================================
# Step 1: Define discretized continuous variables
# ============================================================================

# People: Uniform distribution around observed value
people_disc = ContinuousToDiscreteCPD(
    variable_name='People',
    n_bins=N_BINS,
    distribution=stats.uniform,
    dist_params={'loc': max(0, idx_people - 5), 'scale': 10}
)

# Volume: Normal distribution
volume_disc = ContinuousToDiscreteCPD(
    variable_name='Volume',
    n_bins=N_BINS,
    distribution=stats.norm,
    dist_params={'loc': idx_volume, 'scale': 50}
)

# Probability of Death: Beta distribution
prob_death_disc = ContinuousToDiscreteCPD(
    variable_name='Prob_Death',
    n_bins=N_BINS,
    distribution=stats.beta,
    dist_params={'a': 413.5, 'b': 176171.5}
)


# ============================================================================
# Step 2: Building Type - Deterministic constant
# ============================================================================
building_type_cpd = TabularCPD(
    variable='Building_Type',
    variable_card=3,
    values=[[1.0 if i == BUILDING_TYPE else 0.0] for i in range(3)]
)


# ============================================================================
# Step 3: Activation Risk - Depends on Building Type with uncertainty
# ============================================================================
ar_values = np.array([
    [0.985, 0.01, 0.005],   # P(AR=0 | BT=0,1,2)
    [0.01, 0.98, 0.01],     # P(AR=1 | BT=0,1,2)
    [0.005, 0.01, 0.985]    # P(AR=2 | BT=0,1,2)
])

activation_risk_cpd = TabularCPD(
    variable='Activation_Risk',
    variable_card=3,
    values=ar_values,
    evidence=['Building_Type'],
    evidence_card=[3]
)


# ============================================================================
# Step 4: Fire Probability - Deterministic function of Volume and AR
# ============================================================================

def compute_fire_prob(volume_val, ar_level):
    """Compute fire probability for given volume value and AR level"""
    alpha = risk_factors_array['alpha'][ar_level]
    beta = risk_factors_array['beta'][ar_level]
    gamma = risk_factors_array['gamma'][ar_level]
    
    prob_fire = 2/np.pi * np.arctan(
        1.0 * np.exp(alpha) * volume_val ** beta / np.exp(21 * gamma)
    )
    return prob_fire


# Calculate all possible fire probabilities
fire_probs_all = []
for ar in range(3):
    for vol_bin in range(N_BINS):
        volume_val = volume_disc.bin_means[vol_bin]
        fire_prob = compute_fire_prob(volume_val, ar)
        fire_probs_all.append(fire_prob)

# Create binner for fire probability
fire_prob_binner = DeterministicVariableBinner(
    variable_name='Fire_Probability',
    n_bins=N_BINS,
    all_possible_values=np.array(fire_probs_all)
)


def fire_prob_func(volume_bin, ar_level):
    """Deterministic function for fire probability"""
    volume_val = volume_disc.bin_means[volume_bin]
    prob_fire = compute_fire_prob(volume_val, ar_level)
    fire_bin = fire_prob_binner.discretize_value(prob_fire)
    return fire_bin


# Create CPD using deterministic function
fire_probability_cpd = create_deterministic_cpd(
    child='Fire_Probability',
    parents=['Volume', 'Activation_Risk'],
    parent_cards=[N_BINS, 3],
    func=fire_prob_func
)


# ============================================================================
# Step 5: Expected Deaths - Deterministic function of People, Fire_Prob, Prob_Death
# ============================================================================

# Calculate all possible death values
deaths_all = []
for people_bin in range(N_BINS):
    for fire_bin in range(len(fire_prob_binner.bin_means)):
        for death_bin in range(N_BINS):
            people_val = people_disc.bin_means[people_bin]
            fire_val = fire_prob_binner.bin_means[fire_bin]
            prob_death_val = prob_death_disc.bin_means[death_bin]
            
            expected_death = people_val * fire_val * prob_death_val
            deaths_all.append(expected_death)

# Create binner for expected deaths
N_DEATH_BINS = 10
expected_deaths_binner = DeterministicVariableBinner(
    variable_name='Expected_Deaths',
    n_bins=N_DEATH_BINS,
    all_possible_values=np.array(deaths_all)
)


def expected_deaths_func(people_bin, fire_bin, death_bin):
    """Deterministic function for expected deaths"""
    people_val = people_disc.bin_means[people_bin]
    fire_val = fire_prob_binner.bin_means[fire_bin]
    prob_death_val = prob_death_disc.bin_means[death_bin]
    
    expected_death = people_val * fire_val * prob_death_val
    death_state = expected_deaths_binner.discretize_value(expected_death)
    return death_state


# Create CPD using deterministic function
expected_deaths_cpd = create_deterministic_cpd(
    child='Expected_Deaths',
    parents=['People', 'Fire_Probability', 'Prob_Death'],
    parent_cards=[N_BINS, len(fire_prob_binner.bin_means), N_BINS],
    func=expected_deaths_func
)


# ============================================================================
# Step 6: Build the Bayesian Network
# ============================================================================
model = DiscreteBayesianNetwork([
    ('Building_Type', 'Activation_Risk'),
    ('Volume', 'Fire_Probability'),
    ('Activation_Risk', 'Fire_Probability'),
    ('Fire_Probability', 'Expected_Deaths'),
    ('People', 'Expected_Deaths'),
    ('Prob_Death', 'Expected_Deaths')
])

# Add all CPDs
model.add_cpds(
    building_type_cpd,
    activation_risk_cpd,
    people_disc.get_cpd(),
    volume_disc.get_cpd(),
    prob_death_disc.get_cpd(),
    fire_probability_cpd,
    expected_deaths_cpd
)

# Verify model
assert model.check_model(), "Model validation failed!"
print("\n✓ Discrete Bayesian Network created successfully!")


# ============================================================================
# Step 7: Perform inference
# ============================================================================
inference = VariableElimination(model)

# Query: What's the distribution of expected deaths given the building type?
result = inference.query(['Expected_Deaths'])
print(f"\n{'='*70}")
print(f"P(Expected_Deaths | Building_Type={BUILDING_TYPE}):")
print(f"{'='*70}")
print(result)

# Map bin indices to actual values
print(f"\nExpected Deaths Interpretation:")
for i, prob in enumerate(result.values):
    if prob > 0.001:  # Only show bins with significant probability
        bin_lower = expected_deaths_binner.bins[i]
        bin_upper = expected_deaths_binner.bins[i+1]
        bin_mean = expected_deaths_binner.bin_means[i]
        print(f"  Bin {i}: [{bin_lower:.6f}, {bin_upper:.6f}] "
              f"(mean: {bin_mean:.6f}) - Probability: {prob:.4f}")


# ============================================================================
# Step 8: Sample from the network
# ============================================================================
sampler = BayesianModelSampling(model)
samples = sampler.forward_sample(size=10000)

print(f"\n{'='*70}")
print("Sample Statistics (Discrete Bins):")
print(f"{'='*70}")
print(samples.describe())

# Convert discrete bins back to continuous values for interpretation
samples['People_Value'] = samples['People'].apply(lambda x: people_disc.continuous_from_bin(x))
samples['Volume_Value'] = samples['Volume'].apply(lambda x: volume_disc.continuous_from_bin(x))
samples['Prob_Death_Value'] = samples['Prob_Death'].apply(lambda x: prob_death_disc.continuous_from_bin(x))
samples['Fire_Prob_Value'] = samples['Fire_Probability'].apply(lambda x: fire_prob_binner.continuous_from_bin(x))
samples['Expected_Deaths_Value'] = samples['Expected_Deaths'].apply(lambda x: expected_deaths_binner.continuous_from_bin(x))

print(f"\n{'='*70}")
print("Continuous Value Statistics:")
print(f"{'='*70}")
print(samples[['People_Value', 'Volume_Value', 'Prob_Death_Value', 
               'Fire_Prob_Value', 'Expected_Deaths_Value']].describe())


# ============================================================================
# Step 9: Visualize results
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Discrete bins
samples['People'].hist(bins=N_BINS, ax=axes[0, 0], edgecolor='black')
axes[0, 0].set_title(f'People Distribution (Discrete Bins)')
axes[0, 0].set_xlabel('Bin Index')
axes[0, 0].set_ylabel('Frequency')

samples['Volume'].hist(bins=N_BINS, ax=axes[0, 1], edgecolor='black')
axes[0, 1].set_title(f'Volume Distribution (Discrete Bins)')
axes[0, 1].set_xlabel('Bin Index')
axes[0, 1].set_ylabel('Frequency')

samples['Prob_Death'].hist(bins=N_BINS, ax=axes[0, 2], edgecolor='black')
axes[0, 2].set_title(f'Probability of Death (Discrete Bins)')
axes[0, 2].set_xlabel('Bin Index')
axes[0, 2].set_ylabel('Frequency')

samples['Activation_Risk'].hist(bins=3, ax=axes[1, 0], edgecolor='black')
axes[1, 0].set_title(f'Activation Risk\n(Building Type = {risk_level_map[BUILDING_TYPE]})')
axes[1, 0].set_xlabel('Risk Level')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_xticks([0, 1, 2])
axes[1, 0].set_xticklabels(['Low', 'Middle', 'High'])

samples['Fire_Probability'].hist(bins=len(fire_prob_binner.bin_means), ax=axes[1, 1], edgecolor='black')
axes[1, 1].set_title(f'Fire Probability (Discrete Bins)')
axes[1, 1].set_xlabel('Bin Index')
axes[1, 1].set_ylabel('Frequency')

samples['Expected_Deaths'].hist(bins=N_DEATH_BINS, ax=axes[1, 2], edgecolor='black')
axes[1, 2].set_title(f'Expected Deaths (Discrete Bins)')
axes[1, 2].set_xlabel('Bin Index')
axes[1, 2].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(f'bn_discrete_bins_buildingtype_{BUILDING_TYPE}.png', dpi=150)
plt.show()

# Continuous values plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

samples['People_Value'].hist(bins=30, ax=axes[0, 0], edgecolor='black', alpha=0.7)
axes[0, 0].set_title(f'People (Continuous Values)')
axes[0, 0].set_xlabel('Number of People')
axes[0, 0].set_ylabel('Frequency')

samples['Volume_Value'].hist(bins=30, ax=axes[0, 1], edgecolor='black', alpha=0.7)
axes[0, 1].set_title(f'Volume (Continuous Values)')
axes[0, 1].set_xlabel('Volume (m³)')
axes[0, 1].set_ylabel('Frequency')

samples['Prob_Death_Value'].hist(bins=30, ax=axes[0, 2], edgecolor='black', alpha=0.7)
axes[0, 2].set_title(f'Probability of Death (Continuous Values)')
axes[0, 2].set_xlabel('Probability')
axes[0, 2].set_ylabel('Frequency')

samples['Activation_Risk'].hist(bins=3, ax=axes[1, 0], edgecolor='black', alpha=0.7)
axes[1, 0].set_title(f'Activation Risk')
axes[1, 0].set_xlabel('Risk Level')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_xticks([0, 1, 2])
axes[1, 0].set_xticklabels(['Low', 'Middle', 'High'])

samples['Fire_Prob_Value'].hist(bins=30, ax=axes[1, 1], edgecolor='black', alpha=0.7)
axes[1, 1].set_title(f'Fire Probability (Continuous Values)')
axes[1, 1].set_xlabel('Probability')
axes[1, 1].set_ylabel('Frequency')

samples['Expected_Deaths_Value'].hist(bins=30, ax=axes[1, 2], edgecolor='black', alpha=0.7)
axes[1, 2].set_title(f'Expected Deaths (Continuous Values)')
axes[1, 2].set_xlabel('Expected Number')
axes[1, 2].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(f'bn_continuous_values_buildingtype_{BUILDING_TYPE}.png', dpi=150)
plt.show()


# ============================================================================
# Step 10: Summary
# ============================================================================
print("\n" + "="*70)
print("DISCRETE BAYESIAN NETWORK SUMMARY")
print("="*70)
print(f"Building Type: {BUILDING_TYPE} ({risk_level_map[BUILDING_TYPE]})")
print(f"Number of bins for continuous variables: {N_BINS}")
print(f"Number of bins for expected deaths: {N_DEATH_BINS}")
print(f"Total number of nodes: {len(model.nodes())}")
print(f"Total number of edges: {len(model.edges())}")
print(f"Model is valid: {model.check_model()}")
print(f"\nExpected Deaths Statistics:")
print(f"  Mean: {samples['Expected_Deaths_Value'].mean():.6f}")
print(f"  Std:  {samples['Expected_Deaths_Value'].std():.6f}")
print(f"  95% CI: [{samples['Expected_Deaths_Value'].quantile(0.025):.6f}, "
      f"{samples['Expected_Deaths_Value'].quantile(0.975):.6f}]")
print("="*70)