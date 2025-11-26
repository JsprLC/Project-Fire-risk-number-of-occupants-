import numpy as np
import pandas as pd

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


from scipy import integrate, stats
from itertools import product
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
N_BINS = 5  # Number of bins for continuous variables
N_DEATH_BINS = 10  # Number of bins for expected deaths
N_SAMPLES = 10000  # Number of samples for inference

# Risk factors
risk_factors_array = {
    'alpha': np.array([-9.011, -9.005, -9.422]),
    'beta':  np.array([0.434, 0.536, 0.669]),
    'gamma': np.array([0.018, 0.034, 0.030])
}

risk_level_map = {0: "Low Risk", 1: "Middle Risk", 2: "High Risk"}

# Activation Risk CPD
a_r_cpd = np.array([
    [0.985, 0.01, 0.005],
    [0.01, 0.98, 0.01],
    [0.005, 0.01, 0.985]
    ])

# ============================================================================
# Helper Classes
# ============================================================================

class ContinuousToDiscreteCPD:
    """Helper class to discretize continuous distributions into TabularCPD using quantile binning"""
    
    def __init__(self, variable_name, n_bins, distribution, dist_params):
        self.variable_name = variable_name
        self.n_bins = n_bins
        self.distribution = distribution
        self.dist_params = dist_params
        
        # Calculate bins based on quantiles
        quantiles = np.linspace(0.001, 0.999, n_bins + 1)
        self.bins = distribution.ppf(quantiles, **dist_params)
        self.bin_means = self._calculate_bin_means()
        
    def _calculate_bin_means(self):
        """Calculate the mean value within each bin using numerical integration"""
        bin_means = []
        
        for i in range(self.n_bins):
            lower = self.bins[i]
            upper = self.bins[i + 1]
            
            def weighted_pdf(x):
                return x * self.distribution.pdf(x, **self.dist_params)
            
            numerator, _ = integrate.quad(weighted_pdf, lower, upper)
            denominator = (self.distribution.cdf(upper, **self.dist_params) - 
                          self.distribution.cdf(lower, **self.dist_params))
            
            if denominator < 1e-10:
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
        
        probs = np.array(probs)
        probs = probs / probs.sum()
        return probs
    
    def continuous_from_bin(self, bin_index):
        """Convert a bin index back to a representative continuous value (bin mean)"""
        if 0 <= bin_index < self.n_bins:
            return self.bin_means[bin_index]
        else:
            raise ValueError(f"Bin index {bin_index} out of range")
    
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
        self.variable_name = variable_name
        self.n_bins = n_bins
        
        # Create quantile-based bins
        quantiles = np.linspace(0, 1, n_bins + 1)
        self.bins = np.quantile(all_possible_values, quantiles)
        
        # Ensure unique bins
        self.bins = np.unique(self.bins)
        if len(self.bins) < n_bins + 1:
            self.bins = np.linspace(min(all_possible_values), 
                                   max(all_possible_values), 
                                   n_bins + 1)
        
        self.bin_means = self._calculate_bin_means(all_possible_values)
        
    def _calculate_bin_means(self, all_values):
        """Calculate the empirical mean of values in each bin"""
        bin_means = []
        
        for i in range(len(self.bins) - 1):
            lower = self.bins[i]
            upper = self.bins[i + 1]
            
            if i < len(self.bins) - 2:
                mask = (all_values >= lower) & (all_values < upper)
            else:
                mask = (all_values >= lower) & (all_values <= upper)
            
            values_in_bin = all_values[mask]
            
            if len(values_in_bin) > 0:
                bin_mean = np.mean(values_in_bin)
            else:
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
            raise ValueError(f"Bin index {bin_index} out of range")


def create_deterministic_cpd(child, parents, parent_cards, func):
    """Create a CPD for deterministic relationships"""
    parent_combos = list(product(*[range(card) for card in parent_cards]))
    n_combos = len(parent_combos)
    
    child_vals = [func(*combo) for combo in parent_combos]
    child_card = max(child_vals) + 1
    
    cpd_values = np.zeros((child_card, n_combos))
    
    for i, parent_vals in enumerate(parent_combos):
        child_val = func(*parent_vals)
        cpd_values[child_val, i] = 1.0
    
    return TabularCPD(child, child_card, cpd_values,
                     evidence=parents, evidence_card=parent_cards)



# ============================================================================
# Bayesian Network Builder
# ============================================================================

class BuildingRiskBayesianNetwork:
    """Build and run Bayesian Network for a single building"""
    
    def __init__(self, building_type, num_occupants, volume):
        self.building_type = building_type
        self.num_occupants = num_occupants
        self.volume = volume
        
        # Initialize discretizers
        self._initialize_discretizers()
        
        # Build the network
        self.model = self._build_network()
        
    def _initialize_discretizers(self):
        """Initialize all continuous variable discretizers"""
        # People: Uniform distribution around observed value
        self.people_disc = ContinuousToDiscreteCPD(
            variable_name='People',
            n_bins=N_BINS,
            distribution=stats.uniform,
            dist_params={'loc': max(0, self.num_occupants - 5), 'scale': 10}
        )
        
        # Volume: Normal distribution
        self.volume_disc = ContinuousToDiscreteCPD(
            variable_name='Volume',
            n_bins=N_BINS,
            distribution=stats.norm,
            dist_params={'loc': self.volume, 'scale': 50}
        )
        
        # Probability of Death: Beta distribution
        self.prob_death_disc = ContinuousToDiscreteCPD(
            variable_name='Prob_Death',
            n_bins=N_BINS,
            distribution=stats.beta,
            dist_params={'a': 413.5, 'b': 176171.5}
        )
        
        # Fire probability binner
        fire_probs_all = []
        for ar in range(3):
            for vol_bin in range(N_BINS):
                volume_val = self.volume_disc.bin_means[vol_bin]
                fire_prob = self._compute_fire_prob(volume_val, ar)
                fire_probs_all.append(fire_prob)
        
        self.fire_prob_binner = DeterministicVariableBinner(
            variable_name='Fire_Probability',
            n_bins=N_BINS,
            all_possible_values=np.array(fire_probs_all)
        )
        
        # Expected deaths binner
        deaths_all = []
        for people_bin in range(N_BINS):
            for fire_bin in range(len(self.fire_prob_binner.bin_means)):
                for death_bin in range(N_BINS):
                    people_val = self.people_disc.bin_means[people_bin]
                    fire_val = self.fire_prob_binner.bin_means[fire_bin]
                    prob_death_val = self.prob_death_disc.bin_means[death_bin]
                    
                    expected_death = people_val * fire_val * prob_death_val
                    deaths_all.append(expected_death)
        
        self.expected_deaths_binner = DeterministicVariableBinner(
            variable_name='Expected_Deaths',
            n_bins=N_DEATH_BINS,
            all_possible_values=np.array(deaths_all)
        )
    
    def _compute_fire_prob(self, volume_val, ar_level):
        """Compute fire probability for given volume value and AR level"""
        alpha = risk_factors_array['alpha'][ar_level]
        beta = risk_factors_array['beta'][ar_level]
        gamma = risk_factors_array['gamma'][ar_level]
        
        prob_fire = 2/np.pi * np.arctan(
            1.0 * np.exp(alpha) * volume_val ** beta / np.exp(21 * gamma)
        )
        return prob_fire
    
    def _build_network(self):
        """Build the complete Bayesian Network"""
        # Building Type CPD
        building_type_cpd = TabularCPD(
            variable='Building_Type',
            variable_card=3,
            values=[[1.0 if i == self.building_type else 0.0] for i in range(3)]
        )
        
        # Activation Risk CPD
        ar_values = a_r_cpd
        
        activation_risk_cpd = TabularCPD(
            variable='Activation_Risk',
            variable_card=3,
            values=ar_values,
            evidence=['Building_Type'],
            evidence_card=[3]
        )
        
        # Fire Probability CPD
        def fire_prob_func(volume_bin, ar_level):
            volume_val = self.volume_disc.bin_means[volume_bin]
            prob_fire = self._compute_fire_prob(volume_val, ar_level)
            fire_bin = self.fire_prob_binner.discretize_value(prob_fire)
            return fire_bin
        
        fire_probability_cpd = create_deterministic_cpd(
            child='Fire_Probability',
            parents=['Volume', 'Activation_Risk'],
            parent_cards=[N_BINS, 3],
            func=fire_prob_func
        )
        
        # Expected Deaths CPD
        def expected_deaths_func(people_bin, fire_bin, death_bin):
            people_val = self.people_disc.bin_means[people_bin]
            fire_val = self.fire_prob_binner.bin_means[fire_bin]
            prob_death_val = self.prob_death_disc.bin_means[death_bin]
            
            expected_death = people_val * fire_val * prob_death_val
            death_state = self.expected_deaths_binner.discretize_value(expected_death)
            return death_state
        
        expected_deaths_cpd = create_deterministic_cpd(
            child='Expected_Deaths',
            parents=['People', 'Fire_Probability', 'Prob_Death'],
            parent_cards=[N_BINS, len(self.fire_prob_binner.bin_means), N_BINS],
            func=expected_deaths_func
        )
        
        # Build model
        model = DiscreteBayesianNetwork([
            ('Building_Type', 'Activation_Risk'),
            ('Volume', 'Fire_Probability'),
            ('Activation_Risk', 'Fire_Probability'),
            ('Fire_Probability', 'Expected_Deaths'),
            ('People', 'Expected_Deaths'),
            ('Prob_Death', 'Expected_Deaths')
        ])
        
        # Add CPDs
        model.add_cpds(
            building_type_cpd,
            activation_risk_cpd,
            self.people_disc.get_cpd(),
            self.volume_disc.get_cpd(),
            self.prob_death_disc.get_cpd(),
            fire_probability_cpd,
            expected_deaths_cpd
        )
        
        assert model.check_model(), "Model validation failed!"
        
        return model
    
    def get_expected_deaths_distribution(self):
        """
        Get the complete discrete distribution of expected deaths
        
        Returns:
            tuple: (bin_means array, probabilities array)
        """
        inference = VariableElimination(self.model)
        result = inference.query(['Expected_Deaths'])
        
        # Get all bin means and probabilities (including zeros)
        bin_means = self.expected_deaths_binner.bin_means
        probabilities = result.values
        
        return bin_means, probabilities
    
    def get_distribution_dict(self):
        """
        Get distribution as dictionary
        
        Returns:
            dict: {bin_idx: {'bin_mean': value, 'probability': prob}}
        """
        bin_means, probabilities = self.get_expected_deaths_distribution()
        
        distribution_dict = {}
        for bin_idx in range(len(bin_means)):
            distribution_dict[bin_idx] = {
                'bin_mean': bin_means[bin_idx],
                'probability': probabilities[bin_idx]
            }
        
        return distribution_dict




# ============================================================================
# Process Multiple Buildings
# ============================================================================

def process_buildings(buildings_gdf, building_type_col='building_type'):
    """
    Process multiple buildings and return complete discrete distributions
    
    Args:
        buildings_gdf: GeoDataFrame with building information
        building_type_col: Column name for building type
    
    Returns:
        pd.DataFrame: DataFrame with complete distribution for each building
    """
    all_results = []
    
    print(f"Processing {len(buildings_gdf)} buildings...")
    
    for idx in tqdm(range(len(buildings_gdf))):
        building = buildings_gdf.iloc[idx]
        
        # Extract building parameters
        gml_id = building['gml_id']
        num_occupants = building['total_occupants']
        building_type = 2 # Activation risk is high for residential buildings
        
        # Calculate volume (adjust based on your data)
        volume = building.get('_volume', 1865.1)
        
        try:
            # Build Bayesian Network
            bn = BuildingRiskBayesianNetwork(
                building_type=int(building_type),
                num_occupants=num_occupants,
                volume=volume
            )
            
            # Get complete discrete distribution
            # ==========================================================================!!!!!!!!!!!!!!!!!!!!
            # Very important: get full distribution including zero probabilities
            # ========================================================================== 
            bin_means, probabilities = bn.get_expected_deaths_distribution()
            
            # Create row with all bins
            result_row = {
                'gml_id': gml_id,
                'building_type': building_type,
                'num_occupants': num_occupants,
                'volume': volume
            }
            
            # Add each bin's mean and probability
            for bin_idx in range(len(bin_means)):
                result_row[f'bin_{bin_idx}_mean'] = bin_means[bin_idx]
                result_row[f'bin_{bin_idx}_prob'] = probabilities[bin_idx]
            
            all_results.append(result_row)
        
        except Exception as e:
            print(f"\nError processing building {gml_id}: {str(e)}")
            continue
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    return results_df


# ============================================================================
# Post-Processing Functions
# ============================================================================
def create_long_format_results(results_df):
    """
    Convert wide format to long format for easier analysis
    
    Args:
        results_df: Wide format DataFrame from process_buildings()
    
    Returns:
        pd.DataFrame: Long format with one row per building-bin combination
    """
    # Identify bin columns
    bin_mean_cols = [col for col in results_df.columns if col.startswith('bin_') and col.endswith('_mean')]
    bin_prob_cols = [col for col in results_df.columns if col.startswith('bin_') and col.endswith('_prob')]
    
    n_bins = len(bin_mean_cols)
    
    long_data = []
    
    for _, row in results_df.iterrows():
        gml_id = row['gml_id']
        building_type = row['building_type']
        num_occupants = row['num_occupants']
        volume = row['volume']
        
        for bin_idx in range(n_bins):
            bin_mean = row[f'bin_{bin_idx}_mean']
            bin_prob = row[f'bin_{bin_idx}_prob']
            
            long_data.append({
                'gml_id': gml_id,
                'building_type': building_type,
                'num_occupants': num_occupants,
                'volume': volume,
                'bin_index': bin_idx,
                'bin_mean': bin_mean,
                'probability': bin_prob
            })
    
    return pd.DataFrame(long_data)


def compute_statistics_from_distribution(results_df):
    """
    Compute summary statistics from the discrete distributions
    
    Args:
        results_df: Wide format DataFrame from process_buildings()
    
    Returns:
        pd.DataFrame: Summary statistics for each building
    """
    summary_data = []
    
    # Identify bin columns
    bin_mean_cols = [col for col in results_df.columns if col.startswith('bin_') and col.endswith('_mean')]
    bin_prob_cols = [col for col in results_df.columns if col.startswith('bin_') and col.endswith('_prob')]
    
    for _, row in results_df.iterrows():
        gml_id = row['gml_id']
        
        # Extract means and probabilities
        means = np.array([row[col] for col in bin_mean_cols])
        probs = np.array([row[col] for col in bin_prob_cols])
        
        # Calculate statistics
        expected_value = np.sum(means * probs)
        variance = np.sum((means ** 2) * probs) - expected_value ** 2
        std_dev = np.sqrt(variance)
        
        # Calculate quantiles
        cumulative_probs = np.cumsum(probs)
        q025_idx = np.searchsorted(cumulative_probs, 0.025)
        q50_idx = np.searchsorted(cumulative_probs, 0.50)
        q975_idx = np.searchsorted(cumulative_probs, 0.975)
        
        summary_data.append({
            'gml_id': gml_id,
            'expected_deaths_mean': expected_value,
            'expected_deaths_std': std_dev,
            'expected_deaths_q025': means[q025_idx],
            'expected_deaths_median': means[q50_idx],
            'expected_deaths_q975': means[q975_idx],
            'building_type': row['building_type'],
            'num_occupants': row['num_occupants'],
            'volume': row['volume']
        })
    
    return pd.DataFrame(summary_data)
