"""
Hidden Markov Model for Manufacturing Quality Control
=====================================================

Models a manufacturing process where product quality depends on a hidden state
that evolves over time. Partially observed data (missing observations) are handled
using the Baum-Welch algorithm.

Author: Data Science Team
Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class HiddenMarkovModel:
    """
    Hidden Markov Model for manufacturing quality control.
    
    Hidden states: 0 = Good, 1 = Bad
    Observations: 1 = Good, 0 = Bad, None = Not tested
    """
    
    def __init__(self, n_states: int = 2, random_state: int = 42):
        """
        Initialize HMM.
        
        Args:
            n_states: Number of hidden states (default: 2 for Good/Bad)
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Initialize parameters
        self.transition_matrix = None  # A[i,j] = P(state_t = j | state_t-1 = i)
        self.emission_matrix = None    # B[i,j] = P(obs = j | state = i)
        self.initial_state_prob = None # π[i] = P(state_0 = i)
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize parameters with reasonable defaults."""
        # Initial state probability (assume more likely to start in Good state)
        self.initial_state_prob = np.array([0.8, 0.2])
        
        # Transition matrix (diagonal-dominant: more likely to stay in same state)
        self.transition_matrix = np.array([
            [0.95, 0.05],  # Good -> [Good, Bad]
            [0.10, 0.90]   # Bad  -> [Good, Bad]
        ])
        
        # Emission matrix (nearly diagonal: state is observed correctly most of the time)
        self.emission_matrix = np.array([
            [0.95, 0.05],  # State Good -> observe [Good, Bad]
            [0.05, 0.95]   # State Bad  -> observe [Good, Bad]
        ])
    
    def forward_algorithm(self, observations: List[Optional[int]]) -> np.ndarray:
        """
        Forward algorithm for HMM inference.
        Compute forward probabilities: α[t,i] = P(obs[0:t], state_t = i)
        
        Args:
            observations: List of observations (1=Good, 0=Bad, None=not tested)
            
        Returns:
            Forward probabilities matrix of shape (T, n_states)
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        # Initialize: α[0,i] = π[i] * B[i, obs[0]]
        if observations[0] is None:
            # No observation: uniform emission
            alpha[0] = self.initial_state_prob
        else:
            alpha[0] = self.initial_state_prob * self.emission_matrix[:, observations[0]]
        
        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                # Sum over previous states
                state_prob = np.sum(alpha[t-1, :] * self.transition_matrix[:, j])
                
                # Multiply by emission probability
                if observations[t] is None:
                    alpha[t, j] = state_prob
                else:
                    alpha[t, j] = state_prob * self.emission_matrix[j, observations[t]]
        
        return alpha
    
    def backward_algorithm(self, observations: List[Optional[int]]) -> np.ndarray:
        """
        Backward algorithm for HMM inference.
        Compute backward probabilities: β[t,i] = P(obs[t+1:T] | state_t = i)
        
        Args:
            observations: List of observations
            
        Returns:
            Backward probabilities matrix of shape (T, n_states)
        """
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # Initialize: β[T-1,i] = 1
        beta[-1] = 1.0
        
        # Backward pass
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    # β[t,i] += A[i,j] * B[j, obs[t+1]] * β[t+1,j]
                    if observations[t+1] is None:
                        emission_prob = 1.0
                    else:
                        emission_prob = self.emission_matrix[j, observations[t+1]]
                    
                    beta[t, i] += self.transition_matrix[i, j] * emission_prob * beta[t+1, j]
        
        return beta
    
    def posterior_state_probability(self, observations: List[Optional[int]]) -> np.ndarray:
        """
        Compute posterior probability of hidden states given observations.
        γ[t,i] = P(state_t = i | observations)
        
        Args:
            observations: List of observations
            
        Returns:
            Posterior probabilities matrix of shape (T, n_states)
        """
        alpha = self.forward_algorithm(observations)
        beta = self.backward_algorithm(observations)
        
        # γ[t,i] = (α[t,i] * β[t,i]) / P(observations)
        gamma = (alpha * beta) / np.sum(alpha * beta, axis=1, keepdims=True)
        
        return gamma
    
    def baum_welch(self, observations: List[Optional[int]], n_iterations: int = 100, 
                   tolerance: float = 1e-6) -> Dict:
        """
        Baum-Welch algorithm (EM algorithm) for parameter estimation.
        Handles missing observations (None values).
        
        Args:
            observations: List of observations
            n_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary with training history
        """
        T = len(observations)
        history = {'log_likelihood': []}
        
        for iteration in range(n_iterations):
            # E-step: compute forward-backward probabilities
            alpha = self.forward_algorithm(observations)
            beta = self.backward_algorithm(observations)
            
            # Likelihood
            likelihood = np.sum(alpha[-1])
            log_likelihood = np.log(likelihood + 1e-10)
            history['log_likelihood'].append(log_likelihood)
            
            # Compute posterior probabilities
            gamma = (alpha * beta) / np.sum(alpha * beta, axis=1, keepdims=True)
            
            # Compute transition probabilities (digamma)
            xi = np.zeros((T-1, self.n_states, self.n_states))
            for t in range(T-1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        if observations[t+1] is None:
                            emission_prob = 1.0
                        else:
                            emission_prob = self.emission_matrix[j, observations[t+1]]
                        
                        xi[t, i, j] = (alpha[t, i] * self.transition_matrix[i, j] * 
                                      emission_prob * beta[t+1, j]) / (likelihood + 1e-10)
            
            # M-step: update parameters
            # Update initial state probability
            self.initial_state_prob = gamma[0]
            
            # Update transition matrix
            for i in range(self.n_states):
                for j in range(self.n_states):
                    numerator = np.sum(xi[:, i, j])
                    denominator = np.sum(gamma[:-1, i])
                    self.transition_matrix[i, j] = (numerator / (denominator + 1e-10) + 1e-10)
            
            # Normalize transition matrix
            self.transition_matrix /= np.sum(self.transition_matrix, axis=1, keepdims=True)
            
            # Update emission matrix
            for j in range(self.n_states):
                for k in range(2):  # k = 0 (Bad) or 1 (Good)
                    numerator = 0
                    denominator = 0
                    for t in range(T):
                        if observations[t] == k or observations[t] is None:
                            if observations[t] == k:
                                numerator += gamma[t, j]
                            denominator += gamma[t, j]
                    
                    self.emission_matrix[j, k] = (numerator / (denominator + 1e-10) + 1e-10)
            
            # Normalize emission matrix
            self.emission_matrix /= np.sum(self.emission_matrix, axis=1, keepdims=True)
            
            # Check convergence
            if iteration > 0:
                ll_diff = abs(history['log_likelihood'][-1] - history['log_likelihood'][-2])
                if ll_diff < tolerance:
                    print(f"Converged after {iteration} iterations")
                    break
        
        return history


class ManufacturingSimulator:
    """
    Simulator for manufacturing process with hidden state and sampling strategy.
    """
    
    def __init__(self, transition_matrix: np.ndarray, 
                 emission_matrix: np.ndarray,
                 initial_state_prob: np.ndarray,
                 random_state: int = 42):
        """
        Initialize simulator.
        
        Args:
            transition_matrix: Transition probabilities between states
            emission_matrix: Emission probabilities (state -> observation)
            initial_state_prob: Initial state distribution
            random_state: Random seed
        """
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.initial_state_prob = initial_state_prob
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_sequence(self, T: int, sampling_frequency: int = 1) -> Tuple[List, List, List]:
        """
        Generate a sequence of hidden states and observations.
        
        Args:
            T: Sequence length
            sampling_frequency: Test every N items (1 = test every item, 10 = test every 10th)
            
        Returns:
            Tuple of (hidden_states, observations, test_indices)
        """
        hidden_states = []
        observations = []
        test_indices = []
        
        # Generate initial state
        state = np.random.choice([0, 1], p=self.initial_state_prob)
        hidden_states.append(state)
        
        # Generate sequence
        for t in range(1, T):
            # Transition to next state
            state = np.random.choice([0, 1], p=self.transition_matrix[state])
            hidden_states.append(state)
        
        # Generate observations based on sampling frequency
        for t in range(T):
            if t % sampling_frequency == 0:
                # Test this item
                obs = np.random.choice([0, 1], p=self.emission_matrix[hidden_states[t]])
                observations.append(obs)
                test_indices.append(t)
            else:
                observations.append(None)
        
        return hidden_states, observations, test_indices
    
    def generate_multiple_sequences(self, n_sequences: int, T: int, 
                                   sampling_frequency: int = 1) -> List[Tuple]:
        """
        Generate multiple sequences.
        
        Args:
            n_sequences: Number of sequences to generate
            T: Length of each sequence
            sampling_frequency: Sampling frequency
            
        Returns:
            List of (hidden_states, observations, test_indices) tuples
        """
        sequences = []
        for _ in range(n_sequences):
            seq = self.generate_sequence(T, sampling_frequency)
            sequences.append(seq)
        return sequences


class SamplingStrategyEvaluator:
    """
    Evaluate sampling strategies using Monte Carlo simulations.
    """
    
    @staticmethod
    def detection_delay(hidden_states: List[int], observations: List[Optional[int]], 
                       test_indices: List[int]) -> Optional[int]:
        """
        Calculate detection delay: time from entering Bad state to first detection.
        
        Args:
            hidden_states: True hidden states
            observations: Observations (including None for untested)
            test_indices: Indices where testing occurred
            
        Returns:
            Detection delay in items, or None if never detected
        """
        # Find first time we enter Bad state
        bad_entry_idx = None
        for t in range(1, len(hidden_states)):
            if hidden_states[t] == 1 and hidden_states[t-1] == 0:
                bad_entry_idx = t
                break
        
        if bad_entry_idx is None:
            return None
        
        # Find first test that detects the bad state
        for test_idx in test_indices:
            if test_idx >= bad_entry_idx and observations[test_idx] == 0:
                return test_idx - bad_entry_idx
        
        return None
    
    @staticmethod
    def undetected_bad_fraction(hidden_states: List[int], observations: List[Optional[int]]) -> float:
        """
        Calculate fraction of bad items that were not detected (not tested).
        
        Args:
            hidden_states: True hidden states
            observations: Observations
            
        Returns:
            Fraction of bad items not detected
        """
        bad_untested = sum(1 for i, (state, obs) in enumerate(zip(hidden_states, observations))
                          if state == 1 and obs is None)
        total_bad = sum(1 for state in hidden_states if state == 1)
        
        return bad_untested / total_bad if total_bad > 0 else 0
    
    @staticmethod
    def false_negatives(hidden_states: List[int], observations: List[Optional[int]]) -> int:
        """Count bad items not detected (state=Bad, obs=None or obs=Good)."""
        return sum(1 for state, obs in zip(hidden_states, observations)
                  if state == 1 and obs != 0)
    
    @staticmethod
    def false_positives(hidden_states: List[int], observations: List[Optional[int]]) -> int:
        """Count good items incorrectly classified (state=Good, obs=Bad)."""
        return sum(1 for state, obs in zip(hidden_states, observations)
                  if state == 0 and obs == 0)
    
    @staticmethod
    def monte_carlo_evaluation(simulator: ManufacturingSimulator, 
                               sampling_frequencies: List[int],
                               n_simulations: int = 100,
                               T: int = 1000) -> Dict:
        """
        Run Monte Carlo simulations to evaluate different sampling strategies.
        
        Args:
            simulator: ManufacturingSimulator instance
            sampling_frequencies: List of sampling frequencies to test
            n_simulations: Number of simulations per strategy
            T: Sequence length per simulation
            
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        
        for freq in sampling_frequencies:
            print(f"Evaluating sampling frequency: 1/{freq}")
            
            delays = []
            undetected_fractions = []
            false_negatives = []
            false_positives = []
            
            sequences = simulator.generate_multiple_sequences(n_simulations, T, freq)
            
            for hidden_states, observations, test_indices in sequences:
                # Detection delay
                delay = SamplingStrategyEvaluator.detection_delay(
                    hidden_states, observations, test_indices)
                if delay is not None:
                    delays.append(delay)
                
                # Undetected bad fraction
                undetected = SamplingStrategyEvaluator.undetected_bad_fraction(
                    hidden_states, observations)
                undetected_fractions.append(undetected)
                
                # Errors
                fn = SamplingStrategyEvaluator.false_negatives(hidden_states, observations)
                fp = SamplingStrategyEvaluator.false_positives(hidden_states, observations)
                false_negatives.append(fn)
                false_positives.append(fp)
            
            results[freq] = {
                'avg_detection_delay': np.mean(delays) if delays else None,
                'std_detection_delay': np.std(delays) if delays else None,
                'avg_undetected_fraction': np.mean(undetected_fractions),
                'std_undetected_fraction': np.std(undetected_fractions),
                'avg_false_negatives': np.mean(false_negatives),
                'avg_false_positives': np.mean(false_positives),
            }
        
        return results


# Visualization functions
def plot_hidden_vs_observed(hidden_states: List[int], observations: List[Optional[int]], 
                           test_indices: List[int], title: str = "Hidden vs Observed States"):
    """
    Plot hidden states vs observations.
    
    Args:
        hidden_states: True hidden states
        observations: Observations (with None for untested)
        test_indices: Indices where testing occurred
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), facecolor='white')
    
    # Plot hidden states
    ax1.plot(hidden_states, linewidth=1, color='black', label='True Hidden State')
    ax1.fill_between(range(len(hidden_states)), 0, hidden_states, alpha=0.3, color='gray')
    ax1.set_ylabel('Hidden State (0=Good, 1=Bad)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{title} - Hidden States', fontsize=12, fontweight='bold')
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot observations
    test_obs = [(i, observations[i]) for i in test_indices if observations[i] is not None]
    if test_obs:
        test_idx, test_val = zip(*test_obs)
        colors = ['green' if v == 1 else 'red' for v in test_val]
        ax2.scatter(test_idx, test_val, c=colors, s=50, alpha=0.6, edgecolors='black', 
                   label='Tested & Observed')
    
    ax2.set_ylabel('Observation (0=Bad, 1=Good)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Time Index', fontsize=11, fontweight='bold')
    ax2.set_title(f'{title} - Observations (Only Tested Items)', fontsize=12, fontweight='bold')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig


def plot_posterior_probability(posterior: np.ndarray, observations: List[Optional[int]],
                              title: str = "Posterior Probability of Bad State"):
    """
    Plot posterior probability of being in Bad state over time.
    
    Args:
        posterior: Posterior probabilities from forward-backward algorithm
        observations: Observations (with None for untested)
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(16, 6), facecolor='white')
    
    # Posterior probability of being in state 1 (Bad)
    prob_bad = posterior[:, 1]
    
    ax.plot(prob_bad, linewidth=1.5, color='red', alpha=0.7, label='P(State=Bad | Obs)')
    ax.fill_between(range(len(prob_bad)), 0, prob_bad, alpha=0.2, color='red')
    
    # Mark observations
    for i, obs in enumerate(observations):
        if obs is not None:
            ax.axvline(x=i, color='blue' if obs == 1 else 'darkred', 
                       alpha=0.3, linestyle='--', linewidth=1)
    
    ax.set_ylabel('Probability', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Index', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_strategy_comparison(results: Dict, title: str = "Sampling Strategy Comparison"):
    """
    Plot comparison of different sampling strategies.
    
    Args:
        results: Dictionary from monte_carlo_evaluation
        title: Plot title
    """
    frequencies = sorted(results.keys())
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10), facecolor='white')
    
    # Detection delay
    delays = [results[f]['avg_detection_delay'] if results[f]['avg_detection_delay'] else 0 
             for f in frequencies]
    ax1.bar(range(len(frequencies)), delays, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Avg Detection Delay (items)', fontsize=11, fontweight='bold')
    ax1.set_title('Detection Delay by Strategy', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(frequencies)))
    ax1.set_xticklabels([f'1/{f}' for f in frequencies])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Undetected fraction
    undetected = [results[f]['avg_undetected_fraction'] for f in frequencies]
    ax2.plot(range(len(frequencies)), undetected, marker='o', linewidth=2, 
            color='red', markersize=8, label='Avg Undetected Fraction')
    ax2.fill_between(range(len(frequencies)), undetected, alpha=0.3, color='red')
    ax2.set_ylabel('Fraction of Bad Items Missed', fontsize=11, fontweight='bold')
    ax2.set_title('Undetected Bad Items by Strategy', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(frequencies)))
    ax2.set_xticklabels([f'1/{f}' for f in frequencies])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # False negatives
    fn = [results[f]['avg_false_negatives'] for f in frequencies]
    ax3.bar(range(len(frequencies)), fn, color='orange', alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Avg False Negatives', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Sampling Strategy', fontsize=11, fontweight='bold')
    ax3.set_title('False Negatives by Strategy', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(frequencies)))
    ax3.set_xticklabels([f'1/{f}' for f in frequencies])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # False positives
    fp = [results[f]['avg_false_positives'] for f in frequencies]
    ax4.bar(range(len(frequencies)), fp, color='purple', alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Avg False Positives', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Sampling Strategy', fontsize=11, fontweight='bold')
    ax4.set_title('False Positives by Strategy', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(len(frequencies)))
    ax4.set_xticklabels([f'1/{f}' for f in frequencies])
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

