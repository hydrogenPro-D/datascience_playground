"""
Hidden Markov Model Example - Manufacturing Quality Control
===========================================================

This script demonstrates the complete HMM workflow:
1. Generate synthetic data with hidden states and partial observations
2. Train HMM using Baum-Welch algorithm
3. Perform inference with forward-backward algorithm
4. Evaluate different sampling strategies using Monte Carlo simulations
5. Visualize results
"""

import sys
sys.path.insert(0, r'/')

import numpy as np
import matplotlib.pyplot as plt
from hmm_quality_control import (
    HiddenMarkovModel, ManufacturingSimulator, SamplingStrategyEvaluator,
    plot_hidden_vs_observed, plot_posterior_probability, plot_strategy_comparison
)

print("=" * 80)
print("HIDDEN MARKOV MODEL FOR MANUFACTURING QUALITY CONTROL")
print("=" * 80)

# ============================================================================
# PART 1: SETUP TRUE PARAMETERS
# ============================================================================
print("\n[1] Setting up true manufacturing process parameters...")

# True parameters (these would be learned from data in practice)
TRUE_TRANSITION = np.array([
    [0.95, 0.05],  # Good -> [Good, Bad]: 95% stay good, 5% transition to bad
    [0.10, 0.90]   # Bad  -> [Good, Bad]: 10% return to good, 90% stay bad
])

TRUE_EMISSION = np.array([
    [0.98, 0.02],  # State Good -> observe [Bad, Good]: 2% error rate
    [0.05, 0.95]   # State Bad  -> observe [Bad, Good]: 5% error rate
])

TRUE_INITIAL = np.array([0.9, 0.1])  # Start in good state 90% of the time

print(f"Transition matrix (True):")
print(f"  P(Good -> Good) = {TRUE_TRANSITION[0,0]:.3f}")
print(f"  P(Good -> Bad)  = {TRUE_TRANSITION[0,1]:.3f}")
print(f"  P(Bad  -> Good) = {TRUE_TRANSITION[1,0]:.3f}")
print(f"  P(Bad  -> Bad)  = {TRUE_TRANSITION[1,1]:.3f}")

# ============================================================================
# PART 2: GENERATE SYNTHETIC DATA
# ============================================================================
print("\n[2] Generating synthetic manufacturing data...")

simulator = ManufacturingSimulator(
    transition_matrix=TRUE_TRANSITION,
    emission_matrix=TRUE_EMISSION,
    initial_state_prob=TRUE_INITIAL,
    random_state=42
)

# Generate one example sequence (test every 5th item)
hidden_states, observations, test_indices = simulator.generate_sequence(T=1000, sampling_frequency=5)

n_tested = sum(1 for o in observations if o is not None)
n_bad = sum(1 for h in hidden_states if h == 1)
n_bad_tested = sum(1 for h, o in zip(hidden_states, observations) if h == 1 and o == 0)

print(f"Generated sequence of length {len(hidden_states)}")
print(f"  Actual bad items: {n_bad} ({100*n_bad/len(hidden_states):.1f}%)")
print(f"  Items tested: {n_tested} ({100*n_tested/len(hidden_states):.1f}%)")
print(f"  Bad items detected: {n_bad_tested} ({100*n_bad_tested/n_bad*100:.1f}% of actual bad)")
print(f"  Undetected bad items: {n_bad - n_bad_tested}")

# ============================================================================
# PART 3: VISUALIZE GROUND TRUTH
# ============================================================================
print("\n[3] Visualizing ground truth (hidden vs observed)...")

fig = plot_hidden_vs_observed(hidden_states, observations, test_indices,
                             title="Ground Truth - Hidden States vs Observations")
plt.savefig('data/hmm_ground_truth.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: data/hmm_ground_truth.png")
plt.show()

# ============================================================================
# PART 4: TRAIN HMM
# ============================================================================
print("\n[4] Training HMM using Baum-Welch algorithm...")

hmm = HiddenMarkovModel(n_states=2, random_state=42)
print("Initial parameters:")
print(f"  Transition matrix:\n{hmm.transition_matrix}")
print(f"  Emission matrix:\n{hmm.emission_matrix}")

training_history = hmm.baum_welch(observations, n_iterations=50, tolerance=1e-6)

print("\nTrained parameters:")
print(f"  Transition matrix:\n{hmm.transition_matrix}")
print(f"  Emission matrix:\n{hmm.emission_matrix}")

# Plot training history
fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
ax.plot(training_history['log_likelihood'], linewidth=2, color='steelblue')
ax.set_ylabel('Log Likelihood', fontsize=11, fontweight='bold')
ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
ax.set_title('Baum-Welch Training Convergence', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/hmm_training_history.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: data/hmm_training_history.png")
plt.show()

# ============================================================================
# PART 5: INFERENCE
# ============================================================================
print("\n[5] Performing inference on observations...")

posterior = hmm.posterior_state_probability(observations)

# Print inference statistics
prob_bad = posterior[:, 1]
print(f"Posterior probability of Bad state:")
print(f"  Mean: {np.mean(prob_bad):.3f}")
print(f"  Max:  {np.max(prob_bad):.3f}")
print(f"  Min:  {np.min(prob_bad):.3f}")

# Visualize posterior probability
fig = plot_posterior_probability(posterior, observations,
                               title="Posterior Probability of Bad State (After Training)")
plt.savefig('data/hmm_posterior_probability.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: data/hmm_posterior_probability.png")
plt.show()

# ============================================================================
# PART 6: MONTE CARLO EVALUATION OF SAMPLING STRATEGIES
# ============================================================================
print("\n[6] Running Monte Carlo simulations to compare sampling strategies...")

sampling_frequencies = [1, 2, 5, 10, 20]  # Test 1/N items
results = SamplingStrategyEvaluator.monte_carlo_evaluation(
    simulator=simulator,
    sampling_frequencies=sampling_frequencies,
    n_simulations=100,
    T=1000
)

# Print results
print("\n" + "=" * 80)
print("SAMPLING STRATEGY EVALUATION RESULTS")
print("=" * 80)

for freq in sampling_frequencies:
    res = results[freq]
    print(f"\nStrategy: Test every {freq} items (1/{freq})")
    print(f"  Detection Delay:")
    if res['avg_detection_delay']:
        print(f"    Avg: {res['avg_detection_delay']:.1f} ± {res['std_detection_delay']:.1f} items")
    else:
        print(f"    Not computed (no bad runs detected)")
    print(f"  Undetected Bad Fraction: {res['avg_undetected_fraction']:.3f} ± {res['std_undetected_fraction']:.3f}")
    print(f"  Avg False Negatives: {res['avg_false_negatives']:.1f}")
    print(f"  Avg False Positives: {res['avg_false_positives']:.1f}")

# ============================================================================
# PART 7: VISUALIZE STRATEGY COMPARISON
# ============================================================================
print("\n[7] Visualizing strategy comparison...")

fig = plot_strategy_comparison(results, title="Sampling Strategy Comparison (Monte Carlo)")
plt.savefig('data/hmm_strategy_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: data/hmm_strategy_comparison.png")
plt.show()

# ============================================================================
# PART 8: APPLY TO YOUR ACTUAL DATA
# ============================================================================
print("\n[8] Applying HMM to your actual electrode data...")

# Import your actual data
import pandas as pd

try:
    # Load your electrode analysis dataframe
    electrode_df = pd.read_csv('../data/electrode_test_results.csv')
    
    # Convert to HMM observation format
    # 1 = Released (Good), 0 = Scraped (Bad), None = Not tested
    actual_observations = []
    for idx, row in electrode_df.iterrows():
        if row['Tested']:
            actual_observations.append(1 if row['Released'] else 0)
        else:
            actual_observations.append(None)
    
    print(f"Loaded {len(actual_observations)} electrode observations")
    print(f"  Tested: {sum(1 for o in actual_observations if o is not None)}")
    print(f"  Untested: {sum(1 for o in actual_observations if o is None)}")
    
    # Train HMM on actual data
    hmm_actual = HiddenMarkovModel(n_states=2, random_state=42)
    history = hmm_actual.baum_welch(actual_observations, n_iterations=50, tolerance=1e-6)
    
    print("\nLearned parameters from your data:")
    print(f"  Transition matrix:\n{hmm_actual.transition_matrix}")
    print(f"  Emission matrix:\n{hmm_actual.emission_matrix}")
    
    # Get posterior probabilities
    posterior_actual = hmm_actual.posterior_state_probability(actual_observations)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), facecolor='white')
    
    # Plot posterior probability of bad state
    prob_bad_actual = posterior_actual[:, 1]
    ax1.plot(prob_bad_actual, linewidth=1, color='red', alpha=0.7)
    ax1.fill_between(range(len(prob_bad_actual)), 0, prob_bad_actual, alpha=0.2, color='red')
    ax1.set_ylabel('P(Bad State | Data)', fontsize=11, fontweight='bold')
    ax1.set_title('Hidden Markov Model: Posterior Probability of Bad State on Actual Data', 
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot detected defects
    detected_bad = [i for i, o in enumerate(actual_observations) if o == 0]
    if detected_bad:
        ax2.scatter(detected_bad, [0]*len(detected_bad), c='red', s=50, alpha=0.6, 
                   label='Detected Bad (Tested)', edgecolors='black')
    
    ax2.set_ylabel('Detection', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Electrode Index', fontsize=11, fontweight='bold')
    ax2.set_title('Detected Defects vs. Hidden State Probability', fontsize=12, fontweight='bold')
    ax2.set_ylim(-0.5, 1.5)
    ax2.grid(True, alpha=0.3)
    if detected_bad:
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig('data/hmm_actual_data_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Saved: data/hmm_actual_data_analysis.png")
    plt.show()
    
except Exception as e:
    print(f"⚠️ Could not apply to actual data: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
The Hidden Markov Model reveals:

1. **Hidden State Dynamics**: The chemistry transitions between Good and Bad states
   with learned probabilities.

2. **Emission Errors**: Observations contain measurement/sampling errors that the
   HMM learns to account for.

3. **Inference Power**: Using forward-backward algorithm, we compute the
   posterior probability of being in a Bad state even for untested items.

4. **Strategy Evaluation**: Monte Carlo simulations show that:
   - More frequent testing reduces undetected bad items
   - But has diminishing returns vs. cost
   - Optimal frequency depends on transition probabilities and risk tolerance

5. **Your ±2 Safety Margin Strategy**: Can be evaluated against this Bayesian
   framework to quantify its effectiveness statistically.

Files saved:
  - data/hmm_ground_truth.png: Ground truth vs observations
  - data/hmm_training_history.png: Training convergence
  - data/hmm_posterior_probability.png: Posterior probabilities
  - data/hmm_strategy_comparison.png: Strategy comparison
  - data/hmm_actual_data_analysis.png: Results on your actual data
""")
print("=" * 80)

