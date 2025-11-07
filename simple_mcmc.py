import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class SimpleMCMC:
    """
    Simple MCMC sampler using Metropolis-Hastings algorithm
    """
    
    def __init__(self, target_mean=0, target_std=1, proposal_std=0.5):
        """
        Initialize the MCMC sampler
        
        Args:
            target_mean: Mean of the target normal distribution
            target_std: Standard deviation of the target distribution
            proposal_std: Standard deviation of the proposal distribution
        """
        self.target_mean = target_mean
        self.target_std = target_std
        self.proposal_std = proposal_std
        self.samples = []
        self.accepted = 0
        self.total_proposals = 0
    
    def log_target_density(self, x):
        """
        Log density of the target distribution (normal distribution)
        """
        return norm.logpdf(x, loc=self.target_mean, scale=self.target_std)
    
    def propose_next_state(self, current_state):
        """
        Propose next state using normal random walk
        """
        return current_state + np.random.normal(0, self.proposal_std)
    
    def metropolis_hastings_step(self, current_state):
        """
        Single Metropolis-Hastings step
        """
        # Propose new state
        proposed_state = self.propose_next_state(current_state)
        
        # Calculate acceptance probability
        log_alpha = (self.log_target_density(proposed_state) - 
                    self.log_target_density(current_state))
        alpha = min(1, np.exp(log_alpha))
        
        # Accept or reject
        self.total_proposals += 1
        if np.random.random() < alpha:
            self.accepted += 1
            return proposed_state
        else:
            return current_state
    
    def sample(self, n_samples=1000, initial_state=0):
        """
        Run MCMC sampling
        
        Args:
            n_samples: Number of samples to generate
            initial_state: Starting value for the chain
        """
        self.samples = []
        self.accepted = 0
        self.total_proposals = 0
        
        current_state = initial_state
        
        for i in range(n_samples):
            current_state = self.metropolis_hastings_step(current_state)
            self.samples.append(current_state)
        
        self.samples = np.array(self.samples)
        return self.samples
    
    def get_acceptance_rate(self):
        """
        Calculate acceptance rate
        """
        if self.total_proposals == 0:
            return 0
        return self.accepted / self.total_proposals
    
    def plot_results(self, burn_in=100):
        """
        Plot the MCMC results
        """
        if len(self.samples) == 0:
            print("No samples available. Run sample() first.")
            return
        
        # Remove burn-in samples
        samples_after_burnin = self.samples[burn_in:]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Trace plot
        axes[0, 0].plot(self.samples)
        axes[0, 0].axvline(burn_in, color='red', linestyle='--', label='Burn-in')
        axes[0, 0].set_title('Trace Plot')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        
        # Histogram of samples (after burn-in)
        axes[0, 1].hist(samples_after_burnin, bins=30, density=True, alpha=0.7, 
                       label='MCMC samples')
        
        # True distribution
        x_range = np.linspace(samples_after_burnin.min() - 1, 
                             samples_after_burnin.max() + 1, 100)
        true_density = norm.pdf(x_range, self.target_mean, self.target_std)
        axes[0, 1].plot(x_range, true_density, 'r-', linewidth=2, 
                       label='True distribution')
        axes[0, 1].set_title('Histogram vs True Distribution')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        
        # Running mean
        running_mean = np.cumsum(self.samples) / np.arange(1, len(self.samples) + 1)
        axes[1, 0].plot(running_mean)
        axes[1, 0].axhline(self.target_mean, color='red', linestyle='--', 
                          label=f'True mean ({self.target_mean})')
        axes[1, 0].axvline(burn_in, color='green', linestyle='--', label='Burn-in')
        axes[1, 0].set_title('Running Mean')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Running Mean')
        axes[1, 0].legend()
        
        # Autocorrelation
        def autocorr(x, max_lag=50):
            x = x - np.mean(x)
            autocorr_result = np.correlate(x, x, mode='full')
            autocorr_result = autocorr_result[autocorr_result.size // 2:]
            autocorr_result = autocorr_result / autocorr_result[0]
            return autocorr_result[:max_lag]
        
        lags = range(50)
        autocorr_values = autocorr(samples_after_burnin)
        axes[1, 1].plot(lags, autocorr_values)
        axes[1, 1].axhline(0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].set_title('Autocorrelation')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\nMCMC Summary:")
        print(f"Total samples: {len(self.samples)}")
        print(f"Samples after burn-in: {len(samples_after_burnin)}")
        print(f"Acceptance rate: {self.get_acceptance_rate():.2%}")
        print(f"Sample mean: {np.mean(samples_after_burnin):.3f} (true: {self.target_mean})")
        print(f"Sample std: {np.std(samples_after_burnin):.3f} (true: {self.target_std})")


def main():
    """
    Demonstrate the MCMC sampler
    """
    print("Simple MCMC Demo - Sampling from Normal Distribution")
    print("=" * 50)
    
    # Create MCMC sampler
    # Target: Normal(mean=2, std=1.5)
    mcmc = SimpleMCMC(target_mean=2, target_std=1.5, proposal_std=0.8)
    
    # Run sampling
    print("Running MCMC sampling...")
    samples = mcmc.sample(n_samples=5000, initial_state=0)
    
    # Show results
    mcmc.plot_results(burn_in=500)
    
    # Example of different proposal standard deviations
    print("\nComparing different proposal standard deviations:")
    print("-" * 50)
    
    proposal_stds = [0.1, 0.5, 1.0, 2.0]
    
    for prop_std in proposal_stds:
        mcmc_test = SimpleMCMC(target_mean=0, target_std=1, proposal_std=prop_std)
        mcmc_test.sample(n_samples=1000)
        acceptance_rate = mcmc_test.get_acceptance_rate()
        print(f"Proposal std: {prop_std:3.1f} -> Acceptance rate: {acceptance_rate:.2%}")


if __name__ == "__main__":
    main() 