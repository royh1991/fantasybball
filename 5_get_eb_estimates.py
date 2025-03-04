import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist

class EmpiricalBayesBasketball:
    def __init__(self, data):
        self.data = data
        self.priors = {}

    def aggregate_data(self):
        """Aggregate game logs by player, position, and season."""
        self.data['season_start'] = pd.to_datetime(self.data['DATE'], errors='coerce').apply(
            lambda x: x.year if x.month >= 10 else x.year - 1
        )
        self.data['season_end'] = self.data['season_start'] + 1

        # Ensure relevant stats are numeric
        stats = ['FG', 'FGA', 'FT', 'FTA', '3P', '3PA']
        for stat in stats:
            self.data[stat] = pd.to_numeric(self.data[stat], errors='coerce')

        # Aggregate by player, position, and season
        self.aggregated_data = self.data.groupby(['NAME', 'Position']).agg(
            total_games=('DATE', 'count'),
            total_FG=('FG', 'sum'),
            total_FGA=('FGA', 'sum'),
            total_FT=('FT', 'sum'),
            total_FTA=('FTA', 'sum'),
            total_3PM=('3P', 'sum'),
            total_3PA=('3PA', 'sum')
        ).reset_index()

    def fit_beta_mle(self, data):
        """Fit a Beta distribution using MLE."""
        def beta_log_likelihood(params, data):
            alpha, beta_param = params
            return -np.sum(beta_dist.logpdf(data, alpha, beta_param))

        initial_guess = [1.0, 1.0]
        result = minimize(
            beta_log_likelihood, initial_guess, args=(data,),
            bounds=[(1e-6, None), (1e-6, None)], method='L-BFGS-B'
        )
        return result.x

    def calculate_recent_weighted_averages(self, player_name, stat, stat_attempts):
        """Calculate weighted averages for makes and attempts based on the 10 most recent games with decay factor 0.9."""
        player_data = self.data[self.data['NAME'] == player_name].sort_values(by='DATE', ascending=False)
        recent_games = player_data.head(10)  # Get the 10 most recent games

        # Apply a decay factor for recent games
        decay_factor = 0.9
        weights = [decay_factor ** i for i in range(len(recent_games))]
        
        # Calculate weighted attempts and makes for recent games
        weighted_makes = np.sum(recent_games[stat] * weights) / np.sum(weights)
        weighted_attempts = np.sum(recent_games[stat_attempts] * weights) / np.sum(weights)
        return weighted_makes, weighted_attempts

    def fit_priors(self):
        """Fit priors (alpha, beta) for percentage stats by position."""
        percentage_stats = {
            'FG%': ('total_FG', 'total_FGA'),
            'FT%': ('total_FT', 'total_FTA'),
            '3P%': ('total_3PM', 'total_3PA')
        }

        # Fit Beta priors for percentage stats
        for stat, (makes, attempts) in percentage_stats.items():
            for position in self.aggregated_data['Position'].unique():
                pos_data = self.aggregated_data[self.aggregated_data['Position'] == position]

                # Filter out rows with zero attempts
                pos_data = pos_data[pos_data[attempts] > 0]

                # Calculate the percentage and clip values to avoid exact 0 and 1
                data = (pos_data[makes] / (pos_data[attempts] + 1e-6)).clip(1e-6, 1 - 1e-6)

                # Fit Beta distribution parameters
                alpha, beta = self.fit_beta_mle(data.values)
                self.priors[(stat, position)] = (alpha, beta)
                print(f"Fitted Beta prior for {stat} ({position}): alpha={alpha}, beta={beta}")

    def calculate_eb_estimates(self):
        """Calculate Empirical Bayes estimates, incorporating recent game data with decay."""
        results = []

        for _, row in self.aggregated_data.iterrows():
            player_results = {'NAME': row['NAME'], 'Position': row['Position']}
            position = row['Position']

            # Loop over priors (stat, position) pairs
            for (stat_name, pos), (alpha, beta) in self.priors.items():
                if pos == position:  # Ensure matching position
                    stat_name_clean = stat_name.replace('%', '')  # Clean stat name

                    # Access columns for recent games and calculate weighted averages
                    makes, attempts = 0, 0
                    if stat_name_clean == 'FG':
                        makes, attempts = self.calculate_recent_weighted_averages(row['NAME'], 'FG', 'FGA')
                    elif stat_name_clean == '3P':
                        makes, attempts = self.calculate_recent_weighted_averages(row['NAME'], '3P', '3PA')
                    elif stat_name_clean == 'FT':
                        makes, attempts = self.calculate_recent_weighted_averages(row['NAME'], 'FT', 'FTA')

                    # Handle zero attempts to avoid division by zero
                    if attempts == 0:
                        eb_estimate = 0
                    else:
                        # Calculate EB estimate with recent weighted makes and attempts
                        eb_estimate = (makes + alpha) / (attempts + alpha + beta)

                    # Store only EB estimate, alpha, and beta
                    player_results[f'eb_{stat_name_clean}_estimate'] = eb_estimate
                    player_results[f'{stat_name_clean}_alpha'] = alpha
                    player_results[f'{stat_name_clean}_beta'] = beta

            results.append(player_results)

        return pd.DataFrame(results)


# Example Usage
eb_model = EmpiricalBayesBasketball(combined_data_with_position)
eb_model.aggregate_data()
eb_model.fit_priors()
estimates = eb_model.calculate_eb_estimates()

print(estimates)
estimates.to_csv("/Users/rhu/fantasybasketball2/data/intermediate/estimates.csv")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

class PlayerAttemptModel:
    def __init__(self, data, estimates):
        self.data = data
        self.estimates = estimates  # Store Empirical Bayes estimates for each player
        self.player_params = {}  # Store fitted parameters for each player

    def negative_binomial_nll(self, params, data):
        """Negative log-likelihood for the Negative Binomial."""
        mu, r = params
        if mu <= 0 or r <= 0:
            return np.inf  # Invalid parameter values
        nll = -np.sum(
            gammaln(data + r) - gammaln(r) - gammaln(data + 1) +
            r * np.log(r / (r + mu)) + data * np.log(mu / (r + mu))
        )
        return nll

    def simulate_player_attempts(self, player_name, stat, num_games=1):
        """Simulate attempts for a specific player and stat."""
        player_data = self.data[self.data['NAME'] == player_name][stat].values
        initial_guess = [np.mean(player_data), 1.0]
        result = minimize(
            self.negative_binomial_nll, initial_guess, args=(player_data,),
            method='L-BFGS-B', bounds=[(1e-6, None), (1e-6, None)]
        )
        mu, r = result.x
        self.player_params[player_name] = (mu, r)
        p = r / (r + mu)
        simulated_attempts = np.random.negative_binomial(r, p, size=num_games)
        return simulated_attempts[0]

    def calculate_game_log_variability(self, player_name, stat_prefix):
        """Calculate the standard deviation of game-to-game shooting percentage."""
        player_data = self.data[self.data['NAME'] == player_name]
        game_percentages = player_data[f'{stat_prefix}'] / player_data[f'{stat_prefix}A']
        return game_percentages.std()

    def simulate_makes(self, player_name, stat_attempts, stat_prefix):
        """Simulate makes based on attempts and EB estimate with added game-to-game variability."""
        player_estimates = self.estimates[self.estimates['NAME'] == player_name].iloc[0]
        eb_estimate = player_estimates[f'eb_{stat_prefix}_estimate']

        # Calculate game-to-game variability in observed shooting percentages
        variability = self.calculate_game_log_variability(player_name, stat_prefix)

        # Draw a shooting percentage around the EB estimate with added variability
        game_shooting_percentage = np.clip(
            np.random.normal(eb_estimate, variability), 0, 1
        )
        
        # Simulate the number of makes based on drawn shooting percentage
        makes = round(stat_attempts * game_shooting_percentage)
        return makes, game_shooting_percentage

    def simulate_game(self, player_name):
        """Simulate a game for a player across fantasy categories."""
        # Field Goals
        fga_attempts = self.simulate_player_attempts(player_name, 'FGA')
        fgm, fg_percentage = self.simulate_makes(player_name, fga_attempts, "FG")
        
        # Three-Point Attempts
        threepa_attempts = self.simulate_player_attempts(player_name, '3PA')
        threepa_attempts = min(threepa_attempts, fga_attempts)
        threepm, threep_percentage = self.simulate_makes(player_name, threepa_attempts, "3P")
        
        # 2-Point Makes and Total Points
        twop_makes = round(fgm - threepm)
        points = (twop_makes * 2) + (threepm * 3)

        # Free Throw Attempts (FTA)
        fta_attempts = self.simulate_player_attempts(player_name, 'FTA')
        ftm, ft_percentage = self.simulate_makes(player_name, fta_attempts, "FT")
        points += ftm  # Add FT points (1 point per make) to total points

        # Rebounds (TRB)
        rebounds = self.simulate_player_attempts(player_name, 'TRB')

        # Blocks (BLK)
        blocks = self.simulate_player_attempts(player_name, 'BLK')

        # Steals (STL)
        steals = self.simulate_player_attempts(player_name, 'STL')

        # Turnovers (TOV)
        turnovers = self.simulate_player_attempts(player_name, 'TOV')

        # Assists (AST)
        assists = self.simulate_player_attempts(player_name, 'AST')

        # Double Doubles (includes TRB in calculation)
        double_double = int(
            sum(stat >= 10 for stat in [points, rebounds, assists, steals, blocks]) >= 2
        )

        # Compile results into a DataFrame
        results = {
            'NAME': player_name,
            'FGA': fga_attempts,
            'FGM': fgm,
            'FG%': fg_percentage,
            '3PA': threepa_attempts,
            '3PM': threepm,
            '3P%': threep_percentage,
            'FTA': fta_attempts,
            'FTM': ftm,
            'FT%': ft_percentage,
            'Points': points,
            'TRB': rebounds,
            'BLK': blocks,
            'STL': steals,
            'TOV': turnovers,
            'AST': assists,
            'DoubleDouble': double_double
        }
        
        return pd.DataFrame([results])

# Example Usage
model = PlayerAttemptModel(combined_data_with_position, estimates)
# Simulate 82 games for Stephen Curry
simulated_games = [model.simulate_game("Stephen Curry") for _ in range(82)]
simulated_games_df = pd.concat(simulated_games, ignore_index=True)

# Calculate the average stats for the season
average_stats = simulated_games_df.select_dtypes(include=np.number).mean()
average_stats



import matplotlib.pyplot as plt

# Plot the distribution of simulated 3P%
plt.figure(figsize=(10, 6))
plt.hist(simulated_games_df['FG%'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Distribution of Simulated 3P% for Stephen Curry")
plt.xlabel("FG%")
plt.ylabel("Frequency")
plt.axvline(simulated_games_df['3P%'].mean(), color='red', linestyle='dashed', linewidth=1)
plt.text(simulated_games_df['3P%'].mean() * 1.05, plt.ylim()[1] * 0.9, f"Mean: {simulated_games_df['3P%'].mean():.2f}", color='red')
plt.show()













































#### OLD STUFF 
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist, betabinom

class EmpiricalBayesBasketball:
    def __init__(self, data):
        self.data = data
        self.priors = {}

    def aggregate_data(self):
        """Aggregate game logs by player, position, and season."""
        self.data['season_start'] = pd.to_datetime(self.data['DATE'], errors='coerce').apply(
            lambda x: x.year if x.month >= 10 else x.year - 1
        )
        self.data['season_end'] = self.data['season_start'] + 1

        # Ensure relevant stats are numeric
        stats = ['FG', 'FGA', 'FT', 'FTA', '3P', '3PA', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']
        for stat in stats:
            self.data[stat] = pd.to_numeric(self.data[stat], errors='coerce')

        # Calculate Double-Doubles (games with 2 categories >= 10)
        self.data['DD'] = (self.data[['PTS', 'TRB', 'AST']] >= 10).sum(axis=1) >= 2

        # Aggregate by player, position, and season
        self.aggregated_data = self.data.groupby(['NAME', 'Position']).agg(
            total_games=('DATE', 'count'),
            total_FG=('FG', 'sum'),
            total_FGA=('FGA', 'sum'),
            total_FT=('FT', 'sum'),
            total_FTA=('FTA', 'sum'),
            total_3PM=('3P', 'sum'),
            total_3PA=('3PA', 'sum'),
            total_TRB=('TRB', 'sum'),
            total_AST=('AST', 'sum'),
            total_STL=('STL', 'sum'),
            total_BLK=('BLK', 'sum'),
            total_TOV=('TOV', 'sum'),
            total_PTS=('PTS', 'sum'),
            total_DD=('DD', 'sum')
        ).reset_index()

    def fit_beta_mle(self, data):
        """Fit a Beta distribution using MLE."""
        def beta_log_likelihood(params, data):
            alpha, beta_param = params
            return -np.sum(beta_dist.logpdf(data, alpha, beta_param))

        initial_guess = [1.0, 1.0]
        result = minimize(
            beta_log_likelihood, initial_guess, args=(data,),
            bounds=[(1e-6, None), (1e-6, None)], method='L-BFGS-B'
        )
        return result.x

    def fit_priors(self):
        """Fit priors (alpha, beta) for percentages and counts by position."""
        percentage_stats = {
            'FG%': ('total_FG', 'total_FGA'),
            'FT%': ('total_FT', 'total_FTA'),
            '3P%': ('total_3PM', 'total_3PA')
        }
        count_stats = ['3PM', 'TRB', 'FG', 'FGA', 'FT', 'FTA', '3PA', 'AST', 'STL', 'BLK', 'TOV', 'PTS', 'DD']

        # Fit Beta priors for percentage stats
        for stat, (makes, attempts) in percentage_stats.items():
            for position in self.aggregated_data['Position'].unique():
                pos_data = self.aggregated_data[self.aggregated_data['Position'] == position]
                data = pos_data[makes] / (pos_data[attempts] + 1e-6)
                data = data.fillna(0)

                alpha, beta = self.fit_beta_mle(data.values)
                self.priors[(stat, position)] = (alpha, beta)
                print(f"Fitted Beta prior for {stat} ({position}): alpha={alpha}, beta={beta}")

        # Fit Beta-Binomial priors for count stats
        for stat in count_stats:
            for position in self.aggregated_data['Position'].unique():
                pos_data = self.aggregated_data[self.aggregated_data['Position'] == position]
                successes = pos_data[f'total_{stat}']
                n = pos_data['total_games']

                alpha, beta = self.fit_beta_binomial_prior(successes, n)
                self.priors[(stat, position)] = (alpha, beta)
                print(f"Fitted Beta-Binomial prior for {stat} ({position}): alpha={alpha}, beta={beta}")

    def fit_beta_binomial_prior(self, successes, n):
        """Fit a Beta-Binomial distribution."""
        def negative_log_likelihood(params):
            alpha, beta = params
            ll = betabinom.logpmf(successes, n, alpha, beta)
            return -ll.sum()

        initial_guess = [1.0, 1.0]
        result = minimize(negative_log_likelihood, initial_guess, method='Nelder-Mead')
        return result.x

    def calculate_eb_estimates(self):
        """Calculate Empirical Bayes estimates for all stats."""
        results = []

        for _, row in self.aggregated_data.iterrows():
            player_results = {'NAME': row['NAME'], 'Position': row['Position']}
            n = row.get('total_games', 0)

            for stat in self.priors:
                if stat[1] == row['Position']:
                    alpha, beta = self.priors[stat]

                    if stat[0] in ['FG%', 'FT%', '3P%']:
                        stat_name = stat[0].replace('%', '')
                        makes = row.get(f'total_{stat_name}', 0)
                        attempts = row.get(f'total_{stat_name}A', 0)
                        eb_estimate = (makes + alpha) / (attempts + alpha + beta)

                        variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
                    else:
                        total_stat = row.get(f'total_{stat[0]}', 0)
                        eb_estimate = (total_stat + alpha) / (n + alpha + beta)

                        variance = (n * alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

                    player_results[f'eb_{stat[0]}_estimate'] = eb_estimate
                    player_results[f'{stat[0]}_variance'] = variance
                    player_results[f'{stat[0]}_game_sd'] = np.sqrt(variance)

            results.append(player_results)

        return pd.DataFrame(results)

# Example Usage
eb_model = EmpiricalBayesBasketball(combined_data_with_position)
eb_model.aggregate_data()
eb_model.fit_priors()
estimates = eb_model.calculate_eb_estimates()
print(estimates)

estimates.to_csv("estimates.csv")





game_logs = combined_data_with_position
# Extract the minutes as an integer from the 'MP' column
game_logs['MP'] = game_logs['MP'].astype(str)
game_logs['MP'] = game_logs['MP'].str.split(':').str[0].astype(int)

# Now categorize players based on minutes played
game_logs['Minutes_Category'] = np.where(game_logs['MP'] >= 25, '>=25 Minutes', '<25 Minutes')
game_logs['FG%_game'] = game_logs['FG'] / (game_logs['FGA'] + 1e-6)

# Calculate the standard deviation of FG% per player and position
player_sd = game_logs.groupby(['NAME', 'Position'])['FG%_game'].std().reset_index()
print(player_sd.head())



# Group by Position and Minutes Category, calculate the standard deviation of FG%
player_sd_by_minutes = (
    game_logs.groupby(['Position', 'Minutes_Category'])['FG%_game']
    .std()
    .reset_index()
    .pivot(index='Position', columns='Minutes_Category', values='FG%_game')
    .reset_index()
)

print(player_sd_by_minutes)






import numpy as np
from scipy.stats import beta, poisson, binom
import numpy as np
import matplotlib.pyplot as plt

# Historical data for Aaron Gordon
avg_FGA = 5.68  # Average FG attempts per game
alpha, beta_param = 88.93, 106.72  # Beta distribution parameters for FG%

fg = []
def simulate_game(player):
    """Simulate a game for a player."""
    df = estimates['NAME'] == player
    # Step 1: Simulate FG Attempts
    estimates['eb_FG_estimate']
    fga = poisson.rvs(avg_FGA)

    # Step 2: Simulate FG% using a Beta distribution
    fg_pct = beta.rvs(alpha, beta_param)

    # Step 3: Simulate FG Makes using a Binomial distribution
    fg_makes = binom.rvs(fga, fg_pct)

    # Calculate the actual shooting percentage for this game
    actual_fg_pct = fg_makes / fga if fga > 0 else 0

    return fga, fg_pct, fg_makes, actual_fg_pct

# Run a simulation
for i in range(0, 400):
    fga, fg_pct, fg_makes, actual_fg_pct = simulate_game()
    print(f"Simulated Game - FG Attempts: {fga}, FG% Sample: {fg_pct:.2%}, "
        f"FG Makes: {fg_makes}, Actual FG%: {actual_fg_pct:.2%}")
    print("Points score was:", fg_makes * 2)
    fg.append(fga)


real_data = np.array([8, 10, 6, 13, 10, 15, 13, 17, 11, 10, 12, 15, 16, 3, 11, 11, 
                      15, 15, 13, 12, 11, 15, 13, 9, 14, 17, 14, 16, 9, 14, 11, 
                      10, 9, 16, 12, 11, 10, 9, 13, 11, 12, 15, 20, 12, 15, 14, 
                      12, 20, 15, 13, 16, 14, 13, 5, 10, 15, 17, 8, 5, 12, 15, 
                      9, 11, 12, 4, 10, 8, 15, 17, 7, 16, 14, 12, 11, 18, 9, 10, 
                      9, 9, 13, 8, 8, 11, 20, 13, 13, 15, 9, 4, 13, 13, 4, 8, 11, 
                      5, 5, 11, 6, 7, 11, 11, 7, 10, 7, 9, 5, 7, 6, 9, 14, 10, 4, 
                      17, 12, 9, 16, 8, 4, 11, 13, 6, 5, 8, 10, 6, 10, 16, 16, 3, 
                      12, 12, 10, 13, 12, 13, 14, 13, 9, 16, 9, 18, 8, 13, 18, 7, 
                      8, 8, 4, 8, 13, 8, 9, 15, 15, 7, 24, 10, 12, 10, 11, 21, 9, 
                      7, 10, 12, 14, 8, 9, 13, 9, 12, 9, 6, 9, 14, 8, 23, 18, 11, 
                      10, 11, 11, 7, 12, 3, 17, 9, 11, 14, 10, 12, 14, 11, 11, 16, 
                      12, 13, 9, 5, 10, 13, 9, 10, 9, 9, 6, 11, 10, 5, 17, 8, 4, 
                      6, 12, 10, 11, 6, 10, 5, 10, 8, 6, 16, 9, 8, 8, 8, 11, 15, 
                      12, 5, 10, 8, 8, 16, 10, 15, 3, 9, 11, 3, 7, 10, 9, 21, 14, 
                      9, 7, 12, 10, 7, 6, 6, 6, 6, 14, 13, 9, 10, 12, 5, 7, 11, 
                      10, 11, 12, 12, 13, 12, 14, 9, 10, 11, 12, 4, 16, 7, 12, 13, 
                      14, 6, 8, 11, 15, 9, 11, 7, 9, 16, 5, 9, 15, 16, 11, 8, 11, 
                      11, 18, 17, 9, 14, 12, 15, 9, 6, 14, 10, 12, 12, 12, 9, 17, 
                      4, 10, 11, 6, 13, 15, 5, 13, 16, 17, 16, 7, 11, 14, 16, 11, 
                      18, 9, 12, 11, 11, 15, 15, 5, 20, 15, 15, 14, 13, 13, 13, 
                      17, 15, 11, 14, 6, 12, 16, 19, 18, 15, 11, 7, 10, 17, 7, 10, 
                      6, 15, 17, 14, 8, 17, 13, 17, 14, 15, 11, 13, 9, 16, 18, 19, 
                      9, 17, 11, 13, 15, 9, 10, 17, 14, 11, 16, 15, 15, 16, 18, 15, 
                      17, 13, 16, 16, 15, 8, 12, 17, 12, 15, 7, 14, 6, 13, 18])


plt.figure(figsize=(12, 6))
plt.hist(fg, bins=20, alpha=0.5, label='Simulated FG Makes', color='b')
plt.hist(a.FGA.values, bins=20, alpha=0.5, label='Real Data', color='g')
plt.xlabel('Field Goals Made')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Comparison of Simulated vs Real FG Makes')
plt.show()

# Call the aggregate_data method to ensure it's populated
eb_model.aggregate_data()

# Now print the aggregated data
print(eb_model.aggregated_data)






# Step 3: Plot both simulated and actual FG data
plt.figure(figsize=(12, 6))

# Plot simulated Beta-Binomial data
plt.hist(poisson.rvs(mu=5.6, size=400), bins=10, alpha=0.6, label='Simulated FG (Beta-Binomial)', color='blue')
# Plot actual FG data for Aaron Gordon
plt.hist(FGA, bins=10, alpha=0.6, label='Simulated FG (Beta-Binomial)', color='blue')

plt.xlabel('Field Goals Made')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Simulated vs Actual FG for Aaron Gordon')

# Display the plot
plt.show()




import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

# Sample data: Replace with your actual FGA data
data = combined_data_with_position[combined_data_with_position.NAME == 'Alex Caruso']['FGA']

# Negative log-likelihood function for Negative Binomial
def negbinom_nll(params, data):
    mu, r = params
    if mu <= 0 or r <= 0:
        return np.inf  # Avoid invalid parameter values
    
    # Negative log-likelihood for NB distribution
    nll = -np.sum(
        gammaln(data + r) - gammaln(r) - gammaln(data + 1) +
        r * np.log(r / (r + mu)) + data * np.log(mu / (r + mu))
    )
    return nll

# Initial guess for mu and r
initial_guess = [np.mean(data), 1]

# Minimize the negative log-likelihood to estimate parameters
result = minimize(negbinom_nll, initial_guess, args=(data,), method='L-BFGS-B', bounds=[(1e-6, None), (1e-6, None)])

mu_nb, r_nb = result.x  # Extract estimated parameters

print(f"Estimated Negative Binomial mean (mu): {mu_nb}")
print(f"Estimated Negative Binomial dispersion (r): {r_nb}")

# Simulate from the fitted Negative Binomial model
from scipy.stats import nbinom
p = r_nb / (r_nb + mu_nb)  # Success probability
simulated_nb = nbinom.rvs(r_nb, p, size=1000)

print(f"Simulated NB Mean: {np.mean(simulated_nb)}")
print(f"Simulated NB Variance: {np.var(simulated_nb)}")

















import numpy as np
import pandas as pd
from scipy.stats import beta, betabinom, binom
import matplotlib.pyplot as plt

class BasketballSimulator:
    def __init__(self, eb_model, data, estimates):
        self.eb_model = eb_model  # Empirical Bayes model with priors
        self.data = data  # Actual combined dataset with player stats
        self.estimates = estimates

    def get_priors(self, stat, position):
        """Retrieve priors for a given stat and position."""
        return self.eb_model.priors.get((stat, position), (1.0, 1.0))

    def simulate_game(self, player_name, player_position):
        """Simulate a single game for a player using Empirical Bayes."""
        # Step 1: Get priors for FG% and FG attempts
        alpha_fg_pct, beta_fg_pct = self.get_priors('FG%', player_position)
        alpha_fga, beta_fga = self.get_priors('FG', player_position)

        # Step 2: Draw FGA from a Beta-Binomial distribution
        mean_fga = self.estimates[self.estimates['NAME'] == player_name]['eb_FGA_estimate'].values[0]
        fga = betabinom.rvs(round(mean_fga), alpha_fga, beta_fga)

        # Step 3: Draw FG% from the Beta posterior
        fg_pct = beta.rvs(alpha_fg_pct, beta_fg_pct)

        # Step 4: Simulate FG makes using a Binomial distribution
        if fga > 0:
            fg_makes = binom.rvs(fga, fg_pct)
        else:
            fg_makes = 0

        # Calculate actual FG% for the game
        actual_fg_pct = fg_makes / fga if fga > 0 else 0

        return fga, fg_pct, fg_makes, actual_fg_pct

    def run_simulation(self, player_name, player_position, num_games=400):
        """Run simulations for the specified player."""
        fg_attempts = []
        fg_makes_list = []

        for _ in range(num_games):
            fga, fg_pct, fg_makes, actual_fg_pct = self.simulate_game(player_name, player_position)
            print(f"Simulated Game - FG Attempts: {fga}, FG% Sample: {fg_pct:.2%}, "
                  f"FG Makes: {fg_makes}, Actual FG%: {actual_fg_pct:.2%}")
            print(f"Points scored: {fg_makes * 2}")

            fg_attempts.append(fga)
            fg_makes_list.append(fg_makes)

        self.plot_results(player_name, fg_makes_list)

    def plot_results(self, player_name, simulated_fg):
        """Plot the simulated FG vs. actual FG from the data."""
        # Get actual FG values for the player from the dataset
        actual_fg = self.data[self.data['NAME'] == player_name]['FG'].values

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.hist(simulated_fg, bins=10, alpha=0.6, label='Simulated FG Makes')
        plt.hist(actual_fg, bins=10, alpha=0.6, label='Actual FG Makes')
        plt.xlabel('Field Goals Made')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.title(f'Simulated vs. Actual FG Makes for {player_name}')
        plt.show()


# Example Usage
# Assuming eb_model and combined_data_with_position are already defined
simulator = BasketballSimulator(eb_model, combined_data_with_position, estimates)

# Specify the player and their position
player_name = "Aaron Gordon"
player_position = "F"  # Example position

# Run the simulation
simulator.run_simulation(player_name, player_position)


combined_data_with_position[combined_data_with_position.NAME=='Aaron Gordon']['FG']

combined_data_with_position[combined_data_with_position['NAME']=='Aaron Gordon']['FGA'].values








import numpy as np
import pandas as pd
from scipy.stats import betabinom
import matplotlib.pyplot as plt

# Step 1: Simulate 1000 Beta-Binomial samples
n_trials = 11
alpha, beta = eb_model.priors.get(('FG', 'G'), (1.0, 1.0))
num_simulations = 400

simulated_data = betabinom.rvs(n_trials, alpha, beta, size=num_simulations)

# Filter Aaron Gordon's actual FG data
actual_fg = combined_data_with_position[combined_data_with_position.NAME == 'Alex Caruso']['FG']

# Step 3: Plot both simulated and actual FG data
plt.figure(figsize=(12, 6))

# Plot simulated Beta-Binomial data
plt.hist(simulated_data, bins=10, alpha=0.6, label='Simulated FG (Beta-Binomial)', color='blue')

# Plot actual FG data for Aaron Gordon
plt.hist(actual_fg, bins=range(0, max(actual_fg) + 2), alpha=0.6, label='Actual FG', color='green')

plt.xlabel('Field Goals Made')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Simulated vs Actual FG for Aaron Gordon')

# Display the plot
plt.show()




#### START HERE #####

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist

class EmpiricalBayesBasketball:
    def __init__(self, data):
        self.data = data
        self.priors = {}

    def aggregate_data(self):
        """Aggregate game logs by player, position, and season."""
        self.data['season_start'] = pd.to_datetime(self.data['DATE'], errors='coerce').apply(
            lambda x: x.year if x.month >= 10 else x.year - 1
        )
        self.data['season_end'] = self.data['season_start'] + 1

        # Ensure relevant stats are numeric
        stats = ['FG', 'FGA', 'FT', 'FTA', '3P', '3PA']
        for stat in stats:
            self.data[stat] = pd.to_numeric(self.data[stat], errors='coerce')

        # Aggregate by player, position, and season
        self.aggregated_data = self.data.groupby(['NAME', 'Position']).agg(
            total_games=('DATE', 'count'),
            total_FG=('FG', 'sum'),
            total_FGA=('FGA', 'sum'),
            total_FT=('FT', 'sum'),
            total_FTA=('FTA', 'sum'),
            total_3PM=('3P', 'sum'),
            total_3PA=('3PA', 'sum')
        ).reset_index()

    def fit_beta_mle(self, data):
        """Fit a Beta distribution using MLE."""
        def beta_log_likelihood(params, data):
            alpha, beta_param = params
            return -np.sum(beta_dist.logpdf(data, alpha, beta_param))

        initial_guess = [1.0, 1.0]
        result = minimize(
            beta_log_likelihood, initial_guess, args=(data,),
            bounds=[(1e-6, None), (1e-6, None)], method='L-BFGS-B'
        )
        return result.x

    def fit_priors(self):
        """Fit priors (alpha, beta) for percentage stats by position."""
        percentage_stats = {
            'FG%': ('total_FG', 'total_FGA'),
            'FT%': ('total_FT', 'total_FTA'),
            '3P%': ('total_3PM', 'total_3PA')
        }

        # Fit Beta priors for percentage stats
        for stat, (makes, attempts) in percentage_stats.items():
            for position in self.aggregated_data['Position'].unique():
                pos_data = self.aggregated_data[self.aggregated_data['Position'] == position]
                data = pos_data[makes] / (pos_data[attempts] + 1e-6)
                data = data.fillna(0)

                alpha, beta = self.fit_beta_mle(data.values)
                self.priors[(stat, position)] = (alpha, beta)
                print(f"Fitted Beta prior for {stat} ({position}): alpha={alpha}, beta={beta}")

    def calculate_eb_estimates(self):
        """Calculate Empirical Bayes estimates and store alpha/beta for each stat."""
        results = []

        for _, row in self.aggregated_data.iterrows():
            player_results = {'NAME': row['NAME'], 'Position': row['Position']}
            n = row.get('total_games', 0)

            # Loop over priors (stat, position) pairs
            for (stat_name, position), (alpha, beta) in self.priors.items():
                if position == row['Position']:  # Ensure matching position
                    stat_name_clean = stat_name.replace('%', '')  # Clean stat name

                    # Access correct columns based on the stat
                    if stat_name_clean == 'FG':
                        makes = row.get('total_FG', 0)  # FG made
                        attempts = row.get('total_FGA', 0)  # FG attempts
                    elif stat_name_clean == '3P':
                        makes = row.get('total_3PM', 0)  # 3P made
                        attempts = row.get('total_3PA', 0)  # 3P attempts
                    elif stat_name_clean == 'FT':
                        makes = row.get('total_FT', 0)  # FT made
                        attempts = row.get('total_FTA', 0)  # FT attempts

                    # Handle zero attempts to avoid division by zero
                    if attempts == 0:
                        eb_estimate = 0
                    else:
                        # Calculate EB estimate
                        eb_estimate = (makes + alpha) / (attempts + alpha + beta)

                    # Calculate variance for the EB estimate
                    variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))

                    # Store results in the player results dictionary
                    player_results[f'eb_{stat_name_clean}_estimate'] = eb_estimate
                    player_results[f'{stat_name_clean}_variance'] = variance
                    player_results[f'{stat_name_clean}_alpha'] = alpha
                    player_results[f'{stat_name_clean}_beta'] = beta

            results.append(player_results)

        return pd.DataFrame(results)



# Example Usage
eb_model = EmpiricalBayesBasketball(combined_data_with_position)
eb_model.aggregate_data()
eb_model.fit_priors()
estimates = eb_model.calculate_eb_estimates()

print(estimates)
estimates.to_csv("estimates.csv")



plt.figure(figsize=(10, 6))
plt.hist(a['3PT%'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('3-Point Shooting Percentage Distribution')
plt.xlabel('3-Point Percentage')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()




import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

class PlayerAttemptModel:
    def __init__(self, data):
        self.data = data
        self.player_params = {}  # Store fitted parameters for each player

    def negative_binomial_nll(self, params, data):
        """Negative log-likelihood for the Negative Binomial."""
        mu, r = params
        if mu <= 0 or r <= 0:
            return np.inf  # Invalid parameter values
        
        # Calculate the Negative Binomial log-likelihood
        nll = -np.sum(
            gammaln(data + r) - gammaln(r) - gammaln(data + 1) +
            r * np.log(r / (r + mu)) + data * np.log(mu / (r + mu))
        )
        return nll

    def fit_player_parameters(self, player_name):
        """Fit Negative Binomial parameters for a specific player."""
        player_data = self.data[self.data['NAME'] == player_name]['3PA'].values

        # Initial guesses for mu (mean) and r (dispersion)
        initial_guess = [np.mean(player_data), 1.0]  # Mean attempts and dispersion

        # Fit the parameters using MLE
        result = minimize(
            self.negative_binomial_nll, initial_guess, args=(player_data,),
            method='L-BFGS-B', bounds=[(1e-6, None), (1e-6, None)]
        )

        mu, r = result.x
        self.player_params[player_name] = (mu, r)  # Store the parameters

        print(f"Fitted parameters for {player_name} -> Mean: {mu:.2f}, Dispersion: {r:.2f}")

    def simulate_attempts(self, player_name, num_games=100):
        """Simulate FG or 3PA attempts for a specific player."""
        if player_name not in self.player_params:
            print(f"No parameters found for {player_name}. Fitting now...")
            self.fit_player_parameters(player_name)

        mu, r = self.player_params[player_name]
        p = r / (r + mu)  # Success probability for Negative Binomial

        # Simulate attempts over the specified number of games
        simulated_attempts = np.random.negative_binomial(r, p, size=num_games)
        return simulated_attempts

# Example Usage

# Assuming you have a DataFrame 'combined_data_with_position' with 'NAME' and 'FGA' columns
model = PlayerAttemptModel(combined_data_with_position)

# Fit parameters for a specific player (e.g., 'Stephen Curry')
model.fit_player_parameters('Kevin Durant')

# Simulate 100 games of attempts for the same player
simulated_attempts = model.simulate_attempts('Kevin Durant', num_games=331)

# Display the simulated attempts and their statistics
print(f"Simulated Attempts (First 10): {simulated_attempts[:10]}")
print(f"Mean: {np.mean(simulated_attempts)}, Variance: {np.var(simulated_attempts)}")


actual_fg = combined_data_with_position[combined_data_with_position.NAME=='Kevin Durant']['3PA']

# Plot simulated Beta-Binomial data
plt.hist(simulated_attempts, bins=10, alpha=0.6, label='Simulated 3P(Neg-Binomial)', color='blue')
# Plot actual FG data for Aaron Gordon
plt.hist(actual_fg, bins=10, alpha=0.6, label='Actual FG', color='green')
plt.show()
