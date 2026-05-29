from difflib import get_close_matches
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import xgboost as xgb

import joblib
import json

from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor

import advanced_stats as am

import warnings
warnings.filterwarnings('ignore')


players = pd.read_csv('players.csv')
valuations = pd.read_csv('player_valuations.csv')
appearances = pd.read_csv('appearances.csv')
transfers = pd.read_csv('transfers.csv')
events = pd.read_csv('game_events.csv')
comps = pd.read_csv('competitions.csv')
nt = pd.read_csv('national_teams.csv')
countries = pd.read_csv('countries.csv')
games = pd.read_csv('games.csv')
club_games = pd.read_csv('club_games.csv')

clubs = pd.read_csv('clubs.csv')



club_features =clubs[['club_id', 'national_team_players', 'net_transfer_record', 'average_age']].drop_duplicates() #select

try:
    player_advanced_metrics = am.load_saved_metrics('advanced_player_metrics.pkl')
    print(f"Loaded advanced metrics for {len(player_advanced_metrics)} players")
except FileNotFoundError:
    print("Run advanced_stats.py first to generate metrics")
    player_advanced_metrics = {}


club_features['net_transfer_record'] = (
    club_features['net_transfer_record']
    .astype(str)
    .str.replace('€', '', regex=False)
    .str.replace('+', '', regex=False)
    .str.replace('m', '', regex=False)
    .str.replace(',', '.', regex=False)  # Handle decimal commas if any
    .str.replace('--', '0', regex=False)  # Handle missing values
    .str.extract(r'([-+]?\d*\.?\d+)')  # Extract the number
    .astype(float)
) # Converts net transfer record to numeric by removing currency symbols and converting to float, handling missing values as 0

# Also ensure national_team_players is numeric
club_features['national_team_players'] = pd.to_numeric(club_features['national_team_players'], errors='coerce').fillna(0)

club_features['average_age'] = pd.to_numeric(club_features['average_age'], errors='coerce').fillna(25)  # Default to 25

valuations['date'] = pd.to_datetime(valuations['date'])  #converts valuation date to datetime
valuations['valuation_year'] = valuations['date'].dt.year  #extracts valuation year for easier analysis
valuations['valuation_month'] = valuations['date'].dt.month #extracts valuation month for easier analysis


players['birth_date'] = pd.to_datetime(players['date_of_birth'], errors='coerce')  #converts birth date to datetime, coercing errors to NaT

games_season = games[['game_id', 'season']].drop_duplicates() #selects game and season data

appearances1 = pd.merge(appearances, games_season, on='game_id', how='left') #connects game season data to appearances, allowing us to identify which season each appearance was in, which is necessary to connect performance data to player valuations based on season end year



appearances1 = appearances1.merge(
    club_games[['game_id', 'club_id']],
    on='game_id',
    how='left'
) #connects club game data to appearances, allowing us to identify which club each appearance was for, which is necessary to determine the league and level of competition for each appearance

appearances1 = appearances1.merge(clubs[['club_id', 'domestic_competition_id']], on='club_id', how='left')  #connects club data to appearances

appearances_clean = appearances1.drop_duplicates(subset=['player_id', 'game_id']) #removes duplicate appearances for the same player in the same game, which can occur if a player is substituted in and out multiple times in a game, to ensure that we are counting each player's performance in a game only once

appearances_clean = appearances_clean.merge(club_features, on='club_id', how='left') #connects club features to appearances, allowing us to include club-level factors in our analysis of player performance and valuation



appearance_counts = appearances_clean.groupby(['player_id', 'season']).size().reset_index(name='games_played') #counts games played for each player


performance = appearances_clean.groupby(['player_id', 'season']).agg({
    'goals': 'sum', 
    'assists': 'sum', 
    'yellow_cards': 'sum',
    'red_cards': 'sum',
    'minutes_played': 'sum'
    }).reset_index()  #aggregates performance data



#get player primary league
league_counts = appearances_clean.groupby(['player_id', 'season', 'domestic_competition_id']).size().reset_index(name='count')

player_league = (
    league_counts.sort_values(['player_id', 'season', 'count'], ascending =[True, True, False])
    .groupby(['player_id', 'season'])
    .first()
    .reset_index()[['player_id', 'season', 'domestic_competition_id']] #for each player and season, selects the league with the most appearances as the primary league
)
player_league = player_league.rename(columns={'domestic_competition_id': 'primary_league_id'}) #renames the league column for clarity



player_stats = pd.merge(appearance_counts, performance, on=['player_id', 'season'], how='left')  #connects performance data to main dataframe
player_stats = pd.merge(player_stats, player_league, on=['player_id', 'season'], how='left')  #connects primary league data to main dataframe

# Add per-90 metrics
player_stats['goals_per_90'] = np.where(player_stats['minutes_played'] > 0, player_stats['goals'] / (player_stats['minutes_played'] / 90), np.nan)
player_stats['assists_per_90'] = np.where(player_stats['minutes_played'] > 0, player_stats['assists'] / (player_stats['minutes_played'] / 90), np.nan)


player_stats['season_end_year'] = player_stats['season'].astype(int)  #extracts season end year for easier analysis




player_club_features = appearances_clean.groupby(['player_id', 'season']).agg({
    'national_team_players': 'mean',  # Average number of internationals at club that season
    'net_transfer_record': 'mean',      # Average net spend of club that season
    'average_age': 'mean'               # Average age of players at club that season

}).reset_index() #aggregates club-level features to the player-season level by taking the average of the number of internationals and net transfer record for the clubs the player was associated with during that season, which allows us to include club context in our analysis of player performance and valuation

player_club_features = player_club_features.rename(columns={
    'national_team_players': 'avg_club_internationals',
    'net_transfer_record': 'avg_club_net_spend',
    'average_age': 'avg_club_average_age'
}) #renames the aggregated club features for clarity, indicating that these are average values for the clubs associated with the player in that season

player_stats = player_stats.merge(player_club_features, on=['player_id', 'season'], how='left') #connects club-level features to player stats, allowing us to include club context in our analysis of player performance and valuation

player_stats['avg_club_internationals'] = player_stats['avg_club_internationals'].fillna(0) #fills missing values for club internationals with 0, which assumes that if we don't have data on the number of internationals at a player's club for a season, we will treat it as if the club had no internationals that season, which is a reasonable assumption for handling missing data in this context
player_stats['avg_club_net_spend'] = player_stats['avg_club_net_spend'].fillna(0) #fills missing values for club-level features with 0, which assumes that if we don't have data on the club's internationals or net spend for that season, we will treat it as if the club had no internationals and no net spend, which is a reasonable assumption for handling missing data in this context
player_stats['avg_club_average_age'] = player_stats['avg_club_average_age'].fillna(25)  #fills missing values for average club age with 25, which is a reasonable assumption for handling missing data in this context, as the average age of players at a club is typically around 25 years old
# Log transform net spend (skewed)
player_stats['log_net_spend'] = np.log1p(player_stats['avg_club_net_spend'].clip(lower=0))

min_season_year = player_stats['season_end_year'].min() #finds the minimum season end year in the player stats data to use as a cutoff for including valuations in the analysis, ensuring that we are only including valuations for players and seasons for which we have performance data, which is necessary to connect player performance to valuations in our regression model

valuations_post = valuations[valuations['valuation_month'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])]  



valuations_filter = valuations_post[valuations_post['valuation_year'] >= min_season_year] #filters the valuations data to include only valuations that occurred in the same year or after the minimum season end year found in the player stats data, ensuring that we are only including valuations for players and seasons for which we have performance data, which is necessary to connect player performance to valuations in our regression model


df = pd.merge(players, valuations_filter, on='player_id')  #connects player info with valuation
#df = pd.merge(df, players[['player_id', 'date_of_birth']], on='player_id', how='left')



df['age'] = (df['date'] - df['birth_date']).dt.days / 365.25 #calculates player age at the time of valuation by finding the difference between valuation date and birth date, and converting it to years
df['age_squared'] = df['age'] ** 2 #creates a new feature for age squared to capture potential non-linear relationship between age and market value


position_dummies = pd.get_dummies(df['position'], prefix='pos')
df = pd.concat([df, position_dummies], axis=1)  #one-hot encodes player positions



valuation_pairs =df[['player_id', 'valuation_year']].drop_duplicates()  #selects unique player and valuation year pairs to connect with national team data

league_dummies = pd.get_dummies(player_stats['primary_league_id'], prefix='league') #one-hot encodes primary league, which is the league where the player had the most appearances in a season, to capture potential differences in market value based on league


player_stats = pd.concat([player_stats, league_dummies], axis=1) #connects the one-hot encoded league data back to the main dataframe based on player and season

def create_name_mapping(df, advanced_metrics_keys):
    name_mapping = {}
    df['name_lower'] = df['name'].str.lower()
    advanced_names_lower = {name.lower().strip(): name for name in advanced_metrics_keys}
    df['matched_name'] = df['name_lower'].map(advanced_names_lower)
    matched = df[df['matched_name'].notna()]
    
    name_mapping = dict(zip(matched['player_id'], matched['matched_name']))
    print(f"  Exact matches: {len(name_mapping)}")
    
    return name_mapping

name_mapping = create_name_mapping(players, player_advanced_metrics.keys()) #creates a mapping of player IDs to names in the advanced metrics data, using exact matches and close matches based on name similarity, to allow us to connect the advanced metrics to the main dataframe based on player ID

id_to_advanced = {}
for player_id, metrics_name in name_mapping.items():
    if metrics_name in player_advanced_metrics:
        id_to_advanced[player_id] = player_advanced_metrics[metrics_name] #creates a mapping of player IDs to their corresponding advanced metrics based on the name mapping, allowing us to connect the advanced metrics to the main dataframe based on player ID

if id_to_advanced:

    advanced_records = []
    for player_id, metrics in id_to_advanced.items():
        advanced_records.append({
            'player_id': player_id,
            'tackles_interceptions_per_90': metrics.get('tackles_interceptions_per_90', 0),
            'clearances_per_90': metrics.get('clearances_per_90', 0),
            'blocks_per_90': metrics.get('blocks_per_90', 0),
            'recoveries_per_90': metrics.get('recoveries_per_90', 0),
            'carries_per_90': metrics.get('carries_per_90', 0),
            'distance': metrics.get('distance', 0),
            'expected_goal_contributions': metrics.get('expected_goal_contributions', 0)
                                 })

    advanced_df = pd.DataFrame(advanced_records) #creates a dataframe from the advanced metrics records, allowing us to connect the advanced metrics to the main dataframe based on player ID

    player_stats = player_stats.merge(advanced_df, on='player_id', how='left') #connects the advanced metrics to the main dataframe based on player ID, allowing us to include the advanced metrics in our analysis of player performance and valuation

    adv_cols = ['tackles_interceptions_per_90', 'clearances_per_90', 'blocks_per_90', 
                'recoveries_per_90', 'carries_per_90', 'distance', 'expected_goal_contributions'] #defines the list of advanced metric columns to check for in the main dataframe, which can help us ensure that we are handling missing values for all of the advanced metrics we want to include in our analysis and model
    for col in adv_cols:
        if col in player_stats.columns:
            player_stats[col] = player_stats[col].fillna(0) #fills missing values for the advanced metrics with 0, which assumes that if we don't have data on a player's tackles, interceptions, clearances, blocks, recoveries, carries, distance, or expected goal contributions for that season, we will treat it as if the player had no contributions in those areas for that season, which is a reasonable assumption for handling missing data in this context
        else:
            player_stats[col] = 0.0 #fills missing values for the advanced metrics with 0, which assumes that if we don't have data on a player's tackles, interceptions, clearances, blocks, recoveries, carries, distance, or expected goal contributions for that season, we will treat it as if the player had no contributions in those areas for that season, which is a reasonable assumption for handling missing data in this context
    
    matched_count = player_stats['tackles_interceptions_per_90'].gt(0).sum() #counts how many players in the main dataframe were successfully matched with advanced metrics by counting how many players have a value greater than 0 for one of the advanced metrics (in this case, tackles and interceptions per 90), which can help us understand the coverage of the advanced metrics in our dataset and assess how much of an impact they may have on our analysis and model performance
    print(f"Matched advanced metrics for {matched_count} players out of {len(player_stats)} total player-season records ({matched_count / len(player_stats) * 100:.2f}% coverage)") #prints the number and percentage of player-season records in the main dataframe that were successfully matched with advanced metrics, which can help us understand the coverage of the advanced metrics in our dataset and assess how much of an impact they may have on our analysis and model performance
else:
    matched_count = 0 #counts how many players in the main dataframe were successfully matched with advanced metrics, which can help us understand the coverage of the advanced metrics in our dataset and assess how much of an impact they may have on our analysis and model performance

df = pd.merge(df, player_stats,
              left_on=['player_id', 'valuation_year'],
              right_on=['player_id', 'season_end_year'],
              how='left')  #connects the one-hot encoded league data to the main dataframe

df['age_goals_interaction'] = df['age'] * df['goals_per_90']
df['age_squared_goals'] = df['age_squared'] * df['goals_per_90']
df['years_from_peak'] = abs(df['age'] - 27)
df['years_from_peak_squared'] = df['years_from_peak'] ** 2
#df['performance_club_interaction'] = df['goals_per_90'] * df['avg_club_internationals']



league_cols = [col for col in df.columns if col.startswith('league_')] #prints the new league columns created by one-hot encoding



df = df.dropna(subset=['goals', 'assists', 'minutes_played', 'yellow_cards', 'red_cards', 'goals_per_90', 'assists_per_90', 'age', 'age_squared', 
                       'pos_Attack', 'pos_Defender', 'pos_Goalkeeper', 'pos_Midfield',  'market_value_in_eur_y'])  #removes rows with missing performance data

print(df[df['player_id'] == 8198][['name', 'goals', 'assists', 'minutes_played', 'yellow_cards', 'red_cards', 'goals_per_90', 'assists_per_90', 'age', 
                                   'market_value_in_eur_y']].head()) 

features = ['goals', 'assists', 'minutes_played', 'yellow_cards', 'red_cards', 'goals_per_90', 
            'pos_Attack', 'pos_Defender', 'pos_Goalkeeper', 'pos_Midfield',  'age', 'age_squared','assists_per_90', 'age_squared_goals', 'years_from_peak', 'avg_club_internationals', 'avg_club_average_age',
           'tackles_interceptions_per_90',
    'clearances_per_90',
    'blocks_per_90', 
    'recoveries_per_90',
    'carries_per_90',
    'distance',
    'expected_goal_contributions' ] + league_cols #defines the features to be used in the regression model, including performance metrics, position, age, and league dummies
x = df[features]
y = df['market_value_in_eur_y'] #defines the target variable as the log of market value to account for the skewed distribution of player valuations, which can help improve the performance of the regression model



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=15,          # Limit depth to prevent overfitting
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)


model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"R²: {r2_score(y_test, y_pred):.4f}") #evaluates the performance of the regression model using R-squared metric, which indicates how well the model explains the variance in the target variable (market value) based on the features used in the model

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', key=abs, ascending=False)

print(feature_importance.head(10))

joblib.dump(model, 'player_valuation_model.pkl')
#joblib.dump(scaler, 'scaler.pkl') #saves the trained regression model and the scaler used for feature scaling to disk using joblib, allowing us to load them later for making predictions on new data without having to retrain the model

with open('features.json', 'w') as f:
    json.dump(features, f) #saves the list of features used in the model to a JSON file, which can be useful for reference when making predictions on new data to ensure that the same features are used in the same order as during training

position_categories = df['position'].unique().tolist() #extracts the unique position categories from the data and saves them to a JSON file, which can be useful for reference when making predictions on new data to ensure that the same position categories are used for one-hot encoding as during training
league_categories = player_stats['primary_league_id'].unique().tolist() #extracts the unique position categories and league categories from the data and saves them to JSON files, which can be useful for reference when making predictions on new data to ensure that the same categories are used for one-hot encoding as during training

with open('position_categories.json', 'w') as f:
    json.dump(position_categories, f) #saves the unique position categories to a JSON file, which can be useful for reference when making predictions on new data to ensure that the same position categories are used for one-hot encoding as during training

with open('league_categories.json', 'w') as f:
    json.dump(league_categories, f) #saves the unique league categories to a JSON file, which can be useful for reference when making predictions on new data to ensure that the same league categories are used for one-hot encoding as during training


