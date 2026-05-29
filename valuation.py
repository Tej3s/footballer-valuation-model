import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')
class FootballPredictor:
    def __init__(self):
        self.model =joblib.load('player_valuation_model.pkl')
        #self.scaler = joblib.load('scaler.pkl')

        with open('features.json', 'r') as f:
            self.features = json.load(f)

        with open('position_categories.json', 'r') as f:
            self.position_categories = json.load(f)

        with open('league_categories.json', 'r') as f:
            self.league_categories = json.load(f)
        
        #print(f"Model expects {len(self.features)} features:")
        #print(self.features[:20])
        print("Is age_squared in features?", 'age_squared' in self.features)
        print("Position of age_squared:", self.features.index('age_squared') if 'age_squared' in self.features else 'NOT FOUND')
        print(type(self.model).__name__)

    def predict(self, player_data):
    # Create a DataFrame from the input data
        df = pd.DataFrame([player_data])
       
        
        
        df['age_squared'] = df['age'] ** 2

        if 'age_squared_goals' in self.features:
            df['age_squared_goals'] = df['age_squared'] * df['goals_per_90']
        
        if 'years_from_peak' in self.features:
            df['years_from_peak'] = abs(df['age'] - 27)

    # One-hot encode position
        for pos in self.position_categories:
            df[f'pos_{pos}'] = (df['position'] == pos).astype(int)

        for league_id in self.league_categories:
            col = f'league_{league_id}'
            df[col] = (df['league_id'] == league_id).astype(int) # One-hot encode league
    
    # Ensure all features are present
        for feature in self.features:
            if feature not in df.columns:
             df[feature] = 0

    # Scale features
        X = df[self.features]
        #X_scaled = self.scaler.transform(X)

    # Predict log market value and convert back to original scale
        log_pred = self.model.predict(X)[0]
        pred_value = (log_pred)

      
    
        

        return pred_value

if __name__ == "__main__":
    predictor = FootballPredictor()
    
    # Example: Prime Ronaldo (Real Madrid, age 27, 1.1 goals/90)
    ronaldo_age33 = {
        'goals': 28,
        'assists': 10,
        'minutes_played': 3700,
        'yellow_cards': 5,
        'red_cards': 0,
        'goals_per_90': 0.68,
        'assists_per_90': 0.24,
        'age': 33,
        'position': 'Attack',
        'league_id': 'IT1',
        'avg_club_internationals': 19,
        'avg_club_average_age': 27,
        # ADD ADVANCED METRICS
        'distance': 4500,
        'carries_per_90': 3.2,
        'tackles_interceptions_per_90': 0.6,
        'clearances_per_90': 0.2,
        'blocks_per_90': 0.1,
        'recoveries_per_90': 2.0,
        'expected_goal_contributions': 0.45
    }
    
    value = predictor.predict(ronaldo_age33)
    print(f"Predicted value of Juventus Ronaldo: €{value:,.0f}")
    
    # CORRECTED Prime Ronaldo
    test_prime_ronaldo = {
        'goals': 60,
        'assists': 15,
        'minutes_played': 4634,
        'yellow_cards': 5,
        'red_cards': 0,
        'goals_per_90': 1.17,
        'assists_per_90': 0.29,
        'age': 27,
        'position': 'Attack',
        'league_id': 'ES1',
        'avg_club_internationals': 20,
        'avg_club_average_age': 27,
        # ADD ADVANCED METRICS (peak Ronaldo)
        'distance': 6800,
        'carries_per_90': 6.5,
        'tackles_interceptions_per_90': 0.9,
        'clearances_per_90': 0.3,
        'blocks_per_90': 0.2,
        'recoveries_per_90': 2.8,
        'expected_goal_contributions': 0.85
    }
    
    value = predictor.predict(test_prime_ronaldo)
    print(f"Predicted value for prime Ronaldo: €{value:,.0f}")
    
    # CORRECTED Mbappe
    test_mbappe = {
        'goals': 44,
        'assists': 10,
        'minutes_played': 3500,
        'yellow_cards': 2,
        'red_cards': 0,
        'goals_per_90': 1.13,
        'assists_per_90': 0.26,
        'age': 24,
        'position': 'Attack',
        'league_id': 'FR1',
        'avg_club_internationals': 20,
        'avg_club_average_age': 26,
        # ADD ADVANCED METRICS
        'distance': 5800,
        'carries_per_90': 7.0,
        'tackles_interceptions_per_90': 0.7,
        'clearances_per_90': 0.2,
        'blocks_per_90': 0.1,
        'recoveries_per_90': 2.3,
        'expected_goal_contributions': 0.78
    }
    
    value = predictor.predict(test_mbappe)
    print(f"Mbappé (age 24): €{value:,.0f}")
    
    # CORRECTED Van Dijk (defender needs different advanced stats)
    test_vandijk_2019 = {
        'goals': 6,
        'assists': 4,
        'minutes_played': 4465,
        'yellow_cards': 4,
        'red_cards': 0,
        'goals_per_90': 0.12,
        'assists_per_90': 0.08,
        'age': 27,
        'position': 'Defender',
        'league_id': 'GB1',
        'avg_club_internationals': 22,
        'avg_club_average_age': 26.7,
        # ADD ADVANCED METRICS (defender specializes in defensive stats)
        'distance': 5200,
        'carries_per_90': 2.5,
        'tackles_interceptions_per_90': 3.5,  # High for defender
        'clearances_per_90': 5.0,              # Very high for CB
        'blocks_per_90': 1.2,
        'recoveries_per_90': 6.0,
        'expected_goal_contributions': 0.15
    }
    
    value = predictor.predict(test_vandijk_2019)
    print(f"Van Dijk (age 27, Premier League): €{value:,.0f}")

    # Mykhailo Mudryk 
    test_mudryk = {
    # Basic stats (pre-Chelsea at Shakhtar)
    'goals': 10,          # In 18 games before Chelsea
    'assists': 8,
    'minutes_played': 1500,
    'yellow_cards': 2,
    'red_cards': 0,
    'goals_per_90': 0.60,
    'assists_per_90': 0.48,
    'age': 22,            # Young = potential premium
    'position': 'Attack',
    'league_id': 'UKR',   # Ukrainian Premier League (low quality)
    'avg_club_internationals': 12,  # Shakhtar (few internationals)
    'avg_club_average_age': 26,
    
    # Physical tools 
    'distance': 6200,     # Elite workrate
    'carries_per_90': 7.5, # Lightning fast dribbling
    'tackles_interceptions_per_90': 1.2,
    'clearances_per_90': 0.3,
    'blocks_per_90': 0.1,
    'recoveries_per_90': 3.0,
    'expected_goal_contributions': 0.42
}

    value = predictor.predict(test_mudryk)
    print(f"Mudryk (Shakhtar stats): €{value:,.0f}")

# Toni Kroos 
    test_prime_kroos_CORRECTED = {
    # Basic stats
    'goals': 5,
    'assists': 12,
    'minutes_played': 4000,
    'yellow_cards': 4,
    'red_cards': 0,
    'goals_per_90': 0.11,
    'assists_per_90': 0.27,
    'age': 27,
    'position': 'Midfield',
    'league_id': 'ES1',
    'avg_club_internationals': 25,
    'avg_club_average_age': 27.5,
    
    # CORRECTED advanced metrics (low workrate)
    'distance': 8500,      # NOT 11000 - he's economical
    'carries_per_90': 1.5, # Rarely dribbles
    'tackles_interceptions_per_90': 1.2,  # LOW - avoids tackling
    'clearances_per_90': 0.6,
    'blocks_per_90': 0.2,
    'recoveries_per_90': 3.5,  # LOW for midfielder
    'expected_goal_contributions': 0.45  # Still creative
}

    value = predictor.predict(test_prime_kroos_CORRECTED)
    print(f"Kroos: €{value:,.0f}")

# N'Golo Kanté 
    test_prime_kante = {
    # Basic stats (modest for his value)
    'goals': 3,              # Rarely scores
    'assists': 4,            # Not a creator
    'minutes_played': 3800,  # Never stops running
    'yellow_cards': 6,
    'red_cards': 0,
    'goals_per_90': 0.07,
    'assists_per_90': 0.09,
    'age': 26,               # Peak physical years
    'position': 'Midfield',
    'league_id': 'GB1',      # Premier League
    'avg_club_internationals': 22,  # Chelsea (Hazard, Courtois, etc.)
    'avg_club_average_age': 26.5,
    
    # ELITE workrate stats (where Kanté dominates)
    'distance': 12500,       # SUPERHUMAN - covers every blade
    'carries_per_90': 3.5,   # Underrated dribbler
    'tackles_interceptions_per_90': 4.8,  # GOD-TIER
    'clearances_per_90': 1.5,
    'blocks_per_90': 0.6,
    'recoveries_per_90': 8.5,  # ABSURD - never stops
    'expected_goal_contributions': 0.12  # Low, not his job
}

    value = predictor.predict(test_prime_kante)
    print(f"N'Golo Kanté: €{value:,.0f}")

# Lionel Messi 
    test_prime_messi = {
    # BASIC STATS - ABSURD NUMBERS
    'goals': 73,             # La Liga season record
    'assists': 29,           # Also elite creator
    'minutes_played': 4500,  # Played almost everything
    'yellow_cards': 4,
    'red_cards': 0,
    'goals_per_90': 1.46,    # HIGHER than Prime Ronaldo's 1.17
    'assists_per_90': 0.58,  # Also higher than Ronaldo's 0.29
    'age': 25,               # Peak Messi (younger than Ronaldo's 27)
    'position': 'Attack',
    'league_id': 'ES1',      # La Liga
    'avg_club_internationals': 26,  # Prime Barca (Xavi, Iniesta, Puyol, etc.)
    'avg_club_average_age': 27,
    
    # ADVANCED METRICS (where Messi is still elite)
    'distance': 9500,        # Not a runner, but intelligent movement
    'carries_per_90': 8.5,   # ELITE - dribbles past everyone
    'tackles_interceptions_per_90': 1.2,  # Not his job
    'clearances_per_90': 0.2,
    'blocks_per_90': 0.1,
    'recoveries_per_90': 3.5,
    'expected_goal_contributions': 1.25   # ELITE - highest ever
}

    value = predictor.predict(test_prime_messi)
    print(f"Lionel Messi (2012 - Peak): €{value:,.0f}")

# João Cancelo
    test_cancelo_2022 = {
    # BASIC STATS (attacking fullback numbers)
    'goals': 3,
    'assists': 10,           # Elite for a fullback
    'minutes_played': 3500,
    'yellow_cards': 8,       # Tactical fouls (Pep special)
    'red_cards': 1,          # That one stupid red
    'goals_per_90': 0.08,
    'assists_per_90': 0.26,  # Better than many midfielders
    'age': 27,               # Peak age for fullback
    'position': 'Defender',  # Officially a fullback
    'league_id': 'GB1',      # Premier League
    'avg_club_internationals': 22,  # Man City (De Bruyne, Sterling, etc.)
    'avg_club_average_age': 26.5,
    
    # ADVANCED METRICS (where Cancelo is unique)
    'distance': 8000,       # Covers a lot of ground as a fullback
    'carries_per_90': 6.5,   # Inverts into midfield constantly
    'tackles_interceptions_per_90': 2.8,  # Decent but not elite
    'clearances_per_90': 1.5,
    'blocks_per_90': 0.3,
    'recoveries_per_90': 5.5,
    'expected_goal_contributions': 0.32
}

    value = predictor.predict(test_cancelo_2022)
    print(f"João Cancelo: €{value:,.0f}")

    test_ter_stegen = {
    'goals': 0,
    'assists': 1,
    'minutes_played': 3500,
    'goals_per_90': 0.0,
    'assists_per_90': 0.03,
    'age': 28,
    'position': 'Goalkeeper',
    'league_id': 'ES1',
    'avg_club_internationals': 22,
    'distance': 4500,  # Much lower than outfield
    'carries_per_90': 0.5,
    'tackles_interceptions_per_90': 0.1,
    'recoveries_per_90': 6.0,  # This helps (claims crosses)
    'expected_goal_contributions': 0.01
}
    value = predictor.predict(test_ter_stegen)
    print(f"Marc-André ter Stegen: €{value:,.0f}")

    
    test_neuer_2014 = {
    # Basic Stats (Goalkeeper context)
    'goals': 0,
    'assists': 0,
    'minutes_played': 4665,  # World Cup + Bayern season [citation:6][citation:7]
    'yellow_cards': 0,
    'red_cards': 0,
    'goals_per_90': 0.0,
    'assists_per_90': 0.0,
    'age': 28,  # Born March 27, 1986 [citation:3]
    'position': 'Goalkeeper',
    'league_id': 'DE1',  # Bundesliga
    'avg_club_internationals': 26,  # Peak Bayern (Müller, Lahm, Schweinsteiger, Kroos, etc.)
    'avg_club_average_age': 27,
    
    # Goalkeeper-Specific Advanced Metrics
    'distance': 5200,  # Sweeper-keeper - covered ground outside box [citation:10]
    'carries_per_90': 1.2,  # Rare for GK but Neuer unique
    'tackles_interceptions_per_90': 0.5,  # Sweeper clearances outside box [citation:10]
    'clearances_per_90': 2.5,  # Sweeper actions
    'blocks_per_90': 0.3,
    'recoveries_per_90': 7.5,
}
    value = predictor.predict(test_neuer_2014)
    print(f"Manuel Neuer (2014 World Cup): €{value:,.0f}")

    