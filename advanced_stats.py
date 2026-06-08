import pandas as pd
import numpy as np
import pickle

more_data = pd.read_csv('players_data-2024_2025.csv')

def calculate_advanced_stats(more_data):

    player_metrics = {}

    for i, row in more_data.iterrows():
        player_name = row['Player']
        minutes_played = row['90s'] 

        if minutes_played == 0 or pd.isna(minutes_played):
            continue
        
        advanced_metrics = {
            #player info
            'position': row.get('Pos', ''),
            'squad': row.get('Squad', ''),
            'competition': row.get('Comp', ''),
            '90s_played': minutes_played,
            
            #defensive stats
            'tackles_interceptions_per_90': (row.get('Tak', 0) + row.get('Int', 0)) / minutes_played,
            'clearances_per_90': row.get('Clr', 0) / minutes_played,
            'blocks_per_90': row.get('Blocks', 0) / minutes_played,
            'recoveries_per_90': row.get('Recov', 0) / minutes_played,

            #progressive stats
            'carries_per_90': row.get('Carries', 0) / minutes_played,
             'distance': row.get('PrgDist_stats_possession', 0),
            
            #gk stats
            'saves': row.get('Saves', 0),
            'save_percentage': row.get('Save%', 0),
            'clean_sheets': row.get('CS', 0),
            'clean sheet_percentage': row.get('CS%', 0),
            
            #other
            'expected_goal_contributions':row.get('onxGA',0),
            
        }
        player_metrics[player_name] = advanced_metrics
        print(f"Processed metrics for {len(player_metrics)} players")
    return player_metrics
    
def create_metrics_dataframe(player_metrics):
    records = []
    for player_name, metrics in player_metrics.items():
        record = {'Player': player_name}
        record.update(metrics)
        records.append(record)

    return pd.DataFrame(records)
    
def save_metrics(player_metrics, output_path='advanced_player_metrics.pkl'):
    with open(output_path, 'wb') as f:
        pickle.dump(player_metrics, f)
        print(f"Saved advanced metrics for {len(player_metrics)} players to {output_path}")

def load_saved_metrics(input_path='advanced_player_metrics.pkl'):
    with open(input_path, 'rb') as f:
        return pickle.load(f)
        
    # Run if executed directly
if __name__ == "__main__":
    # Load and process
    df = pd.read_csv('players_data-2024_2025.csv')
    metrics = calculate_advanced_stats(df)
    
    # Save for later use
    save_metrics(metrics)
    
    # Also save as CSV for inspection
    