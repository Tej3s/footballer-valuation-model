import streamlit as st
import numpy as np
from valuation import FootballPredictor

st.set_page_config(page_title="Football Player Valuation", layout="centered")

st.title("Footballer Valuation Predictor ⚽")
st.caption("Enter player attributes to predict market value based on historical Transfermarkt data")

@st.cache_resource
def load_model():
    return FootballPredictor()

predictor = load_model()

tab1, tab2, tab3 = st.tabs(["Player Performance", "Advanced Stats", "Club Context"])

with tab1:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Performance Stats")
        minutes_played = st.number_input("Minutes Played", min_value=0, step=90, value=2000)
        goals = st.number_input("Goals", min_value=0, step=1, value=15)
        assists = st.number_input("Assists", min_value=0, step=1, value=5)
        goals_per_90 = st.number_input("Goals per 90", min_value=0.0, step=0.05, value=0.6, format="%.2f")
        assists_per_90 = st.number_input("Assists per 90", min_value=0.0, step=0.05, value=0.2, format="%.2f")
        
        
    with col2:
        st.subheader("Discipline and Age")
        yellow_cards = st.number_input("Yellow Cards", min_value=0, step=1, value=5)
        red_cards = st.number_input("Red Cards", min_value=0, step=1, value=0)
        age = st.number_input("Age", min_value=16, max_value=40, step=1, value=26)


with tab2:
    st.subheader("Advanced Metrics")

    col1, col2 = st.columns(2)

    with col1:
        tackles_interceptions = st.slider(
            "Tackles + Interceptions per 90", 
            min_value=0.0, max_value=6.0, value=1.5, step=0.1
            )
        clearances = st.slider(
            "Clearances per 90", 
            min_value=0.0, max_value=8.0, value=1.0, step=0.1,
            )
        blocks = st.slider(
            "Blocks per 90", 
            min_value=0.0, max_value=2.5, value=0.3, step=0.1,
            )
        recoveries = st.slider(
            "Recoveries per 90", 
            min_value=0.0, max_value=12.0, value=4.0, step=0.1,
            )
    
    with col2:
        carries = st.slider(
            "Carries per 90", 
            min_value=0.0, max_value=12.0, value=3.0, step=0.1,
        )
        distance = st.slider(
            "Progressive Distance per 90 (in meters)", 
             min_value=4000, max_value=13000, value=9000, step=500,
            format="%d",
        )
        expected_goal_contrib = st.slider(
            "Expected Goal Contributions per 90", 
            min_value=0.0, max_value=1.5, value=0.25, step=0.05,
        )
with tab3:

    st.subheader("Player & Club Context")
    position = st.selectbox("Position", ["Attack", "Midfield", "Defender", "Goalkeeper"])
    league_id = st.selectbox("League", ["GB1", "ES1", "IT1", "L1", "FR1", "Other"])
    club_internationals = st.number_input("# of Internationals in the Club", min_value=0, step=1, value=15)
    club_net_spend = st.number_input("Club Net Spend (€M)", min_value=0, step=10, value=50)
    club_average_age = st.number_input("Club Average Age", min_value=20, max_value=35, step=1, value=27)

if st.button("💰 Predict Market Value", type="primary"):
    player = {
        'minutes_played': minutes_played,
        'goals': goals,
        'assists': assists,
        'goals_per_90': goals_per_90,
        'assists_per_90': assists_per_90,
        'yellow_cards': yellow_cards,
        'red_cards': red_cards,
        'age': age,
        'position': position,
        'league_id': league_id,
        'avg_club_internationals': club_internationals,
        'log_net_spend': np.log1p(club_net_spend * 1_000_000),  # Convert to euros
        'avg_club_average_age': float(club_average_age),
        'tackles_interceptions_per_90': tackles_interceptions,
        'clearances_per_90': clearances,
        'blocks_per_90': blocks,
        'recoveries_per_90': recoveries,
        'carries_per_90': carries,
        'distance': float(distance),
        'expected_goal_contributions': expected_goal_contrib
    }
    
    with st.spinner("Calculating..."):
        value = predictor.predict(player)
        st.success(f"💰 **Predicted Market Value: €{value:,.0f}**")

def load_model():
    predictor = FootballPredictor()
    st.write(f"Model type: {type(predictor.model).__name__}")  # Add this
    return predictor