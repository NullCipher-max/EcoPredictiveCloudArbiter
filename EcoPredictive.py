import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def generate_carbon_data(days=60):
    """
    Generates hourly synthetic carbon intensity (gCO2/kWh).
    Simulates high grid intensity at night/evening peaks, 
    and lower intensity during mid-day due to solar generation.
    """
    np.random.seed(42)
    hours = days * 24
    timestamps = pd.date_range(start="2026-01-01", periods=hours, freq="h")
    
    hour_of_day = timestamps.hour
    base_intensity = 350 + 120 * np.sin((hour_of_day - 6) / 24 * 2 * np.pi)
    
   
    solar_suppression = np.where((hour_of_day >= 9) & (hour_of_day <= 16), 110, 0)
    
   
    noise = np.random.normal(0, 25, size=hours)
    
    carbon_intensity = np.clip(base_intensity - solar_suppression + noise, 80, 600)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "hour": hour_of_day,
        "day_of_week": timestamps.dayofweek,
        "carbon_intensity": carbon_intensity
    })
    
  
    df["carbon_lag_1h"] = df["carbon_intensity"].shift(1)
    df["carbon_lag_24h"] = df["carbon_intensity"].shift(24)
    df.dropna(inplace=True)
    return df

def train_carbon_predictor(df):
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:].copy()
    
    features = ["hour", "day_of_week", "carbon_lag_1h", "carbon_lag_24h"]
    target = "carbon_intensity"
    
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    model.fit(train_df[features], train_df[target])
    
    test_df["predicted_intensity"] = model.predict(test_df[features])
    rmse = np.sqrt(mean_squared_error(test_df[target], test_df["predicted_intensity"]))
    print(f"[Model Evaluated] Test RMSE: {rmse:.2f} gCO2/kWh")
    return model, test_df

def run_simulation(test_df, num_jobs=50):
    np.random.seed(42)
    test_hours = len(test_df)
    
    jobs = []
    for i in range(num_jobs):
        arrival_idx = np.random.randint(0, test_hours - 24)
        duration = np.random.randint(1, 4)
        deadline = arrival_idx + np.random.randint(duration + 1, 14)
        jobs.append({
            "job_id": f"Job_{i+1}",
            "arrival_idx": arrival_idx,
            "duration": duration,
            "deadline": deadline,
            "power_draw_kw": 2.5 
        })
    
    actual_carbon = test_df["carbon_intensity"].values
    predicted_carbon = test_df["predicted_intensity"].values
    
    naive_emissions_total = 0.0
    arbiter_emissions_total = 0.0
    
    for job in jobs:
        arr = job["arrival_idx"]
        dur = job["duration"]
        dl = job["deadline"]
        power = job["power_draw_kw"]
        naive_carbon_per_kwh = actual_carbon[arr : arr + dur].mean()
        naive_emissions = (power * dur) * naive_carbon_per_kwh
        naive_emissions_total += naive_emissions
        
       
        valid_start_slots = range(arr, dl - dur + 1)
        best_slot = arr
        min_predicted_intensity = float("inf")
        
        for slot in valid_start_slots:
            slot_predicted_avg = predicted_carbon[slot : slot + dur].mean()
            if slot_predicted_avg < min_predicted_intensity:
                min_predicted_intensity = slot_predicted_avg
                best_slot = slot
                
    
        actual_carbon_per_kwh = actual_carbon[best_slot : best_slot + dur].mean()
        arbiter_emissions = (power * dur) * actual_carbon_per_kwh
        arbiter_emissions_total += arbiter_emissions
        
    reduction_pct = ((naive_emissions_total - arbiter_emissions_total) / naive_emissions_total) * 100
    
    print("\n--- SIMULATION RESULTS ---")
    print(f"Total Jobs Processed: {num_jobs}")
    print(f"Naive Cloud Emissions:   {naive_emissions_total / 1000:.3f} kg CO2")
    print(f"Eco-Arbiter Emissions:  {arbiter_emissions_total / 1000:.3f} kg CO2")
    print(f"Total Carbon Reduction:  {reduction_pct:.2f} %")

if __name__ == "__main__":
    print("Generating grid carbon intensity dataset...")
    data = generate_carbon_data(days=60)
    
    print("Training predictive engine...")
    model, test_data = train_carbon_predictor(data)
    
    print("Executing workload orchestration simulation...")
    run_simulation(test_data, num_jobs=60)