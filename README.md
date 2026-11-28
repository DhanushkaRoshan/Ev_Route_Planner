# EV Route Optimization and Traffic Forecasting — Winnipeg

This repository contains the code for my MSc research on traffic volume forecasting, EV charging station availability prediction, and multi-objective route optimization for electric vehicles in Winnipeg, Canada.

## Project Structure

```
├── traffic_forecast/        # Traffic volume prediction and analysis
│   ├── prepare_data.py      # Data loading and preprocessing
│   ├── utility.py           # Shared helper functions and map generation
│   ├── evaluation.py        # Cross-validation metrics (MAE, RMSE, R²)
│   ├── missing_values.py    # Missing data analysis for traffic stations
│   ├── predict_idw.py       # Inverse Distance Weighting interpolation
│   ├── predict_kriging.py   # Regression Kriging (road-type trend + OK residuals)
│   ├── predict_gradient.py  # XGBoost with network centrality features
│   ├── bpr_function.py      # BPR congestion function and Level of Service
│   ├── bpr_montecarlo.py    # Monte Carlo simulation for traffic growth scenarios
│   └── thesis_six_methods.py # Comparison of 6 prediction methods
│
├── ev_stations_forecast/    # EV charging station availability prediction
│   ├── ev_lstm_shared.py    # Shared LSTM components (model, dataset, utils)
│   ├── sarima_model.py      # SARIMA time series model (per-station)
│   ├── sequential_lstm.py   # Single LSTM for all stations jointly
│   └── parallel_lstm_clustered.py  # K-Means clustered parallel LSTMs
│
├── optimization/            # Multi-objective EV route optimization
│   ├── config.py            # Configuration dataclasses and route definitions
│   ├── data_processor.py    # Graph loading, congestion data, charging stations
│   ├── route_planner.py     # Path generation and evaluation
│   ├── optimizer.py         # NSGA-II multi-objective optimizer
│   ├── dynamic_router.py    # Dynamic routing with rolling-horizon replanning
│   ├── visualizer.py        # Plotting (Pareto fronts, route maps, SOC profiles)
│   ├── main.py              # Main entry point for single route optimization
│   ├── run_all_routes.py    # Run optimization across all predefined routes
│   ├── run_experiments.py   # Experiment configurations and batch runner
│   └── analyses/            # Post-optimization analysis scripts
│       ├── matched_comparison.py   # Static vs dynamic plan comparison
│       ├── variance_study.py       # Multi-run variance analysis
│       └── intersection_noop.py    # Intersection replan no-op rate study
└── .gitignore
```

## Methods

### Traffic Volume Forecasting
- **IDW** — Inverse Distance Weighting spatial interpolation with power tuning
- **Regression Kriging** — Road-type trend model + Ordinary Kriging on residuals
- **XGBoost** — Gradient boosting with network centrality features (betweenness, degree, closeness)
- **KNN Regression** — K-Nearest Neighbors with optional network features
- **BPR Function** — Bureau of Public Roads congestion model with KNN
- **Monte Carlo Simulation** — Traffic growth scenario analysis (1000 iterations)

### EV Station Availability Prediction
- **SARIMA** — Seasonal ARIMA with auto parameter selection (per-station)
- **Sequential LSTM** — Single shared LSTM model for all stations
- **Clustered Parallel LSTM** — K-Means clustering (5 clusters) with separate LSTM per cluster

### Route Optimization
- **NSGA-II** — Multi-objective optimization minimizing travel time and charging cost
- **Dynamic Routing** — Rolling-horizon replanning with incident handling
- **Battery-aware pathfinding** — SOC tracking, charge trigger/target thresholds

## Requirements

- Python 3.8+
- pandas, numpy, geopandas, shapely
- scikit-learn, xgboost, pykrige
- PyTorch
- pmdarima
- osmnx, networkx
- folium, plotly, matplotlib, branca

## Data

Traffic data and road network files are not included in this repo due to size. Place the following in the appropriate `data/` directories:

- `traffic_forecast/data/` — Permanent Count Station and Midblock Traffic Count CSVs from the City of Winnipeg
- `optimization/data/` — Winnipeg road network GraphML, traffic congestion CSV, EV station data
