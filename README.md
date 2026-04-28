SmartRoute — AI-Based Bus Route Optimization System
Project Overview
SmartRoute is an intelligent route optimization system designed for urban bus transportation networks. The project integrates machine learning and optimization techniques to compute efficient routes under dynamic traffic conditions. The system predicts travel times using historical traffic and weather data, and then applies optimization algorithms to determine the best sequence of stops.
The framework combines:
    • Time-series prediction using Long Short-Term Memory (LSTM)
    • Combinatorial optimization using Genetic Algorithms (GA)
    • Dynamic routing using real-time traffic simulation
The objective is to reduce travel time, improve route efficiency, and support adaptive decision-making in public transport systems.
Key Features
    • Travel time prediction using LSTM trained on traffic and weather data
    • Route optimization using Genetic Algorithms and brute-force fallback
    • Greedy baseline comparison for performance benchmarking
    • Dynamic routing based on time-of-day traffic conditions
    • Interactive Streamlit dashboard for visualization
    • Real-world inspired travel time matrix simulation
    • Scalability evaluation (Brute Force vs GA)
    • Traffic adaptability analysis across peak and off-peak hours

System Architecture
The system follows a multi-stage pipeline:
    1 Data Generation and Preprocessing Synthetic traffic and weather data are generated and merged to create training datasets.
    2 LSTM Model Learns temporal patterns in travel time based on historical data.
    3 Time Matrix Construction Builds a dynamic travel time matrix using predicted or simulated values.
    4 Route Optimization
        ◦ Small problems: solved using brute-force (optimal)
        ◦ Large problems: solved using Genetic Algorithm (approximate optimal)
    5 Evaluation Compares Greedy, GA, Hybrid, and Optimal solutions.
    6 Visualization Streamlit dashboard displays routes, maps, and performance metrics.

Technologies Used
Category	Tools / Libraries
Programming Language	Python
Machine Learning	TensorFlow, Keras
Optimization	DEAP (Genetic Algorithm)
Data Processing	NumPy, Pandas
Visualization	Matplotlib
Dashboard	Streamlit
Mapping	Folium
Reinforcement Learning (optional module)	Stable-Baselines3

Project Structure
SmartRoute/
│
├── config.py
├── main.py
├── dashboard.py
│
├── utils/
│   └── preprocess.py
│
├── lstm/
│   ├── train_lstm.py
│   └── predict.py
│
├── ga/
│   └── genetic_algorithm.py
│
├── engine/
│   └── optimizer.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│   └── comparison_chart.png
│
└── requirements.txt

Installation
Step 1: Clone the repository
git clone https://github.com/your-username/SmartRoute.git
cd SmartRoute
Step 2: Create virtual environment
python3 -m venv venv
source venv/bin/activate
Step 3: Install dependencies
pip install -r requirements.txt

Running the Project
Step 1: Train models and run pipeline
python main.py
This will:
    • Generate data
    • Train or load the LSTM model
    • Run optimization experiments
    • Save comparison chart in results/

Step 2: Run the dashboard
streamlit run dashboard.py
Open the provided local URL in your browser.
How to Use the Dashboard
    1 Select start and end stops
    2 Choose time of day (peak/off-peak)
    3 Optionally select intermediate waypoints
    4 Click “Optimize route”
    5 View:
        ◦ AI optimized route
        ◦ Greedy baseline route
        ◦ Travel time comparison
        ◦ Map visualization

Evaluation Metrics
The system evaluates performance using:
    • Travel Time (minutes)
    • Percentage Improvement over Greedy
    • Runtime (milliseconds)
    • Optimality comparison (Brute Force vs GA)
Experimental Results Summary
    • Small routes: All methods converge to optimal solution
    • Large routes: Genetic Algorithm significantly outperforms Greedy
    • Traffic adaptability: Up to 90% of routes change under different time conditions
    • Hybrid approach achieves near-optimal performance with low runtime

SDG Alignment
This project supports:
    • SDG 11: Sustainable Cities and Communities Improves efficiency of public transportation systems
    • SDG 13: Climate Action Reduces fuel consumption and emissions through optimized routing

Limitations
    • Synthetic traffic data (not fully real-world validated)
    • Limited number of bus stops (scalability can be extended)
    • Reinforcement learning module not fully integrated in dashboard
    • Real-time deployment not implemented

Future Work
    • Integration with real-time GPS and traffic APIs
    • Expansion to larger city-scale networks
    • Full reinforcement learning-based adaptive routing
    • Deployment as a web or mobile application
    • Multi-objective optimization (cost, emissions, passenger load)

Conclusion
SmartRoute demonstrates how AI techniques can be effectively combined to solve complex transportation problems. By integrating prediction and optimization, the system provides a scalable and adaptive solution for modern urban mobility challenges.
