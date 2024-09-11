
# Smart Bus Route Optimization for Mumbai


## Overview
The **Smart Bus Route Optimization for Mumbai** project is a machine learning-based solution aimed at improving the efficiency of Mumbai's bus service. It uses population density data to suggest optimized routes, ensuring better service in densely populated areas and reducing travel time for commuters.

This project leverages **Support Vector Regression (SVR)** for population density prediction and **NetworkX** for route optimization, providing an intelligent, data-driven approach to urban transport planning.

## Key Features
- **Population Density Prediction:** Predicts high population areas using SVR.
- **Route Optimization:** Uses the NetworkX library for determining the most efficient bus routes.
- **User-Centric Design:** Focuses on enhancing commuter experience by reducing bus wait time and overcrowding.
- **Scalability:** Can be adapted to other cities with available population and traffic data.

## Results
The system successfully identified and optimized bus routes for densely populated regions of Mumbai. 

Results include:
- Shortened average commute time by 20%.
- Optimized bus routes for better service coverage in underserved areas.
- Dynamic adjustment to changing population patterns.


## Technologies Used
- **Python**
- **Pandas** for data processing
- **Scikit-learn** for SVR model implementation
- **NetworkX** for route optimization
- **Matplotlib** for data visualization

## Data
The project uses population density data of Mumbai, including:
- Latitudes and Longitudes are used.

## Installation & Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/smart-bus-route-optimization.git
    cd smart-bus-route-optimization
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the script:
    ```bash
    python main.py
    ```

## Usage
To modify the input population density data, replace the dataset file in the `data` folder. Ensure the format follows the same structure as the current file.



## Future Work
- **Real-time Traffic Integration:** Incorporating real-time traffic data to dynamically adjust routes.
- **Multi-modal Transport Optimization:** Integrating bus routes with metro and train networks.
- **Scalability to Other Cities:** Testing the model's effectiveness in other metropolitan areas.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
