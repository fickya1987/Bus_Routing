import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import networkx as nx
import folium
from folium import plugins
from streamlit_folium import folium_static
import random
import time
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import requests


def load_mumbai_population_data():
    # For demonstration, I'm using random population data within a realistic range
    data = {
        'Area': [f'Area {i + 1}' for i in range(50)],
        'lat': [random.uniform(19.1, 19.2) for _ in range(50)],
        'lon': [random.uniform(72.8, 73) for _ in range(50)],
        'PopulationDensity': [random.randint(50000, 100000) for _ in range(50)],
    }
    return pd.DataFrame(data)


# Function to train population prediction model
def train_population_model(X_train, y_train):
    # Scale features for SVR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = SVR()
    model.fit(X_train_scaled, y_train)
    return model, scaler


# Function to predict population density
def predict_population(model, data):
    data['PredictedPopulation'] = model.predict(data[['lat', 'lon']])
    return data


# Function to identify high population areas
def identify_high_population_areas(data, threshold):
    return data[data['PredictedPopulation'] > threshold]


# Function to construct a graph for route optimization
def construct_graph(data):
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_node(_, pos=(row['lat'], row['lon']), pop=row['PredictedPopulation'])

    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 != node2:
                distance = ((G.nodes[node1]['pos'][0] - G.nodes[node2]['pos'][0]) ** 2 +
                            (G.nodes[node1]['pos'][1] - G.nodes[node2]['pos'][1]) ** 2) ** 0.5
                G.add_edge(node1, node2, weight=distance)

    return G


# # Function to optimize route
# def optimize_route(G):
#     return nx.shortest_path(G, source=min(G.nodes), target=max(G.nodes), weight='weight')

# Function to optimize route with intermediate points
def optimize_route(G, num_intermediate_points=5):
    source = min(G.nodes)
    target = max(G.nodes)

    # Convert set to list
    nodes_list = list(set(G.nodes) - {source, target})

    # Check if there are enough nodes for the requested number of intermediate points
    if len(nodes_list) < num_intermediate_points:
        raise ValueError("Not enough nodes for the requested number of intermediate points.")

    intermediate_points = random.sample(nodes_list, num_intermediate_points)

    optimized_route = [source]
    for point in intermediate_points:
        optimized_route.extend(nx.shortest_path(G, source=optimized_route[-1], target=point, weight='weight'))
    optimized_route.extend(nx.shortest_path(G, source=optimized_route[-1], target=target, weight='weight'))

    return optimized_route


# # Function to simulate route animation
# def animate_route(map_obj, route_coordinates):
#     for i in range(len(route_coordinates) - 1):
#         route_segment = route_coordinates[i:i + 2]
#         folium.PolyLine(route_segment, color='red', weight=2.5, opacity=3).add_to(map_obj)
#         time.sleep(1)  # Adjust the delay between frames as needed
#         folium_static(map_obj)

# Function to animate route (show only the final route without intermediate steps)
def animate_route(map_obj, route_coordinates):
    folium.PolyLine(route_coordinates, color='red', weight=2.5, opacity=3).add_to(map_obj)
    folium_static(map_obj)


# Function to animate vehicle along the route
def animate_vehicle(map_obj, route_coordinates):
    # Create a folium PolyLine for the optimized route
    folium.PolyLine(route_coordinates, color='blue', weight=2.5, opacity=0.7).add_to(map_obj)

    # Add the AntPath for smoother animation
    ant_path = folium.plugins.AntPath(
        locations=route_coordinates,
        color='green',
        pulse_color='red',
        delay=1000,
        weight=5,
        reverse=False,
        dash_array=[10, 20],
        dash_offset=0
    ).add_to(map_obj)

    return ant_path.get_name()  # Return the AntPath name for JavaScript control

    # Animation function to update the vehicle marker location
    # def update_vehicle(i):
    #     if i < len(route_coordinates):
    #         vehicle_marker.location = route_coordinates[i]
    #         return vehicle_group

    # Use the plugins.TimestampedGeoJson plugin for animation
    plugins.TimestampedGeoJson({
        'type': 'FeatureCollection',
        'features': [{'type': 'Feature',
                      'geometry': {'type': 'Point',
                                   'coordinates': route_coordinates[i]},
                      'properties': {'time': i * 1000}  # Timestamp in milliseconds
                      } for i in range(len(route_coordinates))]
    }, period=1000).add_to(map_obj)


# Function to add markers to the map
def add_markers(map_obj, data, icon_color='blue', icon='bus'):
    for index, row in data.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f'Bus Station {index}',
            icon=folium.Icon(color=icon_color, icon_color='white', icon=icon, prefix='fa')
        ).add_to(map_obj)


# Function to create a folium map
def create_folium_map():
    # Extracting map configurations
    map_settings = {
        "location": [19.0760, 72.8777],
        "zoom_start": 12,
        "tiles": "CartoDB Positron"
    }

    # Create a folium map with the specified settings
    m = folium.Map(**map_settings)

    # Add background shape
    bg_shape_settings = {
        "type": "rectangle",
        "color": "#EDEFDA",
        "buffer": 0
    }
    add_background_shape(m, **bg_shape_settings)

    return m


def add_background_shape(map_obj, type, color, buffer):
    if type == "rectangle":
        # Extracting AOI bounds from map configurations
        aoi_bounds = [18.967483328131124, 72.82193459682873, 18.97922386785842, 72.83427526951994]

        # Creating a folium rectangle with AOI bounds
        rect = folium.Rectangle(bounds=[[aoi_bounds[0], aoi_bounds[1]], [aoi_bounds[2], aoi_bounds[3]]],
                                color=color, fill=True, fill_opacity=1.0, weight=0, fill_color=color)
        rect.add_to(map_obj)


# Streamlit App for Bus Route Prediction
def bus_route_app():
    st.title('Predicting Bus routes in Mumbai')

    # Load Mumbai's actual population data
    df = load_mumbai_population_data()

    # Display the loaded data
    st.subheader('Mumbai Population Data:')
    st.write(df)

    # Train population prediction model using SVR
    X_train, X_test, y_train, _ = train_test_split(df[['lat', 'lon']], df['PopulationDensity'], test_size=0.2,
                                                   random_state=42)
    population_model, scaler = train_population_model(X_train, y_train)

    # Predict population density
    df_scaled = scaler.transform(df[['lat', 'lon']])
    df['PredictedPopulation'] = population_model.predict(df_scaled)

    # Set a threshold for high population areas
    threshold = st.slider('Select population threshold for high population areas:', df['PredictedPopulation'].min(),
                          df['PredictedPopulation'].max(), df['PredictedPopulation'].quantile(0.8))

    # Identify high population areas
    high_population_areas = identify_high_population_areas(df, threshold)

    # Display high population areas
    st.subheader('High Population Areas:')
    st.write(high_population_areas)

    # Construct a graph for route optimization
    G = construct_graph(df)

    # # Optimize route
    # optimized_route = optimize_route(G)
    # Optimize route with 5 intermediate points
    optimized_route = optimize_route(G, num_intermediate_points=5)

    # Create a folium map
    m = create_folium_map()

    # Add markers for areas
    add_markers(m, df)

    # Simulate route animation
    animate_route(m, [(df.loc[node]['lat'], df.loc[node]['lon']) for node in optimized_route])

    # Animate the vehicle along the optimized route
    ant_path_name = animate_vehicle(m, [(df.loc[node]['lat'], df.loc[node]['lon']) for node in optimized_route])

    # Display the optimized route on Streamlit map
    st.subheader('Optimized Route on Streamlit Map:')
    st.markdown(folium_static(m), unsafe_allow_html=True)

    # Streamlit button to start the vehicle animation
    if st.button("Start Vehicle Animation"):
        st.write(f"""<script>
                        var antPath = document.getElementById('{ant_path_name}');
                        antPath.options.autoplay = true;
                    </script>""", unsafe_allow_html=True)


if __name__ == '__main__':
    bus_route_app()
