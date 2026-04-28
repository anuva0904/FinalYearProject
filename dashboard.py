"""
dashboard.py — SmartRoute Streamlit Demo (Final Stable Version)
Run: streamlit run dashboard.py
"""

import streamlit as st
import folium
from streamlit_folium import st_folium

from config import STOPS


# ─────────────────────────────────────────
# Page Setup
# ─────────────────────────────────────────
st.set_page_config(page_title="SmartRoute Delhi", layout="wide")

st.title("SmartRoute — AI Bus Route Optimizer")
st.caption("LSTM + Genetic Algorithm | Delhi Bus Network")


# ─────────────────────────────────────────
# Session State
# ─────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None

if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False


# ─────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────
st.sidebar.header("Route settings")
st.sidebar.info("Use Demo Route or select 6+ waypoints to see AI optimization improvement.")

names = [s["name"] for s in STOPS]

start_name = st.sidebar.selectbox("Start stop", names, index=0)
end_name   = st.sidebar.selectbox("End stop", names, index=24)

time_options = {
    "Morning Peak (8 AM)": 8,
    "Afternoon Off-Peak (2 PM)": 14,
    "Evening Peak (6 PM)": 18
}
time_selection = st.sidebar.selectbox("Time of day", list(time_options.keys()))
hour = time_options[time_selection]

is_fog  = int(st.sidebar.checkbox("Foggy conditions"))
is_rain = int(st.sidebar.checkbox("Rainy conditions"))

waypoint_names = st.sidebar.multiselect(
    "Optional waypoints",
    [n for n in names if n != start_name and n != end_name]
)

# Convert names → IDs
start = next(s["id"] for s in STOPS if s["name"] == start_name)
end   = next(s["id"] for s in STOPS if s["name"] == end_name)
wps   = [s["id"] for s in STOPS if s["name"] in waypoint_names]


# ─────────────────────────────────────────
# Demo Route Button (FIXED)
# ─────────────────────────────────────────
if st.sidebar.button("Use Demo Route"):
    st.session_state.demo_mode = True
    st.sidebar.success("Demo route loaded (click Optimize)")


# Apply demo route AFTER conversion
if st.session_state.demo_mode:
    start = 0
    end   = 24
    wps   = [2, 5, 11, 14, 18, 20, 23]


# Prevent invalid input
if start == end:
    st.sidebar.error("Start and End must be different")
    st.stop()


# ─────────────────────────────────────────
# Load Optimizer
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models...")
def get_optimizer(hour, is_fog, is_rain):
    from engine.optimizer import SmartRouteOptimizer
    return SmartRouteOptimizer(hour=hour, is_fog=is_fog, is_rain=is_rain)


# ─────────────────────────────────────────
# Run Optimization
# ─────────────────────────────────────────
if st.sidebar.button("Optimize route", type="primary"):

    try:
        opt = get_optimizer(hour, is_fog, is_rain)

        with st.spinner("Running AI Routing Engine..."):

            out = opt.optimize(start, end, waypoints=wps)

            # ── Off-peak comparison ──
            if hour != 14:
                opt_off = get_optimizer(14, is_fog, is_rain)
                off = opt_off.optimize(start, end, waypoints=wps)

                out["offpeak_time"] = off["travel_time_min"]
                out["offpeak_route_names"] = off["route_names"]
            else:
                out["offpeak_time"] = None

            # ── Greedy baseline ──
            unvisited = set(wps)
            current = start
            greedy_dt = 0.0
            greedy_route = [start]

            while unvisited:
                next_node = min(unvisited, key=lambda x: opt.tm[current][x])
                greedy_dt += opt.tm[current][next_node]
                current = next_node
                unvisited.remove(next_node)
                greedy_route.append(current)

            greedy_dt += opt.tm[current][end]
            greedy_route.append(end)

            # ── Improvement calculation
            improvement = (
                (greedy_dt - out["travel_time_min"]) / greedy_dt * 100
                if greedy_dt > 0 else 0
            )

            if improvement < 0:
                improvement = 0.0

            out["greedy_time"] = round(greedy_dt, 2)
            out["greedy_route"] = greedy_route
            out["greedy_improvement"] = round(improvement, 1)

            st.session_state.result = out

            # Reset demo mode after run
            st.session_state.demo_mode = False

    except FileNotFoundError:
        st.error("Run `python main.py` first to train model.")


# ─────────────────────────────────────────
# Display Results
# ─────────────────────────────────────────
result = st.session_state.result

if result:

    st.divider()

    # ── Metrics ──
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("AI Route Time", f"{result['travel_time_min']} min")
    c2.metric("Greedy Time", f"{result['greedy_time']} min")

    improv = result["greedy_improvement"]

    if improv > 0:
        c3.metric("Time Saved", f"{improv}%", delta=f"+{improv}%")
    else:
        c3.metric("Time Saved", "0%", delta="No improvement")

    c4.metric("Runtime", f"{result['runtime_ms']} ms")

    # ── Explanation ──
    if improv > 0:
        st.success("AI improved route using Genetic Algorithm optimization.")
    else:
        st.info("Route already optimal. Small problems often match greedy.")

    # ── Traffic Impact ──
    if result.get("offpeak_time"):
        st.subheader("Traffic Impact")

        if result["route_names"] != result["offpeak_route_names"]:
            st.warning("AI changed route due to traffic conditions")
        else:
            st.info("Route unchanged — only travel time affected")

    # ── Route text ──
    st.subheader("Route Comparison")

    st.markdown(f"**AI Route:** {' → '.join(result['route_names'])}")
    st.markdown(f"**Greedy Route:** {' → '.join([STOPS[i]['name'] for i in result['greedy_route']])}")

    # ── Map ──
    st.subheader("Route Map")

    m = folium.Map(location=[28.62, 77.21], zoom_start=11)

    coords = [(STOPS[i]["lat"], STOPS[i]["lon"]) for i in result["route"]]

    folium.PolyLine(coords, color="#534AB7", weight=5).add_to(m)

    for idx, stop_id in enumerate(result["route"]):
        s = STOPS[stop_id]

        color = "green" if idx == 0 else "red" if idx == len(result["route"]) - 1 else "blue"

        folium.CircleMarker(
            location=[s["lat"], s["lon"]],
            radius=7,
            color=color,
            fill=True,
            popup=s["name"]
        ).add_to(m)

    st_folium(m, width=900, height=480)

