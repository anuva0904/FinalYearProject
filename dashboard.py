"""
dashboard.py  — Streamlit visual demo
Run: streamlit run dashboard.py
"""

import streamlit as st
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

from config import STOPS, N


# ─────────────────────────────────────────────────────────
# Page Setup
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="SmartRoute Delhi", layout="wide")

st.title("SmartRoute — AI Bus Route Optimizer")
st.caption("LSTM + Genetic Algorithm | Delhi Bus Network")


# Persist result across reruns
if "result" not in st.session_state:
    st.session_state.result = None


# ─────────────────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────────────────
st.sidebar.header("Route settings")

names = [s["name"] for s in STOPS]

start_name = st.sidebar.selectbox("Start stop", names, index=0)
end_name   = st.sidebar.selectbox("End stop", names, index=24)

time_options = {
    "Morning Peak (8 AM)": 8,
    "Afternoon Off-Peak (2 PM)": 14,
    "Evening Peak (6 PM)": 18
}
time_selection = st.sidebar.selectbox("Time of day", list(time_options.keys()), index=0)
hour = time_options[time_selection]

is_fog  = int(st.sidebar.checkbox("Foggy conditions"))
is_rain = int(st.sidebar.checkbox("Rainy conditions"))

waypoint_names = st.sidebar.multiselect(
    "Optional waypoints",
    [n for n in names if n != start_name and n != end_name]
)


# Convert names → stop ids
start = next(s["id"] for s in STOPS if s["name"] == start_name)
end   = next(s["id"] for s in STOPS if s["name"] == end_name)

wps = [s["id"] for s in STOPS if s["name"] in waypoint_names]


# Prevent same start/end
if start == end:
    st.sidebar.error("Start and End stops must be different.")
    st.stop()


# ─────────────────────────────────────────────────────────
# Cache Optimizer (prevents reloading models)
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models...")
def get_optimizer(hour, is_fog, is_rain):
    from engine.optimizer import SmartRouteOptimizer
    return SmartRouteOptimizer(hour=hour, is_fog=is_fog, is_rain=is_rain)


# ─────────────────────────────────────────────────────────
# Optimize Button
# ─────────────────────────────────────────────────────────
if st.sidebar.button("Optimize route", type="primary"):

    try:

        opt = get_optimizer(hour, is_fog, is_rain)

        with st.spinner("Running AI Routing Engine..."):

            out = opt.optimize(
                start,
                end,
                waypoints=wps
            )
            
            # Off-peak comparison
            if hour != 14:
                opt_offpeak = get_optimizer(14, is_fog, is_rain)
                out_off = opt_offpeak.optimize(start, end, waypoints=wps)
                out["offpeak_time"] = out_off["travel_time_min"]
                out["offpeak_route_names"] = out_off["route_names"]
            else:
                out["offpeak_time"] = None
            
            # Calculate greedy TSP for fair baseline
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
            
            out["greedy_time"] = round(float(greedy_dt), 2)
            out["greedy_route"] = greedy_route
            out["greedy_improvement"] = round((greedy_dt - out['travel_time_min']) / greedy_dt * 100, 1) if greedy_dt > 0 else 0.0
            
            st.session_state.result = out

    except FileNotFoundError as e:

        st.error(
            f"Model not trained yet: {e}\n\nRun `python main.py` first."
        )


# ─────────────────────────────────────────────────────────
# Display Results (persistent)
# ─────────────────────────────────────────────────────────
result = st.session_state.result

if result is not None:

    # ── Metrics ───────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "AI Route Time",
        f"{result['travel_time_min']} min"
    )

    c2.metric(
        "Greedy TSP Time",
        f"{result['greedy_time']} min"
    )

    c3.metric(
        "Time Saved",
        f"{result['greedy_improvement']}%"
    )

    c4.metric(
        "AI Runtime",
        f"{result['runtime_ms']} ms"
    )


    # ── Traffic Impact ───────────────────────────────────────────
    if result.get("offpeak_time"):
        st.subheader("Traffic Impact (vs Off-Peak 2 PM)")
        diff_mins = round(result['travel_time_min'] - result['offpeak_time'], 1)
        if result["route_names"] != result["offpeak_route_names"]:
            st.warning(f"⚠️ **AI Changed Route!** Traffic was too heavy, so the AI rerouted to save time.")
            st.markdown(f"**Off-Peak Route:** `{' → '.join(result['offpeak_route_names'])}` ({result['offpeak_time']} min)")
        else:
            st.info(f"🛣️ **Route Unchanged**. Travel time increased by {diff_mins} mins due to traffic.")


    # ── Route Text ───────────────────────────────────────
    st.subheader("Route Comparison (AI vs Greedy)")

    st.markdown(f"**🤖 AI Route:** `{' → '.join(result['route_names'])}`")
    st.markdown(f"**🐢 Greedy Route:** `{' → '.join([STOPS[i]['name'] for i in result['greedy_route']])}`")


    # ── Map ──────────────────────────────────────────────
    st.subheader("Route map")

    m = folium.Map(
        location=[28.62, 77.21],
        zoom_start=11,
        tiles="CartoDB positron"
    )

    coords = [
        (STOPS[i]["lat"], STOPS[i]["lon"])
        for i in result["route"]
    ]

    folium.PolyLine(
        coords,
        color="#534AB7",
        weight=5,
        opacity=0.85
    ).add_to(m)

    for rank, stop_id in enumerate(result["route"]):

        s = STOPS[stop_id]

        color = (
            "green" if rank == 0
            else "red" if rank == len(result["route"]) - 1
            else "blue"
        )

        folium.CircleMarker(
            location=[s["lat"], s["lon"]],
            radius=7,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=s["name"],
        ).add_to(m)

    st_folium(m, width=900, height=480)


