"""Main Streamlit application for car park simulation."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.car_park import CarPark, BayType, BayStatus
from src.models.vehicle import Vehicle, VisitPurpose
from src.simulation import SimulationEngine, ASSIGNMENT_STRATEGIES

# Color schemes
PURPOSE_COLORS = {
    "shopping": "#4CAF50",
    "dining": "#FF9800", 
    "entertainment": "#E91E63",
    "medical": "#2196F3",
    "emergency": "#F44336",
    "quick_errand": "#9C27B0",
    "commute": "#607D8B"
}

LEVEL_COLORS = {
    -1: "#5C6BC0",
    0: "#26A69A",
    1: "#FFA726"
}

# Page configuration
st.set_page_config(
    page_title="WPARK Car Park Simulation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .bay-available { background-color: #90EE90; }
    .bay-occupied { background-color: #FF6B6B; }
    .bay-bluebadge { background-color: #4169E1; }
    .bay-parent { background-color: #DDA0DD; }
    .bay-ev { background-color: #32CD32; }
    
    .level-header {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'engine' not in st.session_state:
        st.session_state.engine = SimulationEngine()
    if 'auto_run' not in st.session_state:
        st.session_state.auto_run = False
    if 'speed' not in st.session_state:
        st.session_state.speed = 1.0
    if 'occupancy_history' not in st.session_state:
        st.session_state.occupancy_history = []
    if 'last_history_time' not in st.session_state:
        st.session_state.last_history_time = 0


def render_sidebar():
    """Render sidebar controls."""
    st.sidebar.title("🚗 Simulation Controls")
    
    engine = st.session_state.engine
    
    # Run controls
    st.sidebar.subheader("⏱️ Time Control")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶️ Step", use_container_width=True):
            engine.step(st.session_state.speed)
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            engine.reset()
            st.session_state.occupancy_history = []
            st.session_state.last_history_time = 0
    
    # Auto-run toggle
    st.session_state.auto_run = st.sidebar.checkbox(
        "Auto-Run Simulation",
        value=st.session_state.auto_run
    )
    
    # Speed control
    st.session_state.speed = st.sidebar.slider(
        "Time Step (minutes)",
        min_value=0.5,
        max_value=10.0,
        value=1.0,
        step=0.5
    )
    
    # Configuration
    st.sidebar.subheader("⚙️ Configuration")
    
    engine.arrival_rate = st.sidebar.slider(
        "Arrival Rate (cars/min)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5
    )
    
    engine.assignment_strategy = st.sidebar.selectbox(
        "Assignment Strategy",
        options=list(ASSIGNMENT_STRATEGIES.keys()),
        index=list(ASSIGNMENT_STRATEGIES.keys()).index(engine.assignment_strategy)
    )
    
    # Legend
    st.sidebar.subheader("🎨 Bay Legend")
    st.sidebar.markdown("""
    - 🟢 **Available** - Standard bay
    - 🔴 **Occupied** - In use
    - 🔵 **Blue Badge** - Accessible parking
    - 🟣 **Parent & Child** - Family parking
    - 🟡 **EV** - Electric vehicle charging
    """)


def render_metrics(engine: SimulationEngine):
    """Render top metrics row."""
    state = engine.get_state_summary()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "⏰ Simulation Time",
            f"{int(state['current_time'])} min",
            f"{int(state['current_time'] // 60)}h {int(state['current_time'] % 60)}m"
        )
    
    with col2:
        st.metric(
            "🚗 Occupancy",
            f"{state['occupancy_rate']:.1%}",
            f"{state['total_occupied']}/{state['total_capacity']}"
        )
    
    with col3:
        st.metric(
            "📥 Arrivals",
            state['total_arrivals']
        )
    
    with col4:
        st.metric(
            "📤 Departures",
            state['total_departures']
        )
    
    with col5:
        st.metric(
            "⏳ Waiting",
            state['waiting_queue'],
            f"Denied: {state['total_denied']}"
        )


def get_bay_display_char(bay) -> str:
    """Get character to display for a bay."""
    if bay.status == BayStatus.OCCUPIED:
        return "🚗"
    elif bay.bay_type == BayType.BLUE_BADGE:
        return "♿"
    elif bay.bay_type == BayType.PARENT_CHILD:
        return "👶"
    elif bay.bay_type == BayType.EV:
        return "⚡"
    else:
        return "⬜"


def render_level_grid(level, show_details: bool = False):
    """Render a parking level as a grid."""
    st.markdown(f"### {level.name}")
    
    # Level stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Capacity:** {level.capacity}")
    with col2:
        st.write(f"**Occupied:** {level.occupied_count}")
    with col3:
        st.write(f"**Available:** {level.available_count}")
    
    # Shops on this floor
    if level.nearby_shops:
        st.caption(f"🏪 Shops: {', '.join(level.nearby_shops[:5])}{'...' if len(level.nearby_shops) > 5 else ''}")
    
    # Grid display
    row_labels = sorted(set(bay.row for bay in level.bays))
    bays_per_row = max(bay.position for bay in level.bays)
    
    # Create grid
    grid_html = "<div style='font-family: monospace; line-height: 1.8;'>"
    
    for row in row_labels:
        row_bays = [bay for bay in level.bays if bay.row == row]
        row_bays.sort(key=lambda b: b.position)
        
        row_html = f"<span style='color: #666;'>{row}:</span> "
        for bay in row_bays:
            color = bay.get_color()
            title = f"{bay.id} - {bay.bay_type.value} - {bay.status.value}"
            if bay.occupied_by:
                title += f" (Vehicle: {bay.occupied_by})"
            row_html += f"<span title='{title}' style='background-color: {color}; padding: 2px 4px; margin: 1px; border-radius: 2px; cursor: pointer;'>{get_bay_display_char(bay)}</span>"
        
        grid_html += row_html + "<br>"
    
    grid_html += "</div>"
    st.markdown(grid_html, unsafe_allow_html=True)
    
    # Occupancy bar
    occupancy = level.occupancy_rate
    st.progress(occupancy, text=f"Occupancy: {occupancy:.1%}")


def render_car_park_view(engine: SimulationEngine):
    """Render the car park visualization."""
    st.header("🏢 Car Park Overview")
    
    # Tab for each level
    tabs = st.tabs([level.name for level in engine.car_park.levels])
    
    for i, tab in enumerate(tabs):
        with tab:
            render_level_grid(engine.car_park.levels[i])


def render_vehicle_list(engine: SimulationEngine):
    """Render list of active vehicles."""
    st.header("🚙 Active Vehicles")
    
    if not engine.active_vehicles:
        st.info("No vehicles currently parked.")
        return
    
    # Show first 10 vehicles
    vehicles = list(engine.active_vehicles.values())[:10]
    
    for vehicle in vehicles:
        with st.expander(f"🚗 {vehicle.id} - {vehicle.purpose.value}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Size:** {vehicle.size.value}")
                st.write(f"**Purpose:** {vehicle.purpose.value}")
                st.write(f"**Mobility:** {vehicle.mobility.value}")
            with col2:
                st.write(f"**Bay:** {vehicle.assigned_bay_id}")
                st.write(f"**Target Shop:** {vehicle.target_shop or 'None'}")
                stay = engine.current_time - vehicle.arrival_time
                st.write(f"**Stay:** {stay:.0f}/{vehicle.estimated_stay_minutes:.0f} min")
    
    if len(engine.active_vehicles) > 10:
        st.caption(f"... and {len(engine.active_vehicles) - 10} more vehicles")


def render_statistics(engine: SimulationEngine):
    """Render statistics view with Plotly charts."""
    st.header("📊 Statistics")
    
    stats = engine.stats
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Arrivals by Purpose")
        if stats.arrivals_by_purpose:
            df = pd.DataFrame([
                {"Purpose": k.replace("_", " ").title(), "Count": v, "purpose_key": k}
                for k, v in stats.arrivals_by_purpose.items()
            ])
            colors = [PURPOSE_COLORS.get(row["purpose_key"], "#888888") for _, row in df.iterrows()]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=df["Purpose"],
                    y=df["Count"],
                    marker_color=colors,
                    text=df["Count"],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=40),
                xaxis_title="",
                yaxis_title="Arrivals",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No arrivals yet")
    
    with col2:
        st.subheader("Arrivals by Level")
        if stats.arrivals_by_level:
            df = pd.DataFrame([
                {"Level": f"Level {k}", "Count": v, "level_num": k}
                for k, v in stats.arrivals_by_level.items()
            ])
            colors = [LEVEL_COLORS.get(row["level_num"], "#888888") for _, row in df.iterrows()]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=df["Level"],
                    y=df["Count"],
                    marker_color=colors,
                    text=df["Count"],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=40),
                xaxis_title="",
                yaxis_title="Arrivals",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No arrivals yet")
    
    # Occupancy over time chart
    if hasattr(st.session_state, 'occupancy_history') and st.session_state.occupancy_history:
        st.subheader("Occupancy Over Time")
        history_df = pd.DataFrame(st.session_state.occupancy_history)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_df["time"],
            y=history_df["occupancy"] * 100,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#4CAF50', width=2),
            fillcolor='rgba(76, 175, 80, 0.3)'
        ))
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=20, b=40),
            xaxis_title="Time (minutes)",
            yaxis_title="Occupancy %",
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary stats
    st.subheader("Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Stay Duration", f"{stats.avg_stay_duration:.1f} min")
    with col2:
        st.metric("Peak Occupancy", f"{stats.peak_occupancy:.1%}")
    with col3:
        st.metric("Denied Entry", stats.total_denied)
    with col4:
        st.metric("Queue Length", len(engine.waiting_queue))


def render_shop_directory(engine: SimulationEngine):
    """Render shop directory."""
    st.header("🏪 Shop Directory")
    
    # Group shops by floor
    shops_by_floor = {}
    for shop in engine.shops:
        if shop.floor not in shops_by_floor:
            shops_by_floor[shop.floor] = []
        shops_by_floor[shop.floor].append(shop)
    
    for floor in sorted(shops_by_floor.keys()):
        shops = shops_by_floor[floor]
        level = engine.car_park.get_level(floor)
        level_name = level.name if level else f"Floor {floor}"
        
        with st.expander(f"📍 {level_name} ({len(shops)} shops)"):
            cols = st.columns(3)
            for i, shop in enumerate(shops):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div style='background-color: {shop.get_display_color()}20; 
                                padding: 10px; 
                                border-radius: 5px; 
                                margin: 5px 0;
                                border-left: 4px solid {shop.get_display_color()};'>
                        <strong>{shop.name}</strong><br>
                        <small>{shop.category.value.title()}</small><br>
                        <small>⏱️ ~{shop.avg_visit_minutes:.0f} min visit</small>
                    </div>
                    """, unsafe_allow_html=True)


def update_occupancy_history(engine: SimulationEngine):
    """Track occupancy over time (sample every 5 sim minutes)."""
    current_time = engine.current_time
    if current_time - st.session_state.last_history_time >= 5:
        st.session_state.occupancy_history.append({
            "time": current_time,
            "occupancy": engine.car_park.overall_occupancy
        })
        st.session_state.last_history_time = current_time
        # Keep last 200 points
        if len(st.session_state.occupancy_history) > 200:
            st.session_state.occupancy_history = st.session_state.occupancy_history[-200:]


def main():
    """Main application entry point."""
    init_session_state()
    
    st.title("🚗 WPARK Car Park Simulation")
    st.caption("Neural Network Training Environment - Front-End Simulation")
    
    # Sidebar
    render_sidebar()
    
    engine = st.session_state.engine
    
    # Top metrics
    render_metrics(engine)
    
    st.divider()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏢 Car Park View",
        "🚙 Vehicles",
        "📊 Statistics",
        "🏪 Shops"
    ])
    
    with tab1:
        render_car_park_view(engine)
    
    with tab2:
        render_vehicle_list(engine)
    
    with tab3:
        render_statistics(engine)
    
    with tab4:
        render_shop_directory(engine)
    
    # Auto-run logic with occupancy tracking
    if st.session_state.auto_run:
        engine.step(st.session_state.speed)
        update_occupancy_history(engine)
        time.sleep(0.3)  # Faster refresh
        st.rerun()


if __name__ == "__main__":
    main()
