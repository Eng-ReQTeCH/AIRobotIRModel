import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Load Plotly for the 3d plot
import plotly.graph_objects as go

# Load Matplotlib for 2D Plots
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Imports for Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Define the new accent color
ACCENT_BLUE = "#4a90e2"

# STREAMLIT CONFIG & STYLE
st.set_page_config(
    page_title="AI Robot IK Solver",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Elegance"
st.markdown(f"""
    <style>
    /* Global App Background: Deep Charcoal Gray */
    .stApp {{
        background-color: #1e2126; 
    }}

    /* Sidebar Background: Rich, Cool Blue Accent */
    section[data-testid="stSidebar"] {{
        background-color: #2b3042;
    }}

    /* Ensure all main text, titles, and subheaders are white */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .st-bf, .st-bb {{
        color: white !important;
    }}

    /* Metrics Styling */
    div[data-testid="stMetricValue"] {{
        font-size: 1.2rem;
        color: {ACCENT_BLUE}; /* Blue for values */
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.9rem;
        color: #ffffff; 
    }}

    /* Button Styling */
    div.stButton > button {{
        width: 100%;
        background-color: {ACCENT_BLUE}; 
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
    }}

    /* Toggle Switch Styling (Turns Blue when ON) */
    .st-dg > label > div[data-testid="stToggle"] > div > div:first-child {{
        background-color: {ACCENT_BLUE} !important;
        border-color: {ACCENT_BLUE} !important;
    }}
    .st-dg > label > div[data-testid="stToggle"] > div > div:last-child {{
        background-color: white !important; 
    }}

    /* Table Styling */
    .stDataFrame {{
        border: 1px solid #333;
        border-radius: 5px;
    }}

    /* Streamlit input labels */
    .st-dl {{
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True)


# ROBOT LOGIC CLASS & UTILITIES
class RobotArm3DOF:
    def __init__(self, l1=2.0, l2=2.0, l3=2.0):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    def forward_kinematics(self, theta):
        t1, t2, t3 = theta
        r = self.l2 * np.cos(t2) + self.l3 * np.cos(t2 + t3)
        x = np.cos(t1) * r
        y = np.sin(t1) * r
        z = self.l1 + self.l2 * np.sin(t2) + self.l3 * np.sin(t2 + t3)
        return np.array([x, y, z])

    def get_joint_positions(self, theta):
        t1, t2, t3 = theta

        # J0 (Base)
        x0, y0, z0 = 0, 0, 0
        # J1 (Shoulder)
        x1, y1, z1 = 0, 0, self.l1
        # J2 (Elbow)
        r2 = self.l2 * np.cos(t2)
        x2 = np.cos(t1) * r2
        y2 = np.sin(t1) * r2
        z2 = self.l1 + self.l2 * np.sin(t2)
        # J3 (End Effector)
        r3 = self.l2 * np.cos(t2) + self.l3 * np.cos(t2 + t3)
        x3 = np.cos(t1) * r3
        y3 = np.sin(t1) * r3
        # Using sine to match the FK definition.
        z3 = self.l1 + self.l2 * np.sin(t2) + self.l3 * np.sin(t2 + t3)

        return np.array([[x0, x1, x2, x3], [y0, y1, y2, y3], [z0, z1, z2, z3]])


def angles_to_sincos(angles):
    """Converts 3 angles to 6 features [sin(t1..3), cos(t1..3)]"""
    sines = np.sin(angles)
    cosines = np.cos(angles)
    return np.hstack([sines, cosines])


def sincos_to_angles(sincos_features):
    """Converts 6 features back to 3 angles using arctan2."""
    if sincos_features.ndim == 1:
        sincos_features = sincos_features.reshape(1, -1)

    t1 = np.arctan2(sincos_features[:, 0], sincos_features[:, 3])
    t2 = np.arctan2(sincos_features[:, 1], sincos_features[:, 4])
    t3 = np.arctan2(sincos_features[:, 2], sincos_features[:, 5])

    return np.vstack([t1, t2, t3]).T


# PERSISTENT AI TRAINING
MODEL_FILE = "robot_brain.pkl"
DATA_FILE = "robot_train_data.npz"
TEST_DATA_FILE = "robot_test_results.npz"
SCALER_FILE = "robot_scaler.pkl"


@st.cache_resource
def get_model_and_robot():
    """Load model if exists, otherwise train a new one."""
    robot = RobotArm3DOF(l1=2, l2=2, l3=2)
    test_metrics = {}
    scaler = None

    # 1. Try to load existing model and data
    if os.path.exists(MODEL_FILE) and os.path.exists(DATA_FILE) and os.path.exists(TEST_DATA_FILE) and os.path.exists(
            SCALER_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            data = np.load(DATA_FILE)
            y_positions = data['y_positions']
            test_data = np.load(TEST_DATA_FILE)

            # Load all test metrics
            test_metrics = {
                'X_test_scaled': test_data['X_test_scaled'],
                'y_test_sincos': test_data['y_test_sincos'],
                'y_pred_sincos': test_data['y_pred_sincos'],
                'r2_score': test_data['r2_score'].item(),  # Extract scalar value.
                'mse': test_data['mse'].item()
            }
            return model, robot, y_positions, test_metrics, scaler, "Loaded from Disk"
        except Exception:
            # If load fails, proceed to retraining
            pass

    # 2. Train new model if no file exists

    # Training Parameters
    num_samples = 100000
    TRAIN_SIZE_TARGET = 15000
    TEST_SIZE_TARGET = 5000

    # Generate random joint angles and calculate corresponding Cartesian positions
    X_angles_raw = np.random.uniform(
        low=[-np.pi, -np.pi / 2, -np.pi / 2],
        high=[np.pi, np.pi / 2, np.pi / 2],
        size=(num_samples, 3)
    )
    y_positions_raw = np.array([robot.forward_kinematics(angles) for angles in X_angles_raw])

    # Filter for 'Elbow Down' configuration: T3 must be negative.
    elbow_down_mask = X_angles_raw[:, 2] < 0

    X_angles_filtered = X_angles_raw[elbow_down_mask]
    y_positions_filtered = y_positions_raw[elbow_down_mask]

    # Adjust sizes after filtering
    num_filtered = X_angles_filtered.shape[0]

    # Check if enough data is available after filtering
    if num_filtered < TRAIN_SIZE_TARGET + TEST_SIZE_TARGET:
        st.error("Insufficient data after filtering for 'Elbow Down' configuration.")
        return None, robot, None, None, None, "Training Failed: Not enough data."

    # Final data slicing for training and testing
    X_angles_full = X_angles_filtered[:TRAIN_SIZE_TARGET + TEST_SIZE_TARGET]
    y_positions_full = y_positions_filtered[:TRAIN_SIZE_TARGET + TEST_SIZE_TARGET]

    # Feature Transformation (y-targets)
    y_sincos_full_6d = angles_to_sincos(X_angles_full)

    # Only predict Sin/Cos for T2 and T3 (Indices 1, 2, 4, 5). T1 is solved analytically.
    y_sincos_full = y_sincos_full_6d[:, [1, 2, 4, 5]]

    # Split the dataset
    X_train_pos = y_positions_full[:TRAIN_SIZE_TARGET]
    y_train_sincos = y_sincos_full[:TRAIN_SIZE_TARGET]

    X_test_pos = y_positions_full[TRAIN_SIZE_TARGET:]
    y_test_sincos = y_sincos_full[TRAIN_SIZE_TARGET:]
    y_test_angles = X_angles_full[TRAIN_SIZE_TARGET:]

    # Data Normalization (X-inputs)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pos)
    X_test_scaled = scaler.transform(X_test_pos)

    # KNN Regression
    model = KNeighborsRegressor(n_neighbors=5, weights='distance')
    model.fit(X_train_scaled, y_train_sincos)

    # Evaluate performance on test set
    y_pred_sincos_4d = model.predict(X_test_scaled)

    # Reconstruct the 6D SinCos vectors for plotting/MSE
    t1_sincos_test = y_sincos_full_6d[TRAIN_SIZE_TARGET:, [0, 3]]
    y_test_sincos_6d = np.hstack(
        [t1_sincos_test[:, 0:1], y_test_sincos[:, 0:2], t1_sincos_test[:, 1:2], y_test_sincos[:, 2:4]])

    # T1 predicted values are calculated directly from X, Y test input positions
    x_test = X_test_pos[:, 0]
    y_test = X_test_pos[:, 1]
    t1_pred_angles = np.arctan2(y_test, x_test)
    t1_pred_sincos = np.vstack([np.sin(t1_pred_angles), np.cos(t1_pred_angles)]).T

    # Predicted 6D vector: T1 (calculated) + T2/T3 (predicted)
    y_pred_sincos_6d = np.hstack(
        [t1_pred_sincos[:, 0:1], y_pred_sincos_4d[:, 0:2], t1_pred_sincos[:, 1:2], y_pred_sincos_4d[:, 2:4]])

    # Convert predictions back to angles for evaluation
    y_pred_angles = sincos_to_angles(y_pred_sincos_6d)

    # Calculate R2 (T2/T3 prediction) and MSE (all 3 reconstructed angles)
    r2_score = model.score(X_test_scaled, y_test_sincos)
    mse = mean_squared_error(y_test_angles, y_pred_angles)

    # Package test metrics
    test_metrics = {
        'X_test_scaled': X_test_scaled,
        'y_test_sincos': y_test_sincos_6d,
        'y_pred_sincos': y_pred_sincos_6d,
        'r2_score': r2_score,
        'mse': mse
    }

    # 3. Save model and data to disk
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    np.savez(DATA_FILE, y_positions=y_positions_full)

    np.savez(TEST_DATA_FILE,
             X_test_scaled=X_test_scaled,
             y_test_sincos=y_test_sincos_6d,
             y_pred_sincos=y_pred_sincos_6d,
             r2_score=r2_score,
             mse=mse)

    return model, robot, y_positions_full, test_metrics, scaler, "Newly Trained (T2/T3 only) & Saved"


# HELPER: PATH GENERATION
def generate_path(start_joints, end_joints, steps=20):
    """Interpolates path and returns full dataframe of trajectory"""
    joints_path = np.linspace(start_joints, end_joints, steps)
    return joints_path


# PLOTTING FUNCTIONS

def configure_3d_layout():
    """Returns the standard 3D scene configuration."""
    limit = 4.5
    return dict(
        xaxis=dict(title='X', range=[-limit, limit], backgroundcolor="#2b3042", gridcolor="#444", showbackground=True,
                   zerolinecolor="#666"),
        yaxis=dict(title='Y', range=[-limit, limit], backgroundcolor="#2b3042", gridcolor="#444", showbackground=True,
                   zerolinecolor="#666"),
        zaxis=dict(title='Z', range=[0, 6], backgroundcolor="#2b3042", gridcolor="#444", showbackground=True,
                   zerolinecolor="#666"),
        aspectmode='cube'
    )


def plot_robot_static_plotly(robot, predicted_joints, target_x, target_y, target_z, path_joints, show_trace):
    """Generates an interactive 3D Plotly figure of the final robot position."""

    # Get joint coordinates
    final_xyz = robot.get_joint_positions(predicted_joints)

    fig = go.Figure()

    # Plot the physical robot links (Line Trace)
    fig.add_trace(go.Scatter3d(
        x=final_xyz[0], y=final_xyz[1], z=final_xyz[2],
        mode='lines+markers',
        line=dict(color=ACCENT_BLUE, width=8),
        marker=dict(size=6, color=ACCENT_BLUE, symbol='circle'),
        name='Robot Arm'
    ))

    # Show the target location
    fig.add_trace(go.Scatter3d(
        x=[target_x], y=[target_y], z=[target_z],
        mode='markers',
        marker=dict(size=15, color='#ff4b4b', symbol='diamond'),
        name='Target Position'
    ))

    # Draw the calculated trajectory (if enabled)
    if show_trace:
        trace_x = [robot.forward_kinematics(j)[0] for j in path_joints]
        trace_y = [robot.forward_kinematics(j)[1] for j in path_joints]
        trace_z = [robot.forward_kinematics(j)[2] for j in path_joints]

        fig.add_trace(go.Scatter3d(
            x=trace_x, y=trace_y, z=trace_z,
            mode='lines',
            line=dict(color='cyan', width=2, dash='dash'),
            name='Path Trace'
        ))

    # Apply 3D plot layout (Dark Theme)
    fig.update_layout(
        title_text='Final Robot Position',
        height=600,
        scene=configure_3d_layout(),
        paper_bgcolor="#1e2126",
        plot_bgcolor="#1e2126",
        font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=50),
        showlegend=True
    )

    return fig


def plot_robot_animation_plotly(robot, target_x, target_y, target_z, path_joints):
    """Generates an interactive 3D Plotly figure with animation frames."""

    # Initial state (Home position)
    home_xyz = robot.get_joint_positions(path_joints[0])

    # Base Figure Setup
    fig = go.Figure(
        data=[
            # 1. Robot Arm State (The trace that moves)
            go.Scatter3d(
                x=home_xyz[0], y=home_xyz[1], z=home_xyz[2],
                mode='lines+markers',
                line=dict(color=ACCENT_BLUE, width=8),
                marker=dict(size=6, color=ACCENT_BLUE, symbol='circle'),
                name='Robot Arm',
                uid='arm_trace'
            ),
            # 2. Target Position (Static)
            go.Scatter3d(
                x=[target_x], y=[target_y], z=[target_z],
                mode='markers',
                marker=dict(size=15, color='#ff4b4b', symbol='diamond'),
                name='Target Position'
            ),
            # 3. Path History Trace (The cumulative line)
            go.Scatter3d(
                x=[], y=[], z=[],
                mode='lines',
                line=dict(color='cyan', width=2, dash='dash'),
                name='Path Trace',
                uid='path_history'
            )
        ],
        layout={
            'title': 'Robot Path Animation (Home to Target)',
            'height': 600,
            'scene': configure_3d_layout(),
            'paper_bgcolor': "#1e2126",
            'plot_bgcolor': "#1e2126",
            'font': dict(color="white"),
            'margin': dict(l=0, r=0, b=0, t=50),
            'showlegend': True
        }
    )

    # Create Frames for Animation
    frames = []
    trace_x, trace_y, trace_z = [], [], []

    for i, joints in enumerate(path_joints):
        current_xyz = robot.get_joint_positions(joints)

        # Add current end-effector position to the cumulative trace
        end_effector_pos = robot.forward_kinematics(joints)
        trace_x.append(end_effector_pos[0])
        trace_y.append(end_effector_pos[1])
        trace_z.append(end_effector_pos[2])

        frame = go.Frame(
            data=[
                # Update the Arm Trace
                go.Scatter3d(
                    x=current_xyz[0], y=current_xyz[1], z=current_xyz[2],
                    uid='arm_trace'
                ),
                # Update the Path History Trace
                go.Scatter3d(
                    x=trace_x, y=trace_y, z=trace_z,
                    uid='path_history'
                )
            ],
            name=str(i)
        )
        frames.append(frame)

    fig.frames = frames

    # Add Play/Pause Buttons
    animation_settings = {
        'frame': {'duration': 50, 'redraw': True},  # 50ms per step
        'fromcurrent': True,
        'transition': {'duration': 0, 'easing': 'linear'}
    }

    fig.update_layout(
        updatemenus=[{
            'buttons': [
                dict(label='Play',
                     method='animate',
                     args=[None, animation_settings]),
                dict(label='Pause',
                     method='animate',
                     args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                                    'transition': {'duration': 0}}])
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }]
    )

    return fig


def plot_training_data(positions):
    """Generates 3D scatter plot of the robot's workspace."""

    fig = plt.figure(figsize=(12, 10))

    # 3D WORKSPACE PLOT
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                c=positions[:, 2], cmap='winter', s=5, alpha=0.6)

    # Dark Mode Styling for all subplots
    for ax in [ax1]:
        ax.set_facecolor('#1e2126')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        ax.set_title("3D Workspace Coverage (X, Y, Z)", color='white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_zlabel('Z Position')
    ax1.set_xlim(-4.5, 4.5)
    ax1.set_ylim(-4.5, 4.5)
    ax1.set_zlim(0, 6)

    # 2D PROJECTIONS
    # X-Z Projection (Side View)
    ax2 = fig.add_subplot(222)
    ax2.scatter(positions[:, 0], positions[:, 2], s=5, alpha=0.6, c=positions[:, 2], cmap='winter')
    ax2.set_xlabel('X Position', color='white')
    ax2.set_ylabel('Z Position', color='white')
    ax2.set_title('X-Z Projection (Side View)', color='white')
    ax2.tick_params(colors='white')
    ax2.set_facecolor('#1e2126')
    ax2.grid(True, linestyle=':', alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    # Y-Z Projection (Side View)
    ax3 = fig.add_subplot(223)
    ax3.scatter(positions[:, 1], positions[:, 2], s=5, alpha=0.6, c=positions[:, 2], cmap='winter')
    ax3.set_xlabel('Y Position', color='white')
    ax3.set_ylabel('Z Position', color='white')
    ax3.set_title('Y-Z Projection (Side View)', color='white')
    ax3.tick_params(colors='white')
    ax3.set_facecolor('#1e2126')
    ax3.grid(True, linestyle=':', alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')

    # X-Y Projection (Top View)
    ax4 = fig.add_subplot(224)
    ax4.scatter(positions[:, 0], positions[:, 1], s=5, alpha=0.6, c=positions[:, 2], cmap='winter')
    ax4.set_xlabel('X Position', color='white')
    ax4.set_ylabel('Y Position', color='white')
    ax4.set_title('X-Y Projection (Top View)', color='white')
    ax4.tick_params(colors='white')
    ax4.set_facecolor('#1e2126')
    ax4.grid(True, linestyle=':', alpha=0.3)
    ax4.set_aspect('equal', adjustable='box')

    fig.patch.set_facecolor('#1e2126')
    plt.tight_layout()

    return fig


def plot_regression_performance(test_metrics):
    """Plots predicted vs true joint angles."""

    # Convert sine/cosine features back to angles (in radians)
    y_test_angles = sincos_to_angles(test_metrics['y_test_sincos'])
    y_pred_angles = sincos_to_angles(test_metrics['y_pred_sincos'])

    # Convert to degrees for readability
    y_test_deg = np.degrees(y_test_angles)
    y_pred_deg = np.degrees(y_pred_angles)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    joint_names = ["Base ($\Theta_1$)", "Shoulder ($\Theta_2$)", "Elbow ($\Theta_3$)"]

    # Determine plot limits
    max_angle = np.ceil(np.max(np.abs(y_test_deg)))
    limits = [-max_angle - 10, max_angle + 10]

    for i in range(3):
        ax = axes[i]
        # Scatter plot: Predicted vs True
        ax.scatter(y_test_deg[:, i], y_pred_deg[:, i],
                   alpha=0.4, s=20, color=ACCENT_BLUE, label='Predicted vs True')

        # Diagonal Line (Perfect Fit)
        ax.plot(limits, limits, color='#ff4b4b', linestyle='--', label='Ideal Prediction (y=x)')

        # Styling for Dark Mode
        ax.set_facecolor('#2b3042')
        ax.set_title(f"Joint {i + 1}: {joint_names[i]} Regression", color='white')
        ax.set_xlabel("True Angle (Degrees)", color='white')
        ax.set_ylabel("Predicted Angle (Degrees)", color='white')

        ax.tick_params(colors='white')
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.legend()

    fig.patch.set_facecolor('#1e2126')
    plt.tight_layout()
    return fig


# MAIN UI LAYOUT
def main():
    st.title("3-DOF Robot AI Path Planner")

    # Load Model and Data
    with st.spinner("Initializing AI Core..."):
        model, robot, training_positions, test_metrics, scaler, status_msg = get_model_and_robot()

    # Check if training failed
    if model is None:
        st.error("Please reset the AI Brain to attempt training again.")
        return

    # SIDEBAR CONTROLS
    st.sidebar.header("Target Coordinates")

    c1, c2, c3 = st.sidebar.columns(3)
    target_x = c1.number_input("X", value=1.5, step=0.1)
    target_y = c2.number_input("Y", value=1.5, step=0.1)
    target_z = c3.number_input("Z", value=3.0, step=0.1)

    st.sidebar.markdown("---")
    st.sidebar.header("View Mode")

    # New control: Toggle switch for animation
    # The default value should probably be False (show static plot first)
    animate_mode = st.sidebar.toggle("Show Path Animation (vs Static Result)", value=False)

    st.sidebar.markdown("---")
    st.sidebar.header("Settings")

    # The static trace switch is only relevant in static mode
    show_trace = st.sidebar.toggle("Show Path Trace on Static Plot", value=True, disabled=animate_mode)

    show_training_plots = st.sidebar.toggle("Show AI Training Workspace", value=False)
    show_performance_plots = st.sidebar.toggle("Show AI Model Performance", value=False)

    if st.sidebar.button("Reset AI Brain"):
        for f in [MODEL_FILE, DATA_FILE, TEST_DATA_FILE, SCALER_FILE]:
            if os.path.exists(f):
                os.remove(f)
        st.cache_resource.clear()
        st.rerun()

    # CALCULATIONS
    target_pos = np.array([[target_x, target_y, target_z]])

    try:
        # Scale the live input
        target_pos_scaled = scaler.transform(target_pos)

        # Predict the 4 sincos features (T2 and T3 only)
        predicted_sincos_4d = model.predict(target_pos_scaled)[0]

        # Calculate T1 directly from X, Y (This joint is easily solved)
        t1_pred_angle = np.arctan2(target_y, target_x)
        t1_pred_sincos = np.array([np.sin(t1_pred_angle), np.cos(t1_pred_angle)])

        # Reconstruct the full 6D SinCos vector
        predicted_sincos_6d = np.array([
            t1_pred_sincos[0],
            predicted_sincos_4d[0],
            predicted_sincos_4d[1],
            t1_pred_sincos[1],
            predicted_sincos_4d[2],
            predicted_sincos_4d[3]
        ])

        # Convert the full 6D sincos prediction back to 3 angles
        predicted_joints = sincos_to_angles(predicted_sincos_6d)[0]

    except Exception as e:
        st.error(f"Prediction Error: {e}. Ensure target coordinates are reachable.")
        return

    # Error Calculation
    actual_pos = robot.forward_kinematics(predicted_joints)
    error = np.linalg.norm(target_pos[0] - actual_pos)

    # Generate Path Data (From home to predicted joints)
    home_joints = np.array([0.0, 0.0, 0.0])
    path_joints = generate_path(home_joints, predicted_joints)

    # Create DataFrame for Table
    path_data = []
    for i, j in enumerate(path_joints):
        pos = robot.forward_kinematics(j)
        path_data.append({
            "Step": i + 1,
            "Base (θ1)": f"{np.degrees(j[0]):.1f}°",
            "Shoulder (θ2)": f"{np.degrees(j[1]):.1f}°",
            "Elbow (θ3)": f"{np.degrees(j[2]):.1f}°",
            "X": f"{pos[0]:.2f}",
            "Y": f"{pos[1]:.2f}",
            "Z": f"{pos[2]:.2f}"
        })
    df_path = pd.DataFrame(path_data)

    # TOP METRICS ROW
    st.markdown("### Live Telemetry")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Base Angle", f"{np.degrees(predicted_joints[0]):.1f}°")
    m2.metric("Shoulder Angle", f"{np.degrees(predicted_joints[1]):.1f}°")
    m3.metric("Elbow Angle", f"{np.degrees(predicted_joints[2]):.1f}°")
    m4.metric("Accuracy Error", f"{error:.4f}m", delta_color="inverse")

    st.markdown("---")

    # MAIN CONTENT GRID
    col_plot, col_data = st.columns([1.5, 1])

    with col_plot:

        if animate_mode:
            st.subheader("Interactive 3D Robot Simulation (Animation Mode)")
            # Generate Plotly animation figure
            plotly_fig = plot_robot_animation_plotly(
                robot,
                target_x,
                target_y,
                target_z,
                path_joints
            )
        else:
            st.subheader("Interactive 3D Robot Simulation (Static Result)")
            # Show the static result plot
            plotly_fig = plot_robot_static_plotly(
                robot,
                predicted_joints,
                target_x,
                target_y,
                target_z,
                path_joints,
                show_trace
            )

        # Render the interactive Plotly figure
        st.plotly_chart(plotly_fig, use_container_width=True)

    with col_data:
        st.subheader("Trajectory Data")
        st.dataframe(
            df_path,
            hide_index=True,
            use_container_width=True,
            height=400
        )

    # VISUALIZATION SECTIONS

    # 1. Training Workspace Plots
    if show_training_plots:
        st.markdown("## AI Model Training Workspace (Elbow Down only)")
        st.info(
            f"The model was trained on {training_positions.shape[0]} filtered data points. The robot links are L1={robot.l1}m, L2={robot.l2}m, L3={robot.l3}m, and the joint angles were constrained to the 'Elbow Down' configuration ($\Theta_3 < 0$) during training.")
        st.pyplot(plot_training_data(training_positions), use_container_width=True)

    # 2. Performance Plots
    if show_performance_plots:
        st.markdown("## AI Model Performance Analysis")
        m1, m2 = st.columns(2)
        m1.metric("R² Score (T2/T3 Prediction)", f"{test_metrics['r2_score']:.4f}", delta="Higher is better",
                  delta_color="normal")
        m2.metric("Angle Mean Squared Error (Total)", f"{test_metrics['mse']:.6f} rad²", delta="Lower is better",
                  delta_color="inverse")

        st.info(
            "These plots show how well the AI predicted the joint angles for 5,000 unseen test positions. A perfect prediction would fall directly on the diagonal red line.")
        st.pyplot(plot_regression_performance(test_metrics), use_container_width=True)

    st.markdown("---")
    st.caption(f"AI Core Status: {status_msg}. | Robot Kinematics: L1={robot.l1}, L2={robot.l2}, L3={robot.l3}")


if __name__ == '__main__':
    main()
