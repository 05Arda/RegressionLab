import numpy as np
import matplotlib.pyplot as plt
import webbrowser
import os
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.patches import Rectangle
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# --- 1. CONFIGURATION ---
FIT_THRESHOLD = 0.20       
INIT_DEGREE = 1            
INIT_ALPHA = 0.0           
INIT_NOISE = 0.2           
INIT_DATA_SIZE = 30        
INIT_TEST_RATIO = 0.4      
INIT_DATASET_TYPE = 'Sine Wave'
INIT_GRAPH_MODE = 'Complexity'
INIT_USE_CV = False

DOC_FILENAME = "documentation.html"

# Global state
state = {
    'X_train': None, 'y_train_clean': None,
    'X_test': None, 'y_test': None,
    'true_X': None, 'true_y': None,
    'noise_base': None,
    'show_residuals': True,
    'last_tooltip_text': None,
    'dataset_type': INIT_DATASET_TYPE, 
    'manual_outliers': {'x': [], 'y': []},
    'graph_mode': INIT_GRAPH_MODE, 
    'use_cv': INIT_USE_CV 
}

# --- DATA GENERATION LOGIC ---
def generate_data_logic(n_samples, type_name):
    X = np.sort(np.random.rand(n_samples))
    y_clean = np.zeros_like(X)
    
    if type_name == 'Sine Wave':
        y_clean = np.cos(1.5 * np.pi * X)
        true_X = np.linspace(0, 1, 300)
        true_y = np.cos(1.5 * np.pi * true_X)
    elif type_name == 'Linear':
        y_clean = 2 * X - 1
        true_X = np.linspace(0, 1, 300)
        true_y = 2 * true_X - 1
    elif type_name == 'Step Function':
        y_clean = (X > 0.5).astype(float) * 2 - 1
        true_X = np.linspace(0, 1, 300)
        true_y = (true_X > 0.5).astype(float) * 2 - 1
    elif type_name == 'Heteroscedastic':
        y_clean = 2 * X
        true_X = np.linspace(0, 1, 300)
        true_y = 2 * true_X
        
    return X, y_clean, true_X[:, np.newaxis], true_y

def initialize_data(n_samples=INIT_DATA_SIZE, test_size=INIT_TEST_RATIO, new_seed=False):
    if new_seed: np.random.seed(None)
    else: np.random.seed(42)

    ds_type = state['dataset_type']
    X, y_clean, tX, ty = generate_data_logic(int(n_samples), ds_type)
    
    state['true_X'] = tX
    state['true_y'] = ty
    
    X_tr, X_te, y_tr_clean, y_te_clean = train_test_split(
        X[:, np.newaxis], y_clean, test_size=test_size, random_state=42 if not new_seed else None
    )
    
    sort_idx = X_te.flatten().argsort()
    state['X_test'] = X_te[sort_idx]
    state['y_test'] = y_te_clean[sort_idx]
    state['X_train'] = X_tr
    state['y_train_clean'] = y_tr_clean
    state['noise_base'] = np.random.randn(len(y_tr_clean))
    state['manual_outliers'] = {'x': [], 'y': []}

def apply_noise_and_outliers(noise_level):
    if state['dataset_type'] == 'Heteroscedastic':
        scaled_noise = state['noise_base'] * noise_level * (state['X_train'].flatten() * 3)
    else:
        scaled_noise = state['noise_base'] * noise_level
        
    y_noisy = state['y_train_clean'] + scaled_noise
    
    X_final = state['X_train']
    y_final = y_noisy
    
    if len(state['manual_outliers']['x']) > 0:
        out_X = np.array(state['manual_outliers']['x'])[:, np.newaxis]
        out_y = np.array(state['manual_outliers']['y'])
        X_final = np.vstack([X_final, out_X])
        y_final = np.concatenate([y_final, out_y])
        
    sort_idx = X_final.flatten().argsort()
    return X_final[sort_idx], y_final[sort_idx]

initialize_data(n_samples=INIT_DATA_SIZE, test_size=INIT_TEST_RATIO)

# --- LAYOUT SETUP ---
fig = plt.figure(figsize=(16, 10))
# Background color for the whole window
fig.patch.set_facecolor('#fafafa')
fig.suptitle("Ultimate ML Lab: Regression Simulator", fontsize=16, fontweight='bold', color='#333333', y=0.98)

# Grid: Adjusted to 1 row to maximize plot area.
# Width ratios: 2 (Main Plot) to 1 (Error Plot) -> Main is 2x larger
gs = GridSpec(1, 2, width_ratios=[2, 1])

ax_main = fig.add_subplot(gs[0, 0])      
ax_error = fig.add_subplot(gs[0, 1])     

# Bottom margin adjusted to 0.35 to allow enough space for panels without overlapping
# Left=0.05 and Right=0.98 to match the panels below
plt.subplots_adjust(left=0.05, right=0.98, bottom=0.35, top=0.92, wspace=0.15)

# --- VISUAL PANELS (BACKGROUND BOXES) ---
# Panel 1: MODEL PARAMETERS (Blue) - Left
# Extends from 0.05 to 0.29 (Width 0.24)
rect1 = Rectangle((0.02, 0.02), 0.27, 0.28, transform=fig.transFigure, color='#e3f2fd', zorder=-1, ec='#90caf9', lw=1)
fig.patches.append(rect1)
fig.text(0.03, 0.27, "1. MODEL HYPERPARAMETERS", fontsize=10, fontweight='bold', color='#1565c0')

# Panel 2: DATA GENERATION (Green) - Center
# Extends from 0.30 to 0.66 (Width 0.36)
rect2 = Rectangle((0.30, 0.02), 0.36, 0.28, transform=fig.transFigure, color='#e8f5e9', zorder=-1, ec='#a5d6a7', lw=1)
fig.patches.append(rect2)
fig.text(0.31, 0.27, "2. DATA GENERATION", fontsize=10, fontweight='bold', color='#2e7d32')

# Panel 3: ANALYSIS & TOOLS (Orange) - Right
# Extends from 0.67 to 0.98 (Width 0.31)
rect3 = Rectangle((0.67, 0.02), 0.31, 0.28, transform=fig.transFigure, color='#fff3e0', zorder=-1, ec='#ffcc80', lw=1)
fig.patches.append(rect3)
fig.text(0.68, 0.27, "3. ANALYSIS & TOOLS", fontsize=10, fontweight='bold', color='#ef6c00')

# --- PLOT ELEMENTS ---
scat_train = ax_main.scatter([], [], c='#1f77b4', alpha=0.6, s=40, label='Train Data')
scat_outliers = ax_main.scatter([], [], c='purple', marker='D', s=80, label='Manual Outliers', zorder=10)
scat_test = ax_main.scatter([], [], c='#d62728', marker='x', s=60, linewidth=2, label='Test Data')

line_model, = ax_main.plot([], [], c='#2ca02c', linewidth=3, label='Model Prediction')
line_truth, = ax_main.plot([], [], c='gray', linestyle='--', alpha=0.5, label='True Function')
poly_confidence = ax_main.fill_between([], [], [], color='#2ca02c', alpha=0.1, label='Confidence Interval')

residual_lines = [] 
line_err_train, = ax_error.plot([], [], c='#1f77b4', marker='o', label='Train Error')
line_err_test, = ax_error.plot([], [], c='#d62728', marker='s', label='Test/CV Error')

current_val_line = ax_error.axvline(x=1, color='#2ca02c', linestyle='--', alpha=0.8, label='Current Value')
optimal_val_line = ax_error.axvline(x=1, color='#ff7f0e', linestyle='-', linewidth=2, alpha=0.8, label='Optimal')
optimal_text = ax_error.text(0.5, 0.9, "", transform=ax_error.transAxes, color='#ff7f0e', fontweight='bold')

info_text = ax_main.text(0.02, 0.95, "", transform=ax_main.transAxes, bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))
tooltip = fig.text(0, 0, "", bbox=dict(facecolor='#ffffcc', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'), zorder=100, visible=False, fontsize=10)

# --- CALCULATION ---
def get_model(degree, alpha):
    return make_pipeline(PolynomialFeatures(degree), StandardScaler(), Ridge(alpha=alpha))

def calculate_complexity_curve(X_tr, y_tr, alpha, max_deg=15):
    train_errs, test_errs, degrees = [], [], range(1, max_deg + 1)
    X_te, y_te = state['X_test'], state['y_test']
    for d in degrees:
        model = get_model(d, alpha)
        if state['use_cv']:
            cv_score = cross_val_score(model, X_tr, y_tr, cv=5, scoring='neg_mean_squared_error')
            test_errs.append(np.sqrt(-cv_score.mean()))
            model.fit(X_tr, y_tr)
            train_errs.append(np.sqrt(mean_squared_error(y_tr, model.predict(X_tr))))
        else:
            model.fit(X_tr, y_tr)
            train_errs.append(np.sqrt(mean_squared_error(y_tr, model.predict(X_tr))))
            test_errs.append(np.sqrt(mean_squared_error(y_te, model.predict(X_te))))
    return list(degrees), train_errs, test_errs, "Model Complexity (Degree)"

def calculate_learning_curve(X_tr, y_tr, degree, alpha):
    train_errs, test_errs, sizes = [], [], []
    X_te, y_te = state['X_test'], state['y_test']
    n_points = len(X_tr)
    min_points = 5
    if n_points < min_points: return [], [], [], "Sample Size"
    step = max(1, n_points // 10)
    for size in range(min_points, n_points + 1, step):
        X_sub, y_sub = X_tr[:size], y_tr[:size]
        sizes.append(size)
        model = get_model(degree, alpha)
        model.fit(X_sub, y_sub)
        train_errs.append(np.sqrt(mean_squared_error(y_sub, model.predict(X_sub))))
        test_errs.append(np.sqrt(mean_squared_error(y_te, model.predict(X_te))))
    return sizes, train_errs, test_errs, "Sample Size (Training Data)"

# --- UPDATE ---
def update(val=None):
    degree = int(slider_degree.val)
    alpha = slider_alpha.val
    noise_lvl = slider_noise.val
    
    X_tr, y_tr = apply_noise_and_outliers(noise_lvl)
    X_te, y_te = state['X_test'], state['y_test']
    
    model = get_model(degree, alpha)
    model.fit(X_tr, y_tr)
    
    X_smooth = np.linspace(0, 1, 300)[:, np.newaxis]
    y_smooth = model.predict(X_smooth)
    
    scat_train.set_offsets(np.c_[X_tr, y_tr])
    scat_test.set_offsets(np.c_[X_te, y_te])
    
    if len(state['manual_outliers']['x']) > 0:
        scat_outliers.set_offsets(np.c_[state['manual_outliers']['x'], state['manual_outliers']['y']])
        scat_outliers.set_visible(True)
    else:
        scat_outliers.set_visible(False)

    line_model.set_data(X_smooth, y_smooth)
    line_truth.set_data(state['true_X'], state['true_y'])
    
    y_pred_tr = model.predict(X_tr)
    std_resid = np.std(y_tr - y_pred_tr)
    global poly_confidence
    poly_confidence.remove()
    poly_confidence = ax_main.fill_between(X_smooth.flatten(), 
                                           y_smooth - 1.96 * std_resid, 
                                           y_smooth + 1.96 * std_resid, 
                                           color='green', alpha=0.15, label='Confidence Interval')

    global residual_lines
    for line in residual_lines: line.remove()
    residual_lines = []
    if state['show_residuals']:
        for i in range(len(X_tr)):
            l, = ax_main.plot([X_tr[i], X_tr[i]], [y_tr[i], y_pred_tr[i]], color='#d62728', alpha=0.3, linewidth=1)
            residual_lines.append(l)

    if state['graph_mode'] == 'Complexity':
        x_vals, tr_errs, te_errs, x_label = calculate_complexity_curve(X_tr, y_tr, alpha)
        current_marker = degree
        if len(te_errs) > 0:
            optimal_val = x_vals[te_errs.index(min(te_errs))]
        else: optimal_val = 1
        if degree < optimal_val: status = "UNDERFITTING"
        elif degree == optimal_val: status = "OPTIMAL"
        else: status = "OVERFITTING"
    else:
        x_vals, tr_errs, te_errs, x_label = calculate_learning_curve(X_tr, y_tr, degree, alpha)
        current_marker = len(X_tr)
        optimal_val = len(X_tr)
        gap = abs(tr_errs[-1] - te_errs[-1]) if tr_errs else 0
        if gap > 0.2: status = "High Variance (Need More Data)"
        else: status = "Converged / Good Fit"

    line_err_train.set_data(x_vals, tr_errs)
    line_err_test.set_data(x_vals, te_errs)
    current_val_line.set_xdata([current_marker])
    optimal_val_line.set_xdata([optimal_val])
    optimal_text.set_text(f"Optimal: {optimal_val}")
    
    ax_error.set_xlabel(x_label)
    ax_error.set_xlim(min(x_vals)-1, max(x_vals)+1)
    
    all_errs = tr_errs + te_errs
    if all_errs:
        max_err = max(all_errs)
        ax_error.set_ylim(0, min(max_err + 0.1, 3.0))

    btn_toggle_res.label.set_text(f"Residuals: {'ON' if state['show_residuals'] else 'OFF'}")
    col = 'green' if "OPTIMAL" in status or "Converged" in status else 'red' if "OVERFITTING" in status else 'black'
    info_text.set_text(f"Degree: {degree} | Alpha: {alpha:.2f}\nData Pts: {len(X_tr)}\nStatus: {status}")
    info_text.set_color(col)
    
    fig.canvas.draw_idle()

# --- HANDLERS ---
def on_click_plot(event):
    if event.inaxes == ax_main:
        if event.button == 1: 
            state['manual_outliers']['x'].append(event.xdata)
            state['manual_outliers']['y'].append(event.ydata)
            update()

def on_param_change(val):
    initialize_data(n_samples=slider_size.val, test_size=slider_split.val, new_seed=False)
    update()

def on_new_seed(event):
    initialize_data(n_samples=slider_size.val, test_size=slider_split.val, new_seed=True)
    update()

def toggle_residuals(event):
    state['show_residuals'] = not state['show_residuals']
    update()

def change_dataset(label):
    state['dataset_type'] = label
    on_new_seed(None)

def change_graph_mode(label):
    state['graph_mode'] = label
    update()

def toggle_cv(label):
    state['use_cv'] = not state['use_cv']
    update()

def open_docs(event):
    """Opens the documentation HTML file in default browser."""
    try:
        path = os.path.realpath(DOC_FILENAME)
        webbrowser.open(f'file://{path}')
        print(f"Opening documentation: {path}")
    except Exception as e:
        print(f"Error opening documentation: {e}")

# --- 7. HOVER TOOLTIP ---
def on_hover(event):
    text = ""
    visible = False
    
    if event.inaxes == ax_main:
        text = "Left Click here to add an OUTLIER point!"
        visible = True
    elif event.inaxes == ax_sl_deg:
        text = "Degree (Complexity): Increases curve flexibility."
        visible = True
    elif event.inaxes == ax_sl_alpha:
        text = "Alpha (Regularization): Penalty term.\nIncrease this to STOP Overfitting even at high degrees."
        visible = True
    elif event.inaxes == ax_sl_noise:
        text = "Noise: Randomness in data."
        visible = True
    elif event.inaxes == ax_error:
        if state['graph_mode'] == 'Complexity':
            text = "Complexity Curve:\nFind the lowest point of the Red Line."
        else:
            text = "Learning Curve:\nSee if more data reduces the error gap."
        visible = True

    if text == state['last_tooltip_text'] and visible == tooltip.get_visible(): return
    state['last_tooltip_text'] = text
    
    if visible:
        tooltip.set_text(text)
        inv = fig.transFigure.inverted()
        pos = inv.transform((event.x + 20, event.y + 20))
        tooltip.set_position(pos)
        tooltip.set_visible(True)
        fig.canvas.draw_idle()
    else:
        if tooltip.get_visible():
            tooltip.set_visible(False)
            fig.canvas.draw_idle()

# --- 8. WIDGETS SETUP (Clean & Spaced Out) ---
bg_col = 'lightgray'

# PANEL 1: MODEL PARAMETERS (Blue)
# x=0.05 to 0.29 (Matches left graph edge)
# Moved Sliders to right (0.12) to allow labels to fit inside the panel
ax_sl_deg = plt.axes([0.12, 0.18, 0.14, 0.03], facecolor=bg_col)
slider_degree = Slider(ax_sl_deg, 'Degree', 1, 15, valinit=INIT_DEGREE, valstep=1)

ax_sl_alpha = plt.axes([0.12, 0.12, 0.14, 0.03], facecolor=bg_col)
slider_alpha = Slider(ax_sl_alpha, 'Alpha', 0.0, 5.0, valinit=INIT_ALPHA)

# PANEL 2: DATA GENERATION (Green)
# x=0.30 to 0.66
ax_radio_data = plt.axes([0.31, 0.05, 0.10, 0.18], facecolor='#e8f5e9')
radio_data = RadioButtons(ax_radio_data, ('Sine Wave', 'Linear', 'Step Function', 'Heteroscedastic'))
for label in radio_data.labels: label.set_fontsize(9)

# Moved sliders to right (0.52)
ax_sl_noise = plt.axes([0.52, 0.18, 0.12, 0.03], facecolor=bg_col)
slider_noise = Slider(ax_sl_noise, 'Noise', 0.0, 1.0, valinit=INIT_NOISE)

ax_sl_size = plt.axes([0.52, 0.13, 0.12, 0.03], facecolor=bg_col)
slider_size = Slider(ax_sl_size, 'Size', 10, 200, valinit=INIT_DATA_SIZE, valstep=5)

ax_btn_seed = plt.axes([0.52, 0.06, 0.12, 0.04])
btn_new_seed = Button(ax_btn_seed, 'New Random Seed')

# PANEL 3: ANALYSIS (Orange)
# x=0.67 to 0.98 (Matches right graph edge)
ax_radio_mode = plt.axes([0.68, 0.05, 0.10, 0.18], facecolor='#fff3e0')
radio_mode = RadioButtons(ax_radio_mode, ('Complexity', 'Learning Curve'))
for label in radio_mode.labels: label.set_fontsize(9)

# Moved sliders to right (0.85)
ax_sl_split = plt.axes([0.85, 0.19, 0.11, 0.03], facecolor=bg_col)
slider_split = Slider(ax_sl_split, 'Test %', 0.1, 0.8, valinit=INIT_TEST_RATIO)

ax_check_cv = plt.axes([0.85, 0.13, 0.11, 0.04], frameon=False, facecolor='#fff3e0')
check_cv = CheckButtons(ax_check_cv, ['Cross-Validation'], [INIT_USE_CV])

ax_btn_res = plt.axes([0.82, 0.06, 0.07, 0.04])
btn_toggle_res = Button(ax_btn_res, 'Residuals')

ax_btn_docs = plt.axes([0.90, 0.06, 0.06, 0.04])
btn_docs = Button(ax_btn_docs, 'Docs')

# --- 9. CONNECTIONS ---
slider_degree.on_changed(update)
slider_alpha.on_changed(update)
slider_noise.on_changed(update)
slider_size.on_changed(on_param_change)
slider_split.on_changed(on_param_change)
btn_new_seed.on_clicked(on_new_seed)
btn_toggle_res.on_clicked(toggle_residuals)
btn_docs.on_clicked(open_docs)
radio_data.on_clicked(change_dataset)
radio_mode.on_clicked(change_graph_mode)
check_cv.on_clicked(toggle_cv)

fig.canvas.mpl_connect("motion_notify_event", on_hover)
fig.canvas.mpl_connect("button_press_event", on_click_plot)

# Initial Plot Config
ax_main.set_title("Data Space & Model Fit")
ax_main.set_ylim(-2.5, 2.5)
ax_main.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax_main.grid(True, alpha=0.3)

ax_error.set_title("Error Analysis")
ax_error.set_ylabel("Error (RMSE)")
ax_error.legend(fontsize=8)
ax_error.grid(True, alpha=0.3)

update()
plt.show()