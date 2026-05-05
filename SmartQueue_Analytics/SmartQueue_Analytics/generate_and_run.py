import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─── GENERATE DATASET ───────────────────────────────────────────────────────
n = 1000
arrival_hours = np.random.choice(
    range(8, 20),
    size=n,
    p=[0.03, 0.06, 0.10, 0.12, 0.10, 0.08, 0.06, 0.10, 0.12, 0.10, 0.08, 0.05]
)
arrival_minutes = np.random.randint(0, 60, n)
days = np.random.choice(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'], n,
                         p=[0.18,0.15,0.17,0.16,0.18,0.16])

# queue length depends on hour
queue_length = np.clip(
    (arrival_hours - 8) * 1.5 + np.random.randint(1, 15, n) +
    np.where(np.isin(days, ['Monday','Friday']), 5, 0), 1, 40
).astype(int)

service_time = np.random.randint(3, 20, n)

# waiting time = queue * avg_service + noise
waiting_time = (queue_length * 2.5 + service_time * 0.8 +
                np.random.normal(0, 5, n)).clip(1)
waiting_time = waiting_time.round(2)

df = pd.DataFrame({
    'token_id': range(1001, 1001+n),
    'arrival_hour': arrival_hours,
    'arrival_minute': arrival_minutes,
    'day_of_week': days,
    'queue_length': queue_length,
    'service_time': service_time,
    'waiting_time': waiting_time
})
df.to_csv('data/smartqueue_dataset.csv', index=False)
print("✅ Dataset saved")
print(df.head())
print(df.describe())

# ─── FEATURE ENGINEERING ────────────────────────────────────────────────────
le = LabelEncoder()
df['day_encoded'] = le.fit_transform(df['day_of_week'])
df['is_peak'] = df['arrival_hour'].apply(lambda x: 1 if x in [10,11,12,16,17,18] else 0)

features = ['arrival_hour', 'arrival_minute', 'queue_length', 'service_time', 'day_encoded', 'is_peak']
X = df[features]
y = df['waiting_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─── MODEL ──────────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"\n📊 Model Results:")
print(f"   MSE  : {mse:.2f}")
print(f"   RMSE : {rmse:.2f} minutes")
print(f"   R²   : {r2:.4f}")

# ─── VISUALIZATIONS ─────────────────────────────────────────────────────────
plt.style.use('dark_background')
palette = ['#00D4FF','#FF6B6B','#FFD93D','#6BCB77','#FF922B','#CC5DE8']

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#0D1117')
fig.suptitle('🚦 SmartQueue Analytics Dashboard', fontsize=22, fontweight='bold',
             color='white', y=1.01)

# 1. Avg wait time by hour
avg_wait = df.groupby('arrival_hour')['waiting_time'].mean()
ax = axes[0, 0]
bars = ax.bar(avg_wait.index, avg_wait.values, color=palette[0], alpha=0.85, edgecolor='#00D4FF', linewidth=0.5)
ax.set_facecolor('#161B22')
ax.set_title('⏱ Average Wait Time by Hour', color='white', fontsize=13, pad=10)
ax.set_xlabel('Hour of Day', color='#8B949E')
ax.set_ylabel('Avg Wait Time (min)', color='#8B949E')
ax.tick_params(colors='#8B949E')
for spine in ax.spines.values(): spine.set_color('#30363D')
# highlight peaks
for bar, h in zip(bars, avg_wait.index):
    if h in [10,11,12,16,17,18]:
        bar.set_color('#FF6B6B')
peak_patch = mpatches.Patch(color='#FF6B6B', label='Peak Hours')
normal_patch = mpatches.Patch(color=palette[0], label='Normal Hours')
ax.legend(handles=[peak_patch, normal_patch], facecolor='#21262D', edgecolor='#30363D', labelcolor='white', fontsize=9)

# 2. Queue length distribution
ax = axes[0, 1]
ax.hist(df['queue_length'], bins=20, color=palette[1], edgecolor='#FF6B6B', alpha=0.8, linewidth=0.5)
ax.set_facecolor('#161B22')
ax.set_title('📊 Queue Length Distribution', color='white', fontsize=13, pad=10)
ax.set_xlabel('Queue Length', color='#8B949E')
ax.set_ylabel('Frequency', color='#8B949E')
ax.tick_params(colors='#8B949E')
for spine in ax.spines.values(): spine.set_color('#30363D')

# 3. Peak vs Non-peak
ax = axes[0, 2]
peak_avg = df.groupby('is_peak')['waiting_time'].mean()
labels = ['Non-Peak', 'Peak Hours']
colors = [palette[3], palette[1]]
wedges, texts, autotexts = ax.pie(peak_avg.values, labels=labels, colors=colors,
                                   autopct='%1.1f%%', startangle=90,
                                   textprops={'color': 'white', 'fontsize': 11})
for at in autotexts: at.set_fontweight('bold')
ax.set_facecolor('#161B22')
ax.set_title('🕐 Peak vs Non-Peak Wait Time', color='white', fontsize=13, pad=10)

# 4. Actual vs Predicted
ax = axes[1, 0]
ax.scatter(y_test, y_pred, alpha=0.5, color=palette[4], s=20, edgecolors='none')
mn, mx = y_test.min(), y_test.max()
ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect Prediction')
ax.set_facecolor('#161B22')
ax.set_title('🎯 Actual vs Predicted Wait Time', color='white', fontsize=13, pad=10)
ax.set_xlabel('Actual (min)', color='#8B949E')
ax.set_ylabel('Predicted (min)', color='#8B949E')
ax.tick_params(colors='#8B949E')
for spine in ax.spines.values(): spine.set_color('#30363D')
ax.legend(facecolor='#21262D', edgecolor='#30363D', labelcolor='white', fontsize=9)
ax.text(0.05, 0.92, f'R² = {r2:.3f}', transform=ax.transAxes, color='#FFD93D',
        fontsize=11, fontweight='bold')

# 5. Avg wait by day
ax = axes[1, 1]
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
day_wait = df.groupby('day_of_week')['waiting_time'].mean().reindex(day_order)
bars2 = ax.bar(range(len(day_order)), day_wait.values, color=palette[5], alpha=0.85,
               edgecolor='#CC5DE8', linewidth=0.5)
ax.set_facecolor('#161B22')
ax.set_title('📅 Avg Wait Time by Day', color='white', fontsize=13, pad=10)
ax.set_xticks(range(len(day_order)))
ax.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat'], color='#8B949E')
ax.set_ylabel('Avg Wait Time (min)', color='#8B949E')
ax.tick_params(colors='#8B949E')
for spine in ax.spines.values(): spine.set_color('#30363D')

# 6. Queue length vs waiting time
ax = axes[1, 2]
sc = ax.scatter(df['queue_length'], df['waiting_time'], c=df['arrival_hour'],
                cmap='plasma', alpha=0.4, s=15, edgecolors='none')
cbar = plt.colorbar(sc, ax=ax)
cbar.ax.tick_params(colors='#8B949E', labelsize=8)
cbar.set_label('Hour of Day', color='#8B949E', fontsize=9)
ax.set_facecolor('#161B22')
ax.set_title('🔗 Queue Length vs Wait Time', color='white', fontsize=13, pad=10)
ax.set_xlabel('Queue Length', color='#8B949E')
ax.set_ylabel('Waiting Time (min)', color='#8B949E')
ax.tick_params(colors='#8B949E')
for spine in ax.spines.values(): spine.set_color('#30363D')

plt.tight_layout(pad=2.5)
plt.savefig('outputs/smartqueue_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.close()
print("✅ Dashboard saved")

# ─── METRICS SUMMARY ────────────────────────────────────────────────────────
coeff = pd.Series(model.coef_, index=features)
summary = {
    'MSE': round(mse,2), 'RMSE': round(rmse,2), 'R2': round(r2,4),
    'Peak_hour_avg_wait_min': round(df[df['is_peak']==1]['waiting_time'].mean(),2),
    'Normal_hour_avg_wait_min': round(df[df['is_peak']==0]['waiting_time'].mean(),2),
    'Busiest_hour': int(avg_wait.idxmax()),
    'Top_coefficient_feature': coeff.abs().idxmax()
}
pd.Series(summary).to_csv('outputs/model_metrics.csv', header=False)
print("✅ Metrics saved:", summary)
