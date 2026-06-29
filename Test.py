import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import lightgbm as lgb
from DataCleaning import dfClean

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
ROBOS = ['Robo Violento', 'Robo General', 'Robo Vehiculo', 'Encerrona', 'Portonazo']
COLORES_TIPO = {
    'Robo Violento': '#e74c3c',
    'Encerrona':     '#e67e22',
    'Robo Vehiculo': '#3498db',
    'Robo General':  '#9b59b6',
    'Portonazo':     '#1abc9c',
}

# ─────────────────────────────────────────────
# CARGA Y PREPARACIÓN
# ─────────────────────────────────────────────
df = dfClean.rename(columns={
    'title':            'titulo',
    'summary':          'descripcion',
    'district':         'comuna',
    'crime_type':       'tipo_delito',
    'publication_date': 'fecha'
})

df = df[df['comuna'] != 'Sin identificar'].copy()
df['fecha'] = pd.to_datetime(df['fecha'])
df['semana'] = df['fecha'].dt.isocalendar().week.astype(int)
df['mes']    = df['fecha'].dt.month

df_robos = df[df['tipo_delito'].isin(ROBOS)].copy()
top10_comunas = df_robos['comuna'].value_counts().head(10).index.tolist()

# ─────────────────────────────────────────────
# ENTRENAMIENTO — LightGBM
# ─────────────────────────────────────────────
X_text = df['titulo'] + ' ' + df['descripcion']
y      = df['tipo_delito']

vectorizador = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
X_tfidf = vectorizador.fit_transform(X_text)

X_train, X_test_split, y_train, y_test_split = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

print("Entrenando LightGBM...")
lgbm = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
lgbm.fit(X_train, y_train)
acc = accuracy_score(y_test_split, lgbm.predict(X_test_split))
print(f"Accuracy LightGBM: {acc:.4f}")

# ─────────────────────────────────────────────
# PREDICCIÓN + PROBABILIDADES POR COMUNA
# ─────────────────────────────────────────────
clases = lgbm.classes_  # orden de clases que conoce el modelo
resumen_comunas = []

comunas_validas = df['comuna'].value_counts()
comunas_validas = comunas_validas[comunas_validas >= 5].index.tolist()

for comuna in comunas_validas:
    sub = df[df['comuna'] == comuna].copy()
    X_com = vectorizador.transform(sub['titulo'] + ' ' + sub['descripcion'])

    # predict_proba → matriz (n_noticias × n_clases), promediamos por comuna
    proba_matrix  = lgbm.predict_proba(X_com)
    proba_promedio = proba_matrix.mean(axis=0)  # probabilidad media por clase

    # Diccionario clase → probabilidad promedio
    proba_dict = {clase: round(float(prob), 3) for clase, prob in zip(clases, proba_promedio)}

    # Solo robos
    proba_robos = {t: proba_dict.get(t, 0.0) for t in ROBOS}
    total_prob_robos = sum(proba_robos.values())

    # Normalizar para que sumen 100% entre los tipos de robo
    if total_prob_robos > 0:
        proba_robos_norm = {t: round(v / total_prob_robos, 3) for t, v in proba_robos.items()}
    else:
        proba_robos_norm = {t: 0.0 for t in ROBOS}

    tipo_robo_pred = max(proba_robos_norm, key=proba_robos_norm.get)

    total_robos_real = sub['tipo_delito'].isin(ROBOS).sum()

    # Tendencia
    ultimas = df_robos[df_robos['comuna'] == comuna]['semana']
    sem_max = ultimas.max() if len(ultimas) > 0 else 0
    robos_ultima   = (ultimas == sem_max).sum()
    robos_anterior = (ultimas == sem_max - 1).sum()
    tendencia = 'subiendo' if robos_ultima > robos_anterior else (
                'bajando'  if robos_ultima < robos_anterior else 'estable')

    resumen_comunas.append({
        'comuna':           comuna,
        'total_noticias':   len(sub),
        'robos_reales':     total_robos_real,
        'robo_mas_pred':    tipo_robo_pred,
        'tendencia':        tendencia,
        'robos_ultima_sem': robos_ultima,
        'probabilidades':   proba_robos_norm,
    })

dfR = pd.DataFrame(resumen_comunas).sort_values('robos_reales', ascending=False)

# ─────────────────────────────────────────────
# TERMINAL — RESULTADOS
# ─────────────────────────────────────────────
print("\n" + "="*65)
print("  ANÁLISIS DE ROBOS POR COMUNA — RM CHILE  (LightGBM)")
print("="*65)

print(f"\n🏆 Comuna con MÁS robos: {dfR.iloc[0]['comuna']} ({dfR.iloc[0]['robos_reales']} eventos)")
print(f"🔮 Tipo predominante predicho: {dfR.iloc[0]['robo_mas_pred']}")

print("\n📊 TOP 10 COMUNAS CON MÁS ROBOS:")
print(f"{'Comuna':<20} {'Robos':>6} {'Tipo predicho':<18} {'Tendencia'}")
print("-"*65)
for _, row in dfR.head(10).iterrows():
    emoji = '📈' if row['tendencia'] == 'subiendo' else ('📉' if row['tendencia'] == 'bajando' else '➡️')
    print(f"{row['comuna']:<20} {row['robos_reales']:>6} {row['robo_mas_pred']:<18} {emoji} {row['tendencia']}")

# ── Probabilidades por comuna ──
print("\n🎯 PROBABILIDAD DE TIPO DE ROBO POR COMUNA:")
print("-"*65)
for _, row in dfR.head(20).iterrows():
    probs = row['probabilidades']
    probs_sorted = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    # Solo muestra los que tienen probabilidad > 5%
    probs_str = ' | '.join([f"{t}: {p:.0%}" for t, p in probs_sorted if p > 0.05])
    print(f"  {row['comuna']:<18} → {probs_str}")

en_alza = dfR[dfR['tendencia'] == 'subiendo'].head(5)
print(f"\n⚠️  COMUNAS EN ALZA:")
for _, row in en_alza.iterrows():
    print(f"  → {row['comuna']} ({row['robos_ultima_sem']} robos esta semana)")

# ─────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.suptitle("Análisis de Robos por Comuna — Región Metropolitana (LightGBM)",
             fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Gráfico 1: Top 10 comunas ──
ax1 = fig.add_subplot(gs[0, 0])
top10_data = dfR.head(10)
bars = ax1.barh(top10_data['comuna'][::-1], top10_data['robos_reales'][::-1],
                color='#e74c3c', edgecolor='white')
for bar, val in zip(bars, top10_data['robos_reales'][::-1]):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             str(val), va='center', fontsize=9)
ax1.set_title("Top 10 comunas con más robos", fontweight='bold')
ax1.set_xlabel("Cantidad de eventos")
ax1.set_xlim(0, top10_data['robos_reales'].max() * 1.15)

# ── Gráfico 2: Composición real por tipo ──
ax2 = fig.add_subplot(gs[0, 1])
pivot = df_robos[df_robos['comuna'].isin(top10_comunas)]\
    .groupby(['comuna', 'tipo_delito']).size().unstack(fill_value=0)
pivot = pivot.reindex(
    df_robos[df_robos['comuna'].isin(top10_comunas)]['comuna'].value_counts().index
)
colores = [COLORES_TIPO.get(c, '#95a5a6') for c in pivot.columns]
pivot.plot(kind='barh', stacked=True, ax=ax2, color=colores, edgecolor='white', linewidth=0.5)
ax2.set_title("Composición de robos por tipo\n(Top 10 comunas)", fontweight='bold')
ax2.set_xlabel("Cantidad")
ax2.legend(title="Tipo", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
ax2.set_ylabel("")

# ── Gráfico 3: Tendencia semanal ──
ax3 = fig.add_subplot(gs[1, 0])
top5 = dfR.head(5)['comuna'].tolist()
colores_lineas = ['#e74c3c','#3498db','#2ecc71','#e67e22','#9b59b6']
todas_semanas = range(df_robos['semana'].min(), df_robos['semana'].max() + 1)
for i, comuna in enumerate(top5):
    serie = df_robos[df_robos['comuna'] == comuna].groupby('semana').size()
    serie = serie.reindex(todas_semanas, fill_value=0)
    ax3.plot(serie.index, serie.values, marker='o', label=comuna,
             color=colores_lineas[i], linewidth=2, markersize=4)
ax3.set_title("Tendencia semanal de robos\n(Top 5 comunas)", fontweight='bold')
ax3.set_xlabel("Semana del año")
ax3.set_ylabel("Cantidad de robos")
ax3.legend(fontsize=8)
ax3.set_xticks(list(todas_semanas))

# ── Gráfico 4: Heatmap de probabilidades por comuna ──
ax4 = fig.add_subplot(gs[1, 1])
top15 = dfR.head(15)
heatmap_data = pd.DataFrame(
    [row['probabilidades'] for _, row in top15.iterrows()],
    index=top15['comuna'].values
)[ROBOS]
sns.heatmap(
    heatmap_data,
    ax=ax4,
    cmap='Reds',
    annot=True,
    fmt='.0%',
    linewidths=0.5,
    cbar_kws={'label': 'Probabilidad'}
)
ax4.set_title("Probabilidad de tipo de robo\npor comuna (modelo LightGBM)", fontweight='bold')
ax4.set_xlabel("")
ax4.tick_params(axis='x', rotation=30)

plt.savefig("analisis_comunas.png", dpi=150, bbox_inches='tight')
print("\n✅ Gráfico guardado: analisis_comunas.png")
plt.show()