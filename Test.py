import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import Counter
 
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
df = pd.read_csv(r"")#-----------------your db
df = df[df['comuna'] != 'Sin identificar'].copy()
df['fecha'] = pd.to_datetime(df['fecha'])
df['semana'] = df['fecha'].dt.isocalendar().week.astype(int)
df['mes']    = df['fecha'].dt.month
 
df_robos = df[df['tipo_delito'].isin(ROBOS)].copy()
top10_comunas = df_robos['comuna'].value_counts().head(10).index.tolist()
 
# ─────────────────────────────────────────────
# ENTRENAMIENTO DEL MEJOR MODELO (Logistic Regression)
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
 
lr = LogisticRegression(max_iter=1000, class_weight='balanced', C=5)
lr.fit(X_train, y_train)
acc = accuracy_score(y_test_split, lr.predict(X_test_split))
print(f"Accuracy del modelo (LR): {acc:.4f}")
 
# ─────────────────────────────────────────────
# PREDICCIÓN POR COMUNA
# Para cada comuna tomamos sus noticias, las vectorizamos
# y pedimos al modelo que prediga el tipo de delito.
# Luego contamos qué tipo predijo más veces → perfil de la comuna.
# ─────────────────────────────────────────────
resumen_comunas = []
 
comunas_validas = df['comuna'].value_counts()
comunas_validas = comunas_validas[comunas_validas >= 5].index.tolist()
 
for comuna in comunas_validas:
    sub = df[df['comuna'] == comuna].copy()
    X_com = vectorizador.transform(sub['titulo'] + ' ' + sub['descripcion'])
    preds = lr.predict(X_com)
 
    total_noticias  = len(sub)
    total_robos_real = sub['tipo_delito'].isin(ROBOS).sum()
    tipo_mas_pred   = Counter(preds).most_common(1)[0][0]
    robo_mas_pred   = Counter([p for p in preds if p in ROBOS])
    tipo_robo_pred  = robo_mas_pred.most_common(1)[0][0] if robo_mas_pred else 'Sin robos'
 
    # Tendencia: robos en última semana vs semana anterior
    ultimas = df_robos[df_robos['comuna'] == comuna]['semana']
    sem_max = ultimas.max() if len(ultimas) > 0 else 0
    sem_ant = sem_max - 1
    robos_ultima  = (ultimas == sem_max).sum()
    robos_anterior = (ultimas == sem_ant).sum()
    tendencia = 'subiendo' if robos_ultima > robos_anterior else ('bajando' if robos_ultima < robos_anterior else 'estable')
 
    resumen_comunas.append({
        'comuna':           comuna,
        'total_noticias':   total_noticias,
        'robos_reales':     total_robos_real,
        'tipo_mas_pred':    tipo_mas_pred,
        'robo_mas_pred':    tipo_robo_pred,
        'tendencia':        tendencia,
        'robos_ultima_sem': robos_ultima,
    })
 
dfR = pd.DataFrame(resumen_comunas).sort_values('robos_reales', ascending=False)
 
# ─────────────────────────────────────────────
# RESULTADOS EN TERMINAL
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  ANÁLISIS DE ROBOS POR COMUNA — RM CHILE")
print("="*60)
 
print(f"\n🏆 Comuna con MÁS robos registrados: {dfR.iloc[0]['comuna']} ({dfR.iloc[0]['robos_reales']} eventos)")
print(f"🔮 Tipo de robo predominante predicho: {dfR.iloc[0]['robo_mas_pred']}")
 
print("\n📊 TOP 10 COMUNAS CON MÁS ROBOS:")
print(f"{'Comuna':<20} {'Robos reales':>12} {'Tipo predicho más frecuente':<25} {'Tendencia'}")
print("-"*75)
for _, row in dfR.head(10).iterrows():
    emoji = '📈' if row['tendencia'] == 'subiendo' else ('📉' if row['tendencia'] == 'bajando' else '➡️')
    print(f"{row['comuna']:<20} {row['robos_reales']:>12} {row['robo_mas_pred']:<25} {emoji} {row['tendencia']}")
 
print("\n🗺️  PERFIL DE ROBO POR COMUNA (tipo más probable según el modelo):")
print(f"{'Comuna':<20} {'Tipo de delito más predicho'}")
print("-"*45)
for _, row in dfR.head(15).iterrows():
    print(f"  {row['comuna']:<18} → {row['robo_mas_pred']}")
 
# Próxima semana: comunas con tendencia subiendo
en_alza = dfR[dfR['tendencia'] == 'subiendo'].head(5)
print(f"\n⚠️  COMUNAS CON ROBOS EN ALZA (última semana):")
for _, row in en_alza.iterrows():
    print(f"  → {row['comuna']} ({row['robos_ultima_sem']} robos esta semana)")
 
# ─────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Análisis de Robos por Comuna — Región Metropolitana", fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
 
# ── Gráfico 1: Top 10 comunas con más robos reales ──
ax1 = fig.add_subplot(gs[0, 0])
top10_data = dfR.head(10)
bars = ax1.barh(top10_data['comuna'][::-1], top10_data['robos_reales'][::-1], color='#e74c3c', edgecolor='white')
for bar, val in zip(bars, top10_data['robos_reales'][::-1]):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             str(val), va='center', fontsize=9)
ax1.set_title("Top 10 comunas con más robos", fontweight='bold')
ax1.set_xlabel("Cantidad de eventos")
ax1.set_xlim(0, top10_data['robos_reales'].max() * 1.15)
 
# ── Gráfico 2: Tipo de robo predicho por comuna (stacked) ──
ax2 = fig.add_subplot(gs[0, 1])
pivot = df_robos[df_robos['comuna'].isin(top10_comunas)]\
    .groupby(['comuna', 'tipo_delito']).size().unstack(fill_value=0)
pivot = pivot.reindex(df_robos[df_robos['comuna'].isin(top10_comunas)]['comuna'].value_counts().index)
colores = [COLORES_TIPO.get(c, '#95a5a6') for c in pivot.columns]
pivot.plot(kind='barh', stacked=True, ax=ax2, color=colores, edgecolor='white', linewidth=0.5)
ax2.set_title("Composición de robos por tipo\n(Top 10 comunas)", fontweight='bold')
ax2.set_xlabel("Cantidad")
ax2.legend(title="Tipo", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
ax2.set_ylabel("")
 
# ── Gráfico 3: Tendencia semanal de robos (top 5 comunas) ──
ax3 = fig.add_subplot(gs[1, 0])
top5 = dfR.head(5)['comuna'].tolist()
colores_lineas = ['#e74c3c','#3498db','#2ecc71','#e67e22','#9b59b6']
for i, comuna in enumerate(top5):
    serie = df_robos[df_robos['comuna'] == comuna].groupby('semana').size()
    todas_semanas = range(df_robos['semana'].min(), df_robos['semana'].max() + 1)
    serie = serie.reindex(todas_semanas, fill_value=0)
    ax3.plot(serie.index, serie.values, marker='o', label=comuna,
             color=colores_lineas[i], linewidth=2, markersize=4)
ax3.set_title("Tendencia semanal de robos\n(Top 5 comunas)", fontweight='bold')
ax3.set_xlabel("Semana del año")
ax3.set_ylabel("Cantidad de robos")
ax3.legend(fontsize=8)
ax3.set_xticks(list(todas_semanas))
 
# ── Gráfico 4: Predicción del modelo — tipo de robo por comuna ──
ax4 = fig.add_subplot(gs[1, 1])
pred_data = dfR[dfR['robo_mas_pred'] != 'Sin robos'].head(12)
tipo_counts = pred_data['robo_mas_pred'].value_counts()
colores_pie = [COLORES_TIPO.get(t, '#95a5a6') for t in tipo_counts.index]
wedges, texts, autotexts = ax4.pie(
    tipo_counts.values,
    labels=tipo_counts.index,
    autopct='%1.0f%%',
    colors=colores_pie,
    startangle=90,
    pctdistance=0.75
)
for text in autotexts:
    text.set_fontsize(9)
ax4.set_title("Tipo de robo predicho\ncomo predominante por comuna", fontweight='bold')
 
plt.savefig(r"", dpi=150, bbox_inches='tight')#-----------------your db
print("\n✅ Gráfico guardado analisis_comunas.png")
plt.show()