"""
Variteks Talep Tahmini — Karar Destek Sistemi
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# =====================================================================
# SAYFA AYARLARI
# =====================================================================
st.set_page_config(
    page_title="Variteks Talep Tahmini",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit'i light temaya zorla
st.markdown("""
<style>
    /* Light tema zorla */
    .stApp {
        background-color: #ffffff;
        color: #1a1a1a;
    }
    
    header[data-testid="stHeader"] {
        background: #ffffff;
    }
    
    section[data-testid="stSidebar"] > div {
        padding-bottom: 2rem;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# VERİ YÜKLEME
# =====================================================================
@st.cache_data
def load_data():
    monthly_df = pd.read_csv('results_v2/monthly_sales.csv', index_col=0)
    n1_size_df = pd.read_csv('results_v2/n1_size_monthly.csv', index_col=0)
    metrics_df = pd.read_csv('results_v2/model_comparison/metrics_table.csv')
    best_df = pd.read_csv('results_v2/model_comparison/best_models.csv')
    forecast_df = pd.read_csv('results_v2/forecasts/forecast_table.csv')
    return monthly_df, n1_size_df, metrics_df, best_df, forecast_df

monthly_df, n1_size_df, metrics_df, best_df, forecast_df = load_data()

all_series = {}
for code in ['X8', 'X2', 'X10', 'X16']:
    all_series[code] = monthly_df[code].astype(float)
for col in n1_size_df.columns:
    all_series[col] = n1_size_df[col].astype(float)
all_series['N1'] = monthly_df['N1'].astype(float)

SERIES_LABELS = {
    'X8': 'X8 (Nexus)',
    'X2': 'X2 (Nexus)',
    'X10': 'X10 (Nexus)',
    'X16': 'X16 (Nexus)',
    'N1': 'N1 Toplam (Nitgo)',
    'N1_M': 'N1 M Bedeni (Nitgo)',
    'N1_L': 'N1 L Bedeni (Nitgo)',
    'N1_XL': 'N1 XL Bedeni (Nitgo)',
}

MODEL_LABELS = {
    'Random_Forest': 'Random Forest',
    'XGBoost': 'XGBoost',
    'MA_3': 'Hareketli Ort. (3 Ay)',
    'MA_6': 'Hareketli Ort. (6 Ay)',
    'SARIMA': 'SARIMA',
    'Holt_Winters': 'Holt-Winters',
    'Naive': 'Naive',
}

# Plotly grafikleri icin ortak layout ayarlari
COMMON_LAYOUT = dict(
    template='plotly_white',
    font=dict(family='Arial', size=13, color='#333333'),
    paper_bgcolor='white',
    plot_bgcolor='white',
    margin=dict(t=40, b=60, l=60, r=30),
)

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("### Variteks Talep Tahmini")
    st.markdown("---")
    
    page = st.radio(
        "Sayfa",
        ["Ana Sayfa", "Ürün Analizi", "Tahmin Sonuçları", "Model Karşılaştırma", "Proje Hakkında"],
        index=0
    )
    
    st.markdown("---")
    st.caption("Veri: Ocak 2023 — Nisan 2026")
    st.caption("Tahmin: Mayıs — Temmuz 2026")

# =====================================================================
# SAYFA 1: ANA SAYFA
# =====================================================================
if page == "Ana Sayfa":
    st.title("Variteks Talep Tahmini Sistemi")
    st.markdown("Medikal kıyafet üretiminde ürün bazlı talep tahmini ve analiz platformu.")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Analiz Edilen Ürün", "5 Kod + 3 Beden")
    with col2:
        st.metric("Denenen Model", "7 Algoritma")
    with col3:
        total_forecast = int(forecast_df['Forecast'].sum())
        st.metric("3 Aylık Toplam Tahmin", f"{total_forecast:,} Adet")
    with col4:
        avg_wmape = best_df['MAPE'].mean()
        st.metric("Ort. WMAPE", f"%{avg_wmape:.1f}")
    
    st.markdown("---")
    
    st.subheader("Gelecek 3 Ay Tahmin Özeti")
    
    summary_data = []
    for _, row in best_df.iterrows():
        s = row['Series']
        f_data = forecast_df[forecast_df['Series'] == s]
        forecasts = f_data['Forecast'].values
        summary_data.append({
            'Ürün Kodu': s,
            'Model': MODEL_LABELS.get(row['Best_Model'], row['Best_Model']),
            'Mayıs': int(forecasts[0]),
            'Haziran': int(forecasts[1]),
            'Temmuz': int(forecasts[2]),
            'Toplam': int(sum(forecasts)),
            'WMAPE': f"%{row['MAPE']:.1f}",
        })
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.subheader("Aylık Satış Trendleri")
    
    fig = go.Figure()
    colors = {'X8': '#d62728', 'X2': '#1f77b4', 'X10': '#2ca02c', 'X16': '#9467bd', 'N1': '#ff7f0e'}
    
    for code in ['X8', 'X2', 'X10', 'X16', 'N1']:
        series = all_series[code]
        fig.add_trace(go.Scatter(
            x=list(series.index), y=series.values,
            name=code, mode='lines+markers',
            line=dict(color=colors[code], width=2),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        **COMMON_LAYOUT,
        height=420,
        legend=dict(
            orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5,
            font=dict(size=13)
        ),
        xaxis=dict(title='Ay', tickangle=-45, dtick=3),
        yaxis_title='Satış (Adet)',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# SAYFA 2: ÜRÜN ANALİZİ
# =====================================================================
elif page == "Ürün Analizi":
    st.title("Ürün Bazlı Analiz")
    st.markdown("---")
    
    selected = st.selectbox("Ürün seçin:", list(SERIES_LABELS.keys()), format_func=lambda x: SERIES_LABELS[x])
    
    series = all_series[selected]
    best_row = best_df[best_df['Series'] == selected].iloc[0]
    f_data = forecast_df[forecast_df['Series'] == selected]
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Toplam Satış", f"{int(series.sum()):,}")
    with col2:
        st.metric("Aylık Ortalama", f"{int(series.mean()):,}")
    with col3:
        cv = (series.std() / series.mean() * 100) if series.mean() > 0 else 0
        st.metric("CV (%)", f"{cv:.0f}")
    with col4:
        active = int((series > 0).sum())
        st.metric("Aktif Ay", f"{active}/40")
    
    st.markdown("---")
    
    st.subheader(f"{selected} — Satış Geçmişi ve Tahmin")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(series.index), y=series.values,
        name='Gerçekleşen', mode='lines+markers',
        line=dict(color='#1f77b4', width=2.5),
        marker=dict(size=4)
    ))
    
    f_months = list(f_data['Period'])
    f_vals = list(f_data['Forecast'])
    f_lower = list(f_data['Lower_95'])
    f_upper = list(f_data['Upper_95'])
    
    fig.add_trace(go.Scatter(
        x=f_months, y=f_vals,
        name='Tahmin', mode='lines+markers',
        line=dict(color='#d62728', width=2.5, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    fig.add_trace(go.Scatter(
        x=f_months + f_months[::-1],
        y=f_upper + f_lower[::-1],
        fill='toself', fillcolor='rgba(214,39,40,0.12)',
        line=dict(color='rgba(214,39,40,0)'),
        name='%95 Güven Aralığı'
    ))
    
    fig.update_layout(
        **COMMON_LAYOUT,
        height=450,
        xaxis=dict(title='Ay', tickangle=-45, dtick=3),
        yaxis_title='Satış (Adet)',
        legend=dict(
            orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5,
            font=dict(size=12)
        ),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Tahmin Detayları")
    detail_df = f_data[['Month', 'Best_Model', 'Forecast', 'Lower_95', 'Upper_95']].copy()
    detail_df.columns = ['Ay', 'Model', 'Tahmin', 'Alt Sınır (%95)', 'Üst Sınır (%95)']
    detail_df['Model'] = detail_df['Model'].map(lambda x: MODEL_LABELS.get(x, x))
    st.dataframe(detail_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.subheader("Mevsimsel Dağılım")
    months_data = pd.DataFrame({'Satış': series.values})
    months_data['Ay_No'] = [int(str(p).split('-')[1]) for p in series.index]
    
    ay_labels = {1:'Oca', 2:'Şub', 3:'Mar', 4:'Nis', 5:'May', 6:'Haz',
                 7:'Tem', 8:'Ağu', 9:'Eyl', 10:'Eki', 11:'Kas', 12:'Ara'}
    months_data['Ay'] = months_data['Ay_No'].map(ay_labels)
    
    fig2 = px.box(months_data, x='Ay', y='Satış', 
                  category_orders={'Ay': list(ay_labels.values())})
    fig2.update_layout(**COMMON_LAYOUT, height=380, xaxis_title='Ay', yaxis_title='Satış (Adet)')
    st.plotly_chart(fig2, use_container_width=True)

# =====================================================================
# SAYFA 3: TAHMİN SONUÇLARI
# =====================================================================
elif page == "Tahmin Sonuçları":
    st.title("Mayıs — Temmuz 2026 Tahminleri")
    st.markdown("---")
    
    st.subheader("Ürün Bazlı Tahmin Karşılaştırması")
    
    pivot = forecast_df.pivot(index='Series', columns='Month', values='Forecast')
    pivot = pivot[['Mayis 2026', 'Haziran 2026', 'Temmuz 2026']]
    
    # Sirayla goster (N1 en sonda)
    order = ['X2', 'X8', 'X10', 'X16', 'N1_M', 'N1_L', 'N1_XL', 'N1']
    pivot = pivot.reindex([o for o in order if o in pivot.index])
    
    fig = go.Figure()
    month_colors = {'Mayis 2026': '#1f77b4', 'Haziran 2026': '#ff7f0e', 'Temmuz 2026': '#2ca02c'}
    month_short = {'Mayis 2026': 'Mayıs', 'Haziran 2026': 'Haziran', 'Temmuz 2026': 'Temmuz'}
    
    for month in ['Mayis 2026', 'Haziran 2026', 'Temmuz 2026']:
        fig.add_trace(go.Bar(
            name=month_short[month],
            x=pivot.index, y=pivot[month],
            marker_color=month_colors[month],
            text=pivot[month].astype(int),
            textposition='outside',
            textfont=dict(size=10),
            constraintext='none'
        ))
    
    fig.update_layout(
        **COMMON_LAYOUT,
        barmode='group',
        bargap=0.25,
        bargroupgap=0.08,
        height=520,
        xaxis_title='Ürün Kodu',
        yaxis_title='Tahmin (Adet)',
        # Y eksenini biraz yukselt ki textler kesilmesin
        yaxis=dict(range=[0, pivot.max().max() * 1.25]),
        legend=dict(
            orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5,
            font=dict(size=14), itemsizing='constant',
            traceorder='normal'
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Detaylı Tahmin Tablosu")
    display_df = forecast_df.copy()
    display_df.columns = ['Ürün', 'Ay', 'Dönem', 'Model', 'Tahmin', 'Alt Sınır', 'Üst Sınır']
    display_df['Model'] = display_df['Model'].map(lambda x: MODEL_LABELS.get(x, x))
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.subheader("N1 — Beden Bazlı Tahmin")
    
    n1_sizes = forecast_df[forecast_df['Series'].isin(['N1_M', 'N1_L', 'N1_XL'])]
    
    fig2 = go.Figure()
    size_colors = {'N1_M': '#ff7f0e', 'N1_L': '#d62728', 'N1_XL': '#9467bd'}
    size_labels_map = {'N1_M': 'M Bedeni', 'N1_L': 'L Bedeni', 'N1_XL': 'XL Bedeni'}
    
    for size_code in ['N1_M', 'N1_L', 'N1_XL']:
        s_data = n1_sizes[n1_sizes['Series'] == size_code]
        # Ay isimlerini kisalt
        short_months = s_data['Month'].str.replace('Mayis', 'May').str.replace('Haziran', 'Haz').str.replace('Temmuz', 'Tem')
        fig2.add_trace(go.Bar(
            name=size_labels_map[size_code],
            x=short_months, y=s_data['Forecast'],
            marker_color=size_colors[size_code],
            text=s_data['Forecast'].astype(int),
            textposition='outside',
            textfont=dict(size=11)
        ))
    
    fig2.update_layout(
        **COMMON_LAYOUT,
        barmode='group',
        bargap=0.3,
        height=420,
        xaxis_title='Ay', yaxis_title='Tahmin (Adet)',
        yaxis=dict(range=[0, n1_sizes['Forecast'].max() * 1.3]),
        legend=dict(
            orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5,
            font=dict(size=13)
        ),
    )
    st.plotly_chart(fig2, use_container_width=True)

# =====================================================================
# SAYFA 4: MODEL KARŞILAŞTIRMA
# =====================================================================
elif page == "Model Karşılaştırma":
    st.title("Model Performans Karşılaştırması")
    st.markdown("---")
    
    selected = st.selectbox("Ürün seçin:", list(SERIES_LABELS.keys()), format_func=lambda x: SERIES_LABELS[x])
    
    s_metrics = metrics_df[metrics_df['Series'] == selected].copy()
    best_model = best_df[best_df['Series'] == selected].iloc[0]['Best_Model']
    
    st.markdown("---")
    
    st.subheader(f"{selected} — MAE Karşılaştırması")
    st.caption("Yeşil çubuk en iyi modeli gösterir.")
    
    s_metrics_sorted = s_metrics.sort_values('MAE')
    colors_bar = ['#2ca02c' if m == best_model else '#1f77b4' for m in s_metrics_sorted['Model']]
    
    fig = go.Figure(go.Bar(
        x=s_metrics_sorted['Model'].map(lambda x: MODEL_LABELS.get(x, x)),
        y=s_metrics_sorted['MAE'],
        marker_color=colors_bar,
        text=s_metrics_sorted['MAE'].round(0).astype(int),
        textposition='outside',
        textfont=dict(size=11)
    ))
    
    fig.update_layout(
        **COMMON_LAYOUT,
        height=440,
        xaxis_title='Model', yaxis_title='MAE (Adet)',
        yaxis=dict(range=[0, s_metrics_sorted['MAE'].max() * 1.2]),
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Tüm Metrikler")
    
    display_metrics = s_metrics.copy()
    display_metrics['Model'] = display_metrics['Model'].map(lambda x: MODEL_LABELS.get(x, x))
    display_metrics.columns = ['Ürün', 'Model', 'MAE', 'RMSE', 'WMAPE (%)']
    display_metrics['WMAPE (%)'] = display_metrics['WMAPE (%)'].apply(lambda x: f"%{x:.1f}")
    display_metrics['MAE'] = display_metrics['MAE'].round(0).astype(int)
    display_metrics['RMSE'] = display_metrics['RMSE'].round(0).astype(int)
    display_metrics = display_metrics.sort_values('MAE')
    
    st.dataframe(display_metrics[['Model', 'MAE', 'RMSE', 'WMAPE (%)']], 
                 use_container_width=True, hide_index=True)

# =====================================================================
# SAYFA 5: PROJE HAKKINDA
# =====================================================================
elif page == "Proje Hakkında":
    st.title("Proje Hakkında")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Firma Bilgileri")
        st.markdown("""
        - **Firma:** Variteks (Medikal Kıyafet Üreticisi)
        - **Üretim Tipi:** Siparişe Dayalı (Make-to-Order)
        - **Markalar:** Nexus, Nitgo, Judo
        - **Veri Aralığı:** Ocak 2023 — Nisan 2026 (40 Ay)
        - **Veri Türü:** Fatura bazlı satış verileri
        """)
        
        st.subheader("Amaç")
        st.markdown("""
        Geçmiş satış verilerini kullanarak ürün kodu bazında 
        Mayıs, Haziran ve Temmuz 2026 için talep tahmini üretmek.
        """)
    
    with col2:
        st.subheader("Kullanılan Yöntemler")
        st.markdown("""
        **Baseline Modeller:**
        - Naive Forecast
        - Hareketli Ortalama (MA-3, MA-6)
        
        **Zaman Serisi Modelleri:**
        - SARIMA
        - Holt-Winters (ETS)
        
        **Makine Öğrenmesi:**
        - Random Forest
        - XGBoost
        """)
        
        st.subheader("Veri Ön İşleme")
        st.markdown("""
        - Outlier temizliği: IQR yöntemi (Winsorization)
        - Hata metriği: WMAPE
        - Feature Engineering: Lag, Rolling, Trend
        """)
    
    st.markdown("---")
    st.subheader("Metrik Açıklamaları")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**MAE** — Ortalama Mutlak Hata (adet). Düşük = İyi.")
    with col2:
        st.info("**RMSE** — Karekök Ortalama Kare Hata. Büyük sapmalara duyarlı.")
    with col3:
        st.info("**WMAPE** — Ağırlıklı Yüzdesel Hata. Küçük satış aylarında patlamayı engeller.")
