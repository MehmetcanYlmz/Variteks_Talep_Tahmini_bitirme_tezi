import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# 1. TEMEL SAYFA AYARI
st.set_page_config(
    page_title="Variteks Talep Tahmini",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stMetricValue"] > div {
    white-space: normal !important;
    word-break: break-word;
    font-size: 1.8rem !important;
    line-height: 1.2 !important;
}
</style>
""", unsafe_allow_html=True)

# 2. VERİ YÜKLEME
@st.cache_data
def load_data():
    monthly = pd.read_csv('results_v2/monthly_sales.csv', index_col=0)
    n1_size = pd.read_csv('results_v2/n1_size_monthly.csv', index_col=0)
    metrics = pd.read_csv('results_v2/model_comparison/metrics_table.csv')
    best = pd.read_csv('results_v2/model_comparison/best_models.csv')
    forecast = pd.read_csv('results_v2/forecasts/forecast_table.csv')
    return monthly, n1_size, metrics, best, forecast

monthly_df, n1_size_df, metrics_df, best_df, forecast_df = load_data()

# 3. YARDIMCI SÖZLÜKLER
SERIES_LABELS = {
    'X8': 'X8 (Nexus)', 'X2': 'X2 (Nexus)', 'X10': 'X10 (Nexus)', 'X16': 'X16 (Nexus)',
    'N1': 'N1 Toplam (Nitgo)', 'N1_M': 'N1 M Bedeni', 'N1_L': 'N1 L Bedeni', 'N1_XL': 'N1 XL Bedeni'
}
MODEL_LABELS = {
    'Random_Forest': 'Random Forest', 'XGBoost': 'XGBoost', 'Linear_Regression': 'Doğrusal Regresyon',
    'MA_3': 'Hareketli Ort. (3 Ay)', 'MA_6': 'Hareketli Ort. (6 Ay)',
    'SARIMA': 'SARIMA', 'Holt_Winters': 'Holt-Winters', 'Naive': 'Naive'
}

# 4. YAN MENÜ (SIDEBAR)
with st.sidebar:
    st.header("Variteks Talep Tahmini")
    st.markdown("---")
    menu = st.radio("Menü", ["Ana Sayfa", "Ürün Analizi", "Tahmin Sonuçları", "Model Karşılaştırma", "Proje Hakkında"])
    st.markdown("---")
    st.caption("Veri: Ocak 2023 - Nisan 2026")
    st.caption("Tahmin: Mayıs - Temmuz 2026")

# 5. SAYFALAR
if menu == "Ana Sayfa":
    st.title("Variteks Talep Tahmini Karar Destek Sistemi")
    st.markdown("---")

    # Özet Metrikler
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Ürün Serisi", "5 Kod + 3 Beden")
    c2.metric("Test Edilen Algoritmalar", "8 Farklı Model")
    c3.metric("3 Aylık Toplam Tahmin", f"{int(forecast_df['Forecast'].sum()):,} Adet")
    c4.metric("Ortalama Hata (WMAPE)", f"%{best_df['MAPE'].mean():.1f}")
    
    st.markdown("---")
    st.subheader("Ürün Bazında Özet Tahmin (Önümüzdeki 3 Ay)")
    
    # Özet Tablo Oluşturma
    summary = []
    for _, row in best_df.iterrows():
        s = row['Series']
        f_data = forecast_df[forecast_df['Series'] == s]['Forecast'].values
        summary.append({
            'Ürün': SERIES_LABELS.get(s, s),
            'Seçilen Model': MODEL_LABELS.get(row['Best_Model'], row['Best_Model']),
            'Mayıs': int(f_data[0]),
            'Haziran': int(f_data[1]),
            'Temmuz': int(f_data[2]),
            'Toplam': int(f_data.sum()),
            'Hata Oranı': f"%{row['MAPE']:.1f}"
        })
    st.table(pd.DataFrame(summary))
    
    st.markdown("---")
    st.subheader("Geçmiş Satış Trendleri (Genel)")
    fig = px.line(monthly_df, x=monthly_df.index, y=['X8', 'X2', 'X10', 'X16', 'N1'],
                  labels={'value': 'Satış (Adet)', 'index': 'Ay', 'variable': 'Ürün'})
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
    st.plotly_chart(fig)

elif menu == "Ürün Analizi":
    st.title("Detaylı Ürün Analizi")
    st.markdown("---")
    
    selected_code = st.selectbox("İncelemek istediğiniz ürünü seçin:", list(SERIES_LABELS.keys()), format_func=lambda x: SERIES_LABELS[x])
    
    if selected_code in monthly_df.columns:
        sales = monthly_df[selected_code]
    else:
        sales = n1_size_df[selected_code]
        
    f_data = forecast_df[forecast_df['Series'] == selected_code]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam Satış (40 Ay)", f"{int(sales.sum()):,}")
    c2.metric("Aylık Ortalama", f"{int(sales.mean()):,}")
    c3.metric("Satış Yapılan Ay", f"{int((sales > 0).sum())} / 40")
    
    st.markdown("---")
    st.subheader(f"{selected_code} Satış ve Tahmin Grafiği")
    
    fig = go.Figure()
    # Geçmiş Veri
    fig.add_trace(go.Scatter(x=list(sales.index), y=sales.values, name='Gerçekleşen', line=dict(color='blue')))
    # Tahmin Verisi
    fig.add_trace(go.Scatter(x=list(f_data['Period']), y=list(f_data['Forecast']), name='Tahmin', line=dict(color='red', dash='dash')))
    # Güven Aralığı
    fig.add_trace(go.Scatter(
        x=list(f_data['Period']) + list(f_data['Period'])[::-1],
        y=list(f_data['Upper_95']) + list(f_data['Lower_95'])[::-1],
        fill='toself', fillcolor='rgba(255,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'), name='%95 Güven Aralığı'
    ))
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
    st.plotly_chart(fig)
    
    st.subheader("Seçilen Ürün İçin Tahmin Tablosu")
    detail_df = f_data[['Month', 'Best_Model', 'Forecast', 'Lower_95', 'Upper_95']].copy()
    detail_df.columns = ['Ay', 'Kullanılan Model', 'Tahmin (Adet)', 'Alt Sınır', 'Üst Sınır']
    detail_df['Kullanılan Model'] = detail_df['Kullanılan Model'].map(lambda x: MODEL_LABELS.get(x, x))
    st.table(detail_df)

elif menu == "Tahmin Sonuçları":
    st.title("Önümüzdeki 3 Ayın Tahmin Karşılaştırması")
    st.markdown("---")
    
    pivot = forecast_df.pivot(index='Series', columns='Month', values='Forecast')
    pivot = pivot[['Mayis 2026', 'Haziran 2026', 'Temmuz 2026']]
    pivot = pivot.reindex(['X2', 'X8', 'X10', 'X16', 'N1_M', 'N1_L', 'N1_XL', 'N1'])
    
    st.subheader("Tüm Ürünlerin Gelecek Satış Beklentileri")
    fig = go.Figure()
    months = {'Mayis 2026': 'Mayıs', 'Haziran 2026': 'Haziran', 'Temmuz 2026': 'Temmuz'}
    
    for col in pivot.columns:
        fig.add_trace(go.Bar(
            name=months[col], x=pivot.index, y=pivot[col],
            text=pivot[col].astype(int), textposition='outside'
        ))
    
    fig.update_layout(
        barmode='group',
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        yaxis=dict(range=[0, pivot.max().max() * 1.25])  # Bar metinleri kesilmesin diye
    )
    st.plotly_chart(fig)

elif menu == "Model Karşılaştırma":
    st.title("Yapay Zeka & İstatistiksel Modellerin Performansı")
    st.markdown("---")
    
    selected_code = st.selectbox("Ürün seçin:", list(SERIES_LABELS.keys()), format_func=lambda x: SERIES_LABELS[x])
    s_metrics = metrics_df[metrics_df['Series'] == selected_code].sort_values('MAE').copy()
    
    st.subheader(f"{selected_code} İçin Hata Oranları (MAE)")
    fig = px.bar(s_metrics, x='Model', y='MAE', text=s_metrics['MAE'].round(0).astype(int))
    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis=dict(range=[0, s_metrics['MAE'].max() * 1.2]))
    st.plotly_chart(fig)
    
    st.subheader("Tüm Metrikler Tablosu")
    s_metrics['Model'] = s_metrics['Model'].map(lambda x: MODEL_LABELS.get(x, x))
    
    display_df = s_metrics.copy()
    display_df.columns = ['Ürün', 'Model', 'MAE', 'RMSE', 'WMAPE (%)']
    display_df['WMAPE (%)'] = display_df['WMAPE (%)'].apply(lambda x: f"%{x:.1f}")
    display_df['MAE'] = display_df['MAE'].round(0).astype(int)
    display_df['RMSE'] = display_df['RMSE'].round(0).astype(int)
    
    st.table(display_df[['Model', 'MAE', 'RMSE', 'WMAPE (%)']])

elif menu == "Proje Hakkında":
    st.title("Proje Hakkında")
    st.markdown("---")

    st.subheader("Projenin Amacı")
    st.markdown("""
    Bu karar destek sistemi, **Variteks** firmasının siparişe dayalı üretim planlamasını
    veri odaklı bir yaklaşımla iyileştirmek amacıyla geliştirilmiştir. Firma, üretim
    miktarlarını geleneksel olarak deneyime dayalı belirlemektedir. Bu projede ise
    geçmiş fatura verileri kullanılarak **makine öğrenmesi ve istatistiksel yöntemlerle**
    gelecek 3 aylık talep tahmini üretilmiş ve karar vericilere sunulmuştur.
    """)

    st.markdown("---")

    st.subheader("Veri Seti")
    v1, v2, v3 = st.columns(3)
    v1.metric("Kapsam", "Ocak 2023 - Nisan 2026")
    v2.metric("Toplam Süre", "40 Ay")
    v3.metric("Tahmin Dönemi", "Mayıs - Temmuz 2026")

    st.markdown("""
    Variteks firmasının fatura kayıtlarından elde edilen ham veriler; tarih, müşteri,
    ülke, marka, ürün kodu, beden, renk ve satış adedi bilgilerini içermektedir.
    Bu veriler aylık bazda toplanarak (aggregate edilerek) zaman serisi formatına
    dönüştürülmüştür.

    **Analiz edilen ürünler:**
    - **Nexus Markası:** X2, X8, X10, X16 kodlu ürünler
    - **Nitgo Markası:** N1 kodlu ürün (M, L, XL beden bazında ayrıştırılmış)
    """)

    st.markdown("---")

    st.subheader("Veri Ön İşleme")
    st.markdown("""
    Ham verilerin doğrudan modellere verilmesi yanıltıcı sonuçlar üretebilir.
    Bu nedenle aşağıdaki ön işleme adımları uygulanmıştır:

    1. **Aykırı Değer Tespiti ve Düzeltme:** IQR (Tukey) yöntemi ile tespit edilen
       aşırı büyük siparişler üst sınıra çekilmiştir. Böylece modelin trendi doğru
       öğrenmesi sağlanmıştır.
    2. **Özellik Mühendisliği (Feature Engineering):** Her bir zaman noktası için
       gecikme değerleri (lag), hareketli ortalamalar, mevsimsellik göstergeleri
       (sin/cos dönüşümleri), trend indeksi ve istatistiksel özetler türetilmiştir.
    3. **Eğitim-Test Ayrımı:** Verinin son 5 ayı test seti olarak ayrılmış,
       modellerin hiç görmediği bu dönem üzerinden performansları ölçülmüştür.
    """)

    st.markdown("---")

    st.subheader("Kullanılan Tahmin Yöntemleri (8 Algoritma)")

    col_ml, col_ts, col_basic = st.columns(3)

    with col_ml:
        st.markdown("**Makine Öğrenmesi**")
        st.markdown("""
        - **Random Forest:** Birden fazla karar ağacının ortalamasını alarak
          tahmin üretir. Aşırı öğrenmeye karşı dayanıklıdır.
        - **XGBoost:** Gradient Boosting tabanlı, sıralı ağaç eğitimi yapar.
          Hata düzeltme mekanizması sayesinde yüksek doğruluk sağlar.
        - **Doğrusal Regresyon:** Bağımsız değişkenlerle satış arasındaki
          doğrusal ilişkiyi modelleyen klasik istatistiksel yöntem.
        """)

    with col_ts:
        st.markdown("**Zaman Serisi**")
        st.markdown("""
        - **SARIMA:** Mevsimsel ARIMA modeli. Verideki trend ve mevsimsel
          örüntüleri matematiksel olarak ayrıştırır.
        - **Holt-Winters:** Üstel düzeltme yöntemi. Seviye, trend ve
          mevsimsellik bileşenlerini ayrı ayrı modeller.
        """)

    with col_basic:
        st.markdown("**Temel Yöntemler**")
        st.markdown("""
        - **Hareketli Ortalama (3 Ay):** Son 3 ayın ortalamasını tahmin
          olarak kullanır.
        - **Hareketli Ortalama (6 Ay):** Son 6 ayın ortalamasını tahmin
          olarak kullanır.
        - **Naive:** Son ayın değerini gelecek için sabit tahmin olarak
          kullanır (kıyaslama amaçlı).
        """)

    st.markdown("---")

    st.subheader("Değerlendirme Yaklaşımı")
    st.markdown("""
    Hata metrikleri olarak **MAE (Ortalama Mutlak Hata)**, **RMSE (Karekök Ortalama Kare Hata)**
    ve **WMAPE (Ağırlıklı Ortalama Mutlak Yüzde Hata)** kullanılmıştır.

    Geleneksel MAPE metriği, düşük satışlı aylarda yüzlerce kat şişerek yanıltıcı sonuçlar
    üretebilmektedir. Bu nedenle endüstride yaygın olarak tercih edilen **WMAPE**
    benimsenmiştir. WMAPE, toplam mutlak hatayı toplam gerçek satışa bölerek hesaplandığından,
    düşük hacimli dönemlerin etkisini dengelemektedir.
    """)

    st.latex(r"WMAPE = \frac{\sum |Gerçek - Tahmin|}{\sum Gerçek} \times 100")

    st.markdown("---")

    st.subheader("Model Seçim Stratejisi")
    st.markdown("""
    Her ürün için 8 algoritmanın tamamı eğitim verisiyle öğretilmiş, ardından test verisi
    üzerinde performansları karşılaştırılmıştır. **En düşük MAE değerine** sahip model,
    o ürünün "en iyi modeli" olarak seçilmiş ve gelecek 3 aylık tahminler bu modelle
    üretilmiştir. Naive modeli, yalnızca kıyaslama (benchmark) amacıyla tutulmuş olup
    nihai seçimde dikkate alınmamıştır.

    Bu yaklaşım sayesinde her ürün, kendi satış karakteristiğine en uygun algoritmayı
    otomatik olarak kullanmaktadır.
    """)

    st.markdown("---")

    st.subheader("Teknik Altyapı")
    st.markdown("""
    | Bileşen | Teknoloji |
    |---|---|
    | Programlama Dili | Python 3.13 |
    | Veri İşleme | Pandas, NumPy |
    | Makine Öğrenmesi | Scikit-learn, XGBoost |
    | Zaman Serisi | Statsmodels (SARIMA, Holt-Winters) |
    | Görselleştirme | Plotly |
    | Web Arayüzü | Streamlit |
    | Versiyon Kontrolü | GitHub |
    """)

