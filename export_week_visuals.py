import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from folium.plugins import HeatMap


def main():
    # 1) Static matplotlib figure (Week 1)
    w1 = pd.read_csv(
        'Week1/Police_Department_Incident_Reports__2018_to_Present_20260203.csv',
        usecols=['Incident Year']
    )
    yearly = w1.groupby('Incident Year').size().reset_index(name='Incidents').sort_values('Incident Year')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly['Incident Year'], yearly['Incidents'], marker='o', linewidth=2)
    ax.set_title('SF Incidents per Year (2018-Present)')
    ax.set_xlabel('Incident Year')
    ax.set_ylabel('Number of incidents')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig('images/yearly_crime_trends.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    # 2) Interactive Plotly chart (Week 2)
    w2_time = pd.read_csv(
        'Week2/Police_Department_Incident_Reports__Historical_2003_to_May_2018_20260210.csv',
        usecols=['Time']
    )
    w2_time['hour'] = pd.to_numeric(w2_time['Time'].astype(str).str.split(':').str[0], errors='coerce')
    hourly = w2_time.dropna(subset=['hour']).groupby('hour').size().reset_index(name='Incidents')
    hourly['hour'] = hourly['hour'].astype(int)
    hourly = hourly.sort_values('hour')

    fig_plotly = px.bar(
        hourly,
        x='hour',
        y='Incidents',
        title='Historical SF Incidents by Hour of Day (2003-2018)',
        labels={'hour': 'Hour of day', 'Incidents': 'Number of incidents'}
    )
    fig_plotly.update_layout(xaxis=dict(dtick=1))
    fig_plotly.write_html('visualizations/hourly_crime_plotly.html', include_plotlyjs='cdn')

    # 3) Interactive Folium map (Week 2)
    w2_xy = pd.read_csv(
        'Week2/Police_Department_Incident_Reports__Historical_2003_to_May_2018_20260210.csv',
        usecols=['X', 'Y']
    ).rename(columns={'Y': 'lat', 'X': 'lon'})

    coords = w2_xy.apply(pd.to_numeric, errors='coerce').dropna()
    coords = coords[(coords['lat'].between(37.68, 37.84)) & (coords['lon'].between(-122.53, -122.35))]
    coords_sample = coords.sample(n=min(8000, len(coords)), random_state=42)

    m = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles='OpenStreetMap')
    HeatMap(coords_sample[['lat', 'lon']].values.tolist(), radius=10, blur=14, min_opacity=0.35).add_to(m)
    m.save('visualizations/sf_crime_heatmap_folium.html')

    print('Created images/yearly_crime_trends.png')
    print('Created visualizations/hourly_crime_plotly.html')
    print('Created visualizations/sf_crime_heatmap_folium.html')


if __name__ == '__main__':
    main()
