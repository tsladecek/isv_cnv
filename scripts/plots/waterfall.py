import plotly.graph_objects as go
import pandas as pd
import numpy as np


def _waterfall(shap_values,
               pathogenic_color="rgb(255, 0, 50)",
               benign_color="rgb(58, 130, 255)",
               width=1000,
               height=800,
               probability=True,
               text_position='outside',  # 'none' for no text
               title='',
               fontsize=15
               ):

    # SHAP Explainer class
    s = shap_values

    visibility_dict = {
        'default': [True, False, False],
        'sorted': [False, True, False],
        'abs_sorted': [False, False, True]
    }

    # DataFrame for Bar Charts
    data = pd.DataFrame({'Feature': s.feature_names, 'SHAP': s.values, 'Value': s.data, 'raw': s.raw_data})

    # colors
    cols = np.array([pathogenic_color if i > 0 else benign_color for i in s.values])
    data['cols'] = cols

    data = data.iloc[::-1]

    # sort data by SHAP value
    data = data.iloc[np.argsort(data.SHAP)]

    # hover text formating
    hover_text = '{}<br>Value: {:2.3f}'

    fig = go.Figure()

    # add value to the left of the name of the feature
    data["alt_Feature"] = [
        f'<span style="font-size: 10px; color: gray">({data.raw.iloc[i]})</span> = <span style="font-size: 14px;">{data.Feature.iloc[i]}</span>'
        for i in range(len(s.feature_names))]

    # Main Bar Plot
    fig.add_trace(
        go.Waterfall(
            x=data.SHAP,
            y=data.alt_Feature,
            base=s.base_values,
            orientation='h',
            hovertext=[hover_text.format(data.Feature.iloc[i], data.Value.iloc[i]) for i in range(len(s.feature_names))],
            hoverinfo="text + delta + initial",
            connector={"mode": "between", "line": {"width": 0.2, "color": "gray", "dash": "solid"}},
            decreasing={
                "marker": {"color": benign_color, "line": {'width': 0.1}}},
            increasing={"marker": {"color": pathogenic_color, "line": {'width': 0}}},
            name='',
            text=[np.round(i, 2) if i <= 0 else f'+{np.round(i, 2)}' for i in data.SHAP],
            textposition=text_position,
            textfont={'color': 'gray'}
        )
    )
    
    fig.add_shape(type='line',
                  x0=s.base_values,
                  y0=0,
                  x1=s.base_values,
                  y1=len(s.feature_names),
                  line=dict(color='grey', width=0.3, dash='dash'),
                  )

    # General Layout
    fig.update_layout(
        template='plotly_white',
        title=title,
        title_x=0.5,
        showlegend=False,
        xaxis_title=None,
        yaxis_title=None,
        width=width,
        height=height,
        xaxis={"showgrid": True,
               "nticks": 5,
               "range": [0, 1.2]# [s.base_values + np.min(np.cumsum(s.values)) - 0.1, s.base_values + np.max(np.cumsum(s.values)) + 0.1]
               },
        hoverlabel=dict(
            font_size=16,
            font_family="Rockwell",
            font=dict(color='white')
        ),
        font=dict(
            size=fontsize,    
        )

    )
    return fig
