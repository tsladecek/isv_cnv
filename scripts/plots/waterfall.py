import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scripts.ml.prepare_df import prepare
from scripts.ml.predict import open_model
from scripts.constants import LOSS_ATTRIBUTES, GAIN_ATTRIBUTES, HUMAN_READABLE, DPI
import shap


# %%
class _SHAP_values:
    def __init__(self, data_path, train_data_path, cnv_type, model_path, idx):
        # load data
        train_X, train_Y, data_X, data_Y = prepare(cnv_type,
                                                   train_data_path,
                                                   data_path, return_train=True)
        
        raw = pd.read_csv(data_path, sep='\t', compression='gzip')
        raw = raw.iloc[idx]
        attributes = [LOSS_ATTRIBUTES, GAIN_ATTRIBUTES][(cnv_type == 'gain') * 1]
        
        # open model
        model = open_model(model_path)
        
        # shap explainer
        explainer = shap.TreeExplainer(model, train_X, model_output='probability', feature_names=attributes)
        
        sv = explainer(data_X[idx])
        
        self.values = sv.values
        self.data = sv.data
        self.feature_names = [HUMAN_READABLE[i] for i in attributes]
        self.base_values = sv.base_values[0]
        self.raw_data = raw.loc[attributes].values.astype(np.int)


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
        f'<span style="font-size: {fontsize-3}px; color: gray">({data.raw.iloc[i]})</span> = <span style="font-size: {fontsize}px;">{data.Feature.iloc[i]}</span>'
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
                  line=dict(color='grey', width=1, dash='dash'),
                  )

    # General Layout
    fig.update_layout(
        template='simple_white',
        title=title,
        title_x=0.5,
        showlegend=False,
        xaxis_title="Attribute Contribution (SHAP probability)",
        yaxis_title=None,
        width=width,
        height=height,
        plot_bgcolor="#FFF",
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
            family="Verdana"
        )

    )
    return fig

# %%
# df = pd.read_csv("data/evaluation_data/five_syndroms.tsv.gz", sep='\t', compression='gzip')

# # %%
# i = 0
# shap_vals = _SHAP_values("data/evaluation_data/five_syndroms.tsv.gz",
#                           "data/train_loss.tsv.gz",
#                           "loss",
#                           "results/ISV_loss.json",
#                           idx=i)
# pos = f"chr{df.iloc[i].chrom}:{df.iloc[i].start}-{df.iloc[i].end}"
# fig = _waterfall(shap_vals, height=1000, width=900,
#                   title=df.iloc[i].info + ', ' + pos,
#                   fontsize=18)

# fig.write_image("../plotly_example.png",
#                 format="png",
#                 scale=DPI/100)
