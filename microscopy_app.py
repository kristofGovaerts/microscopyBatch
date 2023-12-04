from dash import Dash, dcc, html, Input, Output
import os
import glob
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from processing.segmentation import find_blob_centers
from processing.preprocessing import remove_small_objects, remove_spatial_trend, normalize

# global parameters - these are the variables to edit
FOLDER = r'C:\Users\govaerts.kristof\OneDrive - GroupeFD\Documents\Kristof_phenotyping\microscopy\HTH'
EXT = '*.tif'
DETREND = True
LABEL_COLOR = (255, 0, 0)  # color for annotations in RGB format

# do prep work
os.chdir(FOLDER)
files = glob.glob(EXT)

app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.Div([
        dcc.Graph(id='base_im', style={'display': 'inline-block',
                                          'height' : '30vw'}),
        dcc.Graph(id='annotated_im', style={'display': 'inline-block',
                                          'height' : '30vw'})
    ], style = {'height': '1600'}),
    html.Br(),
    html.Div(children="InputType", className="menu-title"),
    html.Div([
        dcc.Dropdown(
            id="filename",
            options=[
                        {"label": f, "value": f}
                        for f in np.sort(files)
                    ],
            value=files[0],
            clearable=False,
            className='dropdown',
            style={'display': 'inline-block', 'width': '20vw'}
        ),
    ]),
    html.Br(),
    html.Div([
        html.Div([html.H4("brightfield thresh")], style={'width': '30%', 'display': 'inline-block'}),
        html.Div([html.H4("fluo thresh")], style={'width': '30%', 'display': 'inline-block'}),
        html.Div([html.H4("size thresh")], style={'width': '30%', 'display': 'inline-block'}),
    ], style={"display": "flex"}),
    html.Div([
        html.Div([dcc.Input(id="thresh1", type="number", value=0.01, debounce=True)],
                 style={'width': '30%', 'display': 'inline-block'}),
        html.Div([dcc.Input(id='thresh2', type='number', value=0.2, debounce=True)],
                 style={'width': '30%', 'display': 'inline-block'}),
        html.Div([dcc.Input(id='sizethresh', type='number', value=50, debounce=True)],
                 style={'width': '30%', 'display': 'inline-block'})
    ], style={'display': 'flex'}),
])


@app.callback(
    [
        Output('base_im', 'figure'),
        Output('annotated_im', 'figure'),
    ],
    [
        Input('filename', 'value'),
        Input('thresh1', 'value'),
        Input('thresh2', 'value'),
        Input('sizethresh', 'value'),
    ]
)
def update_figure(f, thresh1, thresh2, sizethresh):
    im = cv2.imread(f)
    im = imutils.resize(im, width=1292)
    im = cv2.GaussianBlur(im, (5, 5), 5)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv /= [180, 255, 255]  # normalize

    BLANK = False
    if np.median(hsv[:, :, 2]) > 0.5:  # brightfield image, otherwise it's a fluo image
        print("Brightfield image. Inverting.")
        MODE = 'brightfield'
        hsv[:, :, 2] = np.abs(hsv[:, :, 2] - 1)  # invert
        THRESH = thresh1
        # intensity = hsv[:, :, 2] * hsv[:, :, 1]
        intensity = hsv[:, :, 2]
        intensity = cv2.GaussianBlur((255 * normalize(intensity)).astype(np.uint8), (5, 5), 5).astype(np.float32) / 255
    else:
        if np.max(hsv[:, :, 2]) < 0.1:
            print("No fluorescence detected.")
            fig2 = px.imshow(im, title=f + " / Count: {}".format(0))
            BLANK = True

        MODE = 'fluorescence'
        THRESH = thresh2
        print("Fluorescence image. No inversion.")
        intensity = hsv[:, :, 2]

    intensity[intensity < 0.01] = 0
    if DETREND:
        intensity = remove_spatial_trend(intensity, filter_size=21, highpass=False, norm=False)
    intensity[intensity < 0] = 0
    intensity = normalize(intensity)
    intensity[hsv[:, :, 1] < 0.1] = 0

    m = np.zeros((hsv.shape[0], hsv.shape[1]))
    m[intensity > THRESH] = 1
    m = remove_small_objects(m.astype(np.uint8), min_size=sizethresh)

    outim1, centers1 = find_blob_centers(m, to_label=im, color=LABEL_COLOR)

    fig1 = px.imshow(im, title="{} image".format(MODE))

    if not BLANK:
        fig2 = px.imshow(outim1, title=f + " / Count: {}".format(len(centers1)))

    fig1.update_layout(margin={"t":100,"b":0,"r":0,"l":0,"pad":0})
    fig2.update_layout(margin={"t":100,"b":0,"r":0,"l":0,"pad":0})

    return fig1, fig2


if __name__ == '__main__':
    app.run_server(debug=True)
