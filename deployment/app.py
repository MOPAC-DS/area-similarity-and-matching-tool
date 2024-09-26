## MOPAC Spatial Propensity Matching

## Daniel Hammocks - 2024-08-14
## GH: dhammo2

## This code takes the JSON extracted from the Metropolitan Police R HTML Ouput
## and creates a geoJSON/Shapefile of control and intervention groups including
## all data displayed on the graph.

###############################################################################
################################# APPLICATION #################################
###############################################################################


#%% Versioning

# v1.0.0 - Original Dashboard
# v2.0.0 - Introduced Download Option
# v3.0.0 - Added Multi-Page Functionality 
# v4.0.0 - Introduced Quality Assurance
#        - Altered Calculation for PSM to search control only.
# v5.0.0 - Ensure that Area Matching Does Not Return Self
# v6.0.0 - Added Quality Assurance
# v7.0.0 - Complete Quality Assurance Information


#%% NOTES

#DISABLE LOGGING WHEN DONE

#Put Error Notices for QA at top.

#%% Required Libraries

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import io
import base64
from datetime import datetime
from scipy import stats
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


import logging
import os




#%% App Initialisation

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = "MOPAC | DS"

logging.basicConfig(level=logging.DEBUG)

#%% Navigation Layout


navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Overview", href="/")),
        dbc.NavItem(dbc.NavLink("Area Similarity & Matching", href="/match")),
        dbc.NavItem(dbc.NavLink("Quality Assurance", href="/quality")),

    ],
    brand="MOPAC | DS - Area Similarity and Matching Application",
    brand_href="#",
    color="primary",
    dark=True,
    fluid=True
)


#%% Page Layout

overview_layout = html.Div(
    children=[html.Br(),
              html.H1(children="Overview"),
              html.P("This application enables users to identify similar "
                     "geographic areas based on a set of covariates "
                     "(characteristics) related to each area. Two methods have "
                     "been implemented enabling the user to both identify "
                     "similar areas based on a set of characteristics but to "
                     "also identify similar areas once a treatment has already "
                     "been assigned. The two methods are detailed below."),
              
              html.Br(),
              html.Hr(),
              
                html.H3('Data Formatting Requirements'),
                html.Div([
                    html.Div([
                        html.H4('Example Data Structure'),
                        
                        html.Br(),

                        html.Table([
                            html.Thead(
                                html.Tr([
                                    html.Th('Area_ID', style={'border': '1px solid black'}),
                                    html.Th('Population', style={'border': '1px solid black'}),
                                    html.Th('Income', style={'border': '1px solid black'}),
                                    html.Th('Education', style={'border': '1px solid black'}),
                                    html.Th('Treatment (optional)', style={'border': '1px solid black'})
                                ], style={'background-color': '#f2f2f2', 'font-weight': 'bold', 'text-align': 'center'})
                            ),
                            html.Tbody([
                                html.Tr([
                                    html.Td('Area_001', style={'border': '1px solid black'}),
                                    html.Td('10000', style={'border': '1px solid black'}),
                                    html.Td('50000', style={'border': '1px solid black'}),
                                    html.Td('12', style={'border': '1px solid black'}),
                                    html.Td('1', style={'border': '1px solid black'})
                                ]),
                                html.Tr([
                                    html.Td('Area_002', style={'border': '1px solid black'}),
                                    html.Td('15000', style={'border': '1px solid black'}),
                                    html.Td('60000', style={'border': '1px solid black'}),
                                    html.Td('14', style={'border': '1px solid black'}),
                                    html.Td('0', style={'border': '1px solid black'})
                                ]),
                                html.Tr([
                                    html.Td('Area_003', style={'border': '1px solid black'}),
                                    html.Td('12000', style={'border': '1px solid black'}),
                                    html.Td('55000', style={'border': '1px solid black'}),
                                    html.Td('13', style={'border': '1px solid black'}),
                                    html.Td('1', style={'border': '1px solid black'})
                                ]),
                                html.Tr([
                                    html.Td('...', style={'border': '1px solid black'}),
                                    html.Td('...', style={'border': '1px solid black'}),
                                    html.Td('...', style={'border': '1px solid black'}),
                                    html.Td('...', style={'border': '1px solid black'}),
                                    html.Td('...', style={'border': '1px solid black'})
                                ])
                            ], style={'text-align': 'center'}),
                        ], style={
                            'width': '80%',
                            'border-collapse': 'collapse',
                            'margin-top': '10px',
                            'border': '1px solid black'
                        }),
                            
                        html.Br(),
                        html.A(
                            html.Img(src='assets/DataOverview.png',
                                     style={'width': '85%', 'height': 'auto'})
                            , href="assets/DataOverview.png", target="_blank", style={'font-size': '20px', 'text-decoration': 'none', 'color': 'blue'}),
                        html.P("Click to Expand.", style={'width': '85%', 'textAlign': 'center'})
    
    
                        
                    ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),
                        
                
                    html.Div([
                        html.H4('Required Data Format'),
                        html.P('To use the Area Similarity and Matching application, please provide your data in a CSV format with the following structure:'),
                
                        html.H5('1. Area ID/Name'),
                        html.P('A unique identifier or name for each area (e.g., "Area_ID" or "Area_Name"). This will be used to identify and match areas.'),
                
                        html.H5('2. Covariate Columns'),
                        html.P('One or more columns containing the demographic or other relevant variables (covariates) that will be used to compare areas. Examples include population, income, education level, etc.'),
                
                        html.H5('3. [Optional] Treatment Column'),
                        html.P('The treatment column is optional but can be included if you want to perform Propensity Score Matching (PSM). This should be a binary column indicating whether an area has received a treatment (e.g., "Treatment", where 1 = treated, 0 = not treated). If this column is provided, the application will use it to perform PSM. If not provided, the application will simply match areas based on similarity using covariates.'),
                        
                        html.H4('Key Points'),
                        html.Ul([
                            html.Li('The Area ID/Name must be unique for each area.'),
                            html.Li('Covariates should be numeric and relevant to the analysis.'),
                            html.Li('The Treatment column is optional; if included, it allows for Propensity Score Matching, otherwise, the app can identify similar areas without considering treatment.')
                        ])
                    ], style={'width': '66%', 'display': 'inline-block', 'vertical-align': 'top'}),
                  ]),


              html.Hr(),
              dbc.Row([
                  
                  dbc.Col([     
                          html.H3("Propensity Score Matching (PSM)"),
                          html.B("This method assumes the treatment areas are known.", style={'color':'red'}),
                          
                            html.H5('What is the Goal?'),
                            html.P('The goal of PSM is to find areas that are most similar to each other based on certain covariates (characteristics like population, income, etc.) in order to balance the covariates between the treatment and control groups. By ensuring that the treatment and control groups are balanced in terms of covariates, PSM helps reduce confounding bias, leading to more accurate estimates of the treatment effect.'),
                        
                            html.H5('Step 1: Collect Information'),
                            html.Ul([
                                html.Li('Start with a list of areas that received a "treatment" (e.g., a new program or policy) and potential control areas.'),
                                html.Li('Gather information (like demographics) about both treated and nearby areas.')
                            ]),
                        
                            html.H5('Step 2: Calculate a Score (Using Logistic Regression)'),
                            html.Ul([
                                html.Li('Use a statistical method called logistic regression to calculate a "score" for each area.'),
                                html.Li('This propensity score tells us how likely each area is to have received the treatment based on its characteristics.')
                            ]),
                        
                            html.H5('Step 3: Match Areas (Using Nearest Neighbor)'),
                            html.Ul([
                                html.Li('For each treated area, find a similar untreated area with the closest score.'),
                                html.Li('This is like finding the "nearest neighbor" in terms of similarity.')
                            ]),
                        
                            html.H5('Step 4: Identify Similar Areas'),
                            html.Ul([
                                html.Li('After matching, you have pairs of areas where each treated area is matched with a nearby area that is most similar based on the characteristics considered.')
                            ]),
                        
                            html.H5('Result'),
                            html.P('We now know which areas are most similar to each other, helping us understand the impact of the treatment by comparing them.')
                                              
                  ], width = 6),
                  
                  dbc.Col([     
                          html.H3("Nearest Neighbour Matching (NNM)"),
                        
                            html.H5('What is the Goal?'),
                            html.P('We want to find areas that are most similar to each other based on certain characteristics (like population, income, etc.), without focusing on any specific treatment.'),
                        
                            html.H5('Step 1: Collect Information'),
                            html.Ul([
                                html.Li('Start by gathering information about the areas you are interested in, such as demographics, socio-economic data, or any other relevant characteristics.')
                            ]),
                        
                            html.H5('Step 2: Calculate Similarity'),
                            html.Ul([
                                html.Li('Use the characteristics (also called covariates) of each area to calculate a measure of similarity between areas.'),
                                html.Li('The most common approach is to calculate the "distance" between areas in the space defined by these covariates.')
                            ]),
                        
                            html.H5('Step 3: Find Nearest Neighbors'),
                            html.Ul([
                                html.Li('For each area, find the other area that has the smallest "distance" to it.'),
                                html.Li('This is known as finding the "nearest neighbor."'),
                                html.Li('The nearest neighbor is considered the most similar area based on the covariates.')
                            ]),
                        
                            html.H5('Step 4: Identify Similar Areas'),
                            html.Ul([
                                html.Li('After finding the nearest neighbors, you have a list of pairs of areas where each pair consists of the most similar areas based on the characteristics considered.')
                            ]),
                        
                            html.H5('Result'),
                            html.P('You now know which areas are most similar to each other, based purely on their characteristics, allowing for various analyses or comparisons.')

                      
                  ], width = 6)
              
              ])
              

              ]
)


match_layout = html.Div(
    children=[
        html.Br(),
        html.H1("Area Similarity and Matching", className="mb-2"),
        
            dbc.Row([
                dbc.Col(html.P("Upload a CSV file with your geographic areas and demographic variables. "
                                "Select the analysis mode below and follow the steps to identify similar areas."),
                        className="mb-4")
            ]),
            
            dbc.Row([
            
                dbc.Col([
                        html.H4("Settings"),
                        
                        dbc.Col(html.H5("Treatment Status"), className="mb-2"),
                        dbc.Col(html.P("Have the treatment areas already been selected?" ),
                                className="mb-4"),
                        dbc.RadioItems(
                            options=[
                                {"label": "Yes", "value": 1},
                                {"label": "No", "value": 0}
                            ],
                            value=0,  # Default value
                            id="treatment-toggle",
                            inline=True,  # Display radio buttons inline
                            className="mb-4"
                        ),
                        
                        dbc.Col(html.H5("File Upload"), className="mb-2"),
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                'borderWidth': '1px', 'borderStyle': 'dashed',
                                'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px',
                                'margin-left' : '0px'
                            },
                            multiple=False
                        ),
                        dbc.Progress(id="upload-progress",
                                      value=0,
                                      striped=True,
                                      animated=True,
                                      style={'transition': 'width 0.5s',
                                            'backgroundColor': 'lightgray'},
                                      className="mb-4"),
                        html.Div(id='output-data-upload'),
                        
                        
                        dbc.Col(html.H5("Column Selection"), className="mb-2"),
                        
                        html.H6("Select Area ID/Name Column"),
                        dcc.Dropdown(id='area-id-column', placeholder="Select Area ID/Name Column"),
                        
                        html.H6("Select Treatment Column (if applicable)"),
                        dcc.Dropdown(id='treatment-column', placeholder="Select Treatment Column"),
                        
                        html.H6("Select Covariate Columns"),
                        dcc.Dropdown(id='covariate-columns', multi=True, placeholder="Select Covariate Columns"),
                        
                        html.Br(),
                        html.Button("Run Analysis", id="run-analysis", n_clicks=0, className="btn btn-primary"),
                        html.Button("Clear Data", id="clear-button", className="btn btn-primary", 
                                    style={
                                        'backgroundColor': 'red',  # Red background color
                                        'color': 'white',          # White text color
                                        'border': 'none',          # Remove border
                                        'cursor': 'pointer',
                                        'margin-left': '20px'
                                    }),
                        
                        html.Br(),
                        html.Br(),
                        html.H4("Download Results"),
                        dcc.Store(id='raw-results'),
                        html.Div(html.Button("Download Results",
                                              id = "download-button",
                                              n_clicks = 0,
                                              className="btn btn-secondary"),
                                              style={'margin-top': '10px'}),
                        
                        dcc.Download(id="download-data") 
            
                ], width=3),
                
                dbc.Col([     
                        html.H4("Output"),
                        html.Div(id="analysis-output")
                    
                ], width=9)
            
            ])
        
    ]
)

quality_layout = html.Div([
    html.H1("Propensity Score Matching Evaluation Metrics"),
    html.Br(),
    
    dbc.Row([
        dbc.Col([
            html.Button("Run Quality Assurance Script", id="run-quality", n_clicks=0, className="btn btn-primary", style = {'margin-bottom': '20px',}),
        ], width = 3),
        dbc.Col([
            html.Div(id='output-quality-error', style={'color':'red'})
        ], width = 9)
    ]),

    html.Br(),

    dbc.Row([
        html.H3("Distribution of Propensity Scores"),
        
        dbc.Col([     
                html.H4("Plot"),
                
                html.Div(id='propensity-distribution')
                          
                   
        ], width=8),
        
        dbc.Col([     
                html.H4("Explanation"),
                
                html.P(
                    "In this application, the Propensity Score Distribution visualises the likelihood that an area "
                    "or individual would receive the treatment based on observed covariates. The propensity score "
                    "is a value between 0 and 1 that indicates the probability of receiving treatment, given a set "
                    "of characteristics. "
                    "The plot displays the distribution of propensity scores "
                    "for both the treated and control groups. "
                    ),
                
                html.P("Key Considerations:"),
                html.Ul([
                    html.Li(
                        "Overlap of Distributions: Ideally, the propensity score distributions for treated and control "
                        "groups should overlap significantly. This overlap indicates that the groups are comparable in terms "
                        "of covariates, which is crucial for reducing bias in the analysis."
                    ),
                    html.Li(
                        "Quality of Matching: After matching, a significant overlap with similar shapes in the distributions "
                        "suggests successful balancing of covariates between the groups. Poor overlap or disparate shapes may "
                        "indicate inadequate matching."
                    ),
                    html.Li(
                        "Potential Issues: Sparse regions in the distribution suggest a lack of overlap, meaning some treated "
                        "units lack suitable control matches. Additionally, outliers or extreme scores may signal issues with "
                        "the model or covariates."
                    ),
                ]),
                html.P(
                    "By examining these distributions, you can assess the effectiveness of the matching process and identify "
                    "any potential issues that could affect the reliability of the analysis."
                )
            
        ],
            width=4,
            style={
                        'backgroundColor': 'rgba(173, 216, 230, 0.8)',  # Pale blue color
                        'padding': '20px',
                        'border': '1px solid #ccc'
                    })
    ],
    style={
                'backgroundColor': 'rgba(173, 216, 230, 0.5)',  # Pale blue color
                'padding': '20px',
                'border': '1px solid #ccc'
            }),
    
    dbc.Row([
        html.H3("Balance of Covariates"),
        
        dbc.Col([     
                html.H4("Explanation"),
                
        dcc.Markdown('''
        **Balance of Covariates Distribution Plots** are visual tools used to compare the distributions of covariates (variables) between treated and control groups after matching. The goal is to assess if the matching process has made the treated and control groups similar in terms of the covariates. If the matching is successful, the distributions for the treated and control groups should appear more similar, suggesting that the groups are well balanced.
    
        **Standard Mean Difference (SMD)** is a numerical metric used to quantify the balance of covariates between treated and control groups. It measures the difference in means of a covariate between the two groups, standardised by the pooled standard deviation. An SMD close to 0 indicates that the covariate is well-balanced between the groups, while a larger SMD suggests imbalance.
    
        **Interpreting the Plots and SMD:**
        - **Well-Matched Covariates:** After matching, the covariate distributions between treated and control groups should overlap significantly, and the SMD should be close to 0.
        - **Poor Matching:** If the distributions differ significantly or the SMD is still large after matching, it indicates that the matching did not adequately balance the groups.
        - **SMD Threshold:** As a general rule, an SMD below 0.1 is considered acceptable, indicating a good balance between the groups.
    
        These tools help assess the effectiveness of the matching process and ensure that the treatment effect is estimated from comparable groups.
        '''),
                
        ], width=3,
            style={
                        'backgroundColor': 'rgba(172, 230, 174, 0.8)',
                        'padding': '20px',
                        'border': '1px solid #ccc'
                    }),
        
        dbc.Col([     
                html.H4("Plot & Standard Mean Difference (SMD)"),
                html.Div([
                html.Div(id='output-covariate-balance',
                          style={
                              'display': 'grid',
                              'gridTemplateColumns': 'repeat(auto-fill, minmax(400px, 1fr))',
                              'gap': '20px'})
                          ], style={'padding': '20px'})
                          
                   
        ], width=9)
        

    ],
        style={
                    'backgroundColor': 'rgba(172, 230, 174, 0.5)',  # Pale blue color
                    'padding': '20px',
                    'border': '1px solid #ccc'
                }),
    
    dbc.Row([
        html.H3("t-Test"),
        
        dbc.Col([     
                html.H4("Results"),
                
                html.Br(),
                
                html.Div(id="ttest-output")
            
        ], width=3),
        
        dbc.Col([     
                html.H4("Explanation"),
                
                dcc.Markdown('''                    
                    In Propensity Score Matching (PSM), the **t-test** is used to assess the balance of covariates between the treated group (those who received the treatment) and the control group (those who did not receive the treatment).
                    
                    The t-test evaluates the null hypothesis that the means of a specific covariate are equal between the treated and control groups. Essentially, it checks whether the difference in means is statistically significant. A well-balanced covariate between the groups is indicated by a high p-value, suggesting that the matching process has been effective.
                    
                    ###### How to Interpret the t-Test in PSM?
                    
                    - **P-value**:
                      - **High P-value (e.g., p > 0.05)**: Indicates no statistically significant difference in means between the treated and control groups, suggesting that the covariate is well balanced after matching.
                      - **Low P-value (e.g., p < 0.05)**: Indicates a significant difference in means, suggesting that the covariate is not well balanced, even after matching.
                    
                    ###### Why is the t-Test Important in PSM?
                    
                    The t-test is crucial for assessing the success of the matching process. By checking for balance in covariates, it helps ensure that the treated and control groups are comparable. This balance is essential for reducing confounding bias and obtaining accurate estimates of the treatment effect. Significant differences detected by the t-test may indicate that the matching process needs improvement to achieve better balance.
                    
                    In summary, the t-test helps validate the effectiveness of the matching process by evaluating whether covariates are balanced between the treated and control groups, ensuring a more accurate assessment of treatment effects.
                    
                    ''')
            
        ], width=9,
            style={
                        'backgroundColor': 'rgba(157, 39, 245, 0.3)',
                        'padding': '20px',
                        'border': '1px solid #ccc'
                    })    
    ],
        style={
                    'backgroundColor': 'rgba(157, 39, 245, 0.2)',
                    'padding': '20px',
                    'border': '1px solid #ccc'
                })
    

])


#%% App Layout

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        navbar,
        dbc.Container(id="page-content", className="mb-4", fluid=True),
        dcc.Store(id='stored-data', storage_type='session')
    ]
)

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/":
        return overview_layout
    elif pathname == "/match":
        return match_layout
    elif pathname == "/quality":
        return quality_layout
    else:
        return dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognized..."),
            ]
        )



#%% Retain Data Across Pages

#Callback to store input values automatically when they change

@app.callback(
    Output('stored-data', 'data'),
    Input('treatment-toggle', 'value'),
    Input('upload-data', 'contents'),
    Input('upload-data', 'filename'),
    Input('area-id-column', 'value'),
    Input('treatment-column', 'value'),
    Input('covariate-columns', 'value')
)

def store_inputs(treatment_toggle, contents, filename, area_id_col, treatment_col, covariates_cols):
    
    return {'treatment_toggle': treatment_toggle,
            'contents': contents,
            'filename': filename,
            'area_id_col': area_id_col,
            'treatment_col': treatment_col,
            'covariate-columns': covariates_cols}


#Load Data When Revisit Matching
@app.callback(
    Output('treatment-toggle', 'value'),
    Output('upload-data', 'contents'),
    Output('upload-data', 'filename'),
    Output('area-id-column', 'value'),
    Output('treatment-column', 'value'),
    Output('covariate-columns', 'value'),
    Output('run-analysis', 'n_clicks'),
    Input('url', 'pathname'),
    State('stored-data', 'data')
)
def restore_toggle_state(pathname, stored_data):
   
    if pathname == '/match' and stored_data is not None:
        return stored_data.get('treatment_toggle'), stored_data.get('contents'), stored_data.get('filename'), stored_data.get('area_id_col'), stored_data.get('treatment_col'), stored_data.get('covariate-columns'), 1

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


#Add Clear Button

@app.callback(
    Output('stored-data', 'data', allow_duplicate=True),
    Output('treatment-toggle', 'value', allow_duplicate=True),
    Output('upload-data', 'contents', allow_duplicate=True),
    Output('upload-data', 'filename', allow_duplicate=True),
    Output('area-id-column', 'value', allow_duplicate=True),
    Output('treatment-column', 'value', allow_duplicate=True),
    Output('covariate-columns', 'value', allow_duplicate=True),
    Output('run-analysis', 'n_clicks', allow_duplicate=True),
    Input('clear-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_stored_data(n_clicks):
    if n_clicks:

        return None, None, None, None, None, None, None, 0
    
    return dash.no_update
    
#%% App Functions



# Utility function to parse the uploaded CSV file
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return None
    except Exception as e:
        print(e)
        return None
    return df

# Update the progress bar and dropdown options based on uploaded file
@app.callback(
    Output('upload-progress', 'value'),
    Output('upload-progress', 'children'),
    Output('upload-progress', 'style'),
    Output('area-id-column', 'options'),
    Output('treatment-column', 'options'),
    Output('covariate-columns', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)

def update_progress_bar(contents, filename):
    if contents is None:
        return 0, "", {'backgroundColor': 'lightgray'}, [], [], []

    # Start the progress bar at 50% when the file is detected
    progress_value = 50
    progress_text = "Uploading..."

    # Parse the file
    df = parse_contents(contents, filename)
    if df is not None:
        columns = [{'label': col, 'value': col} for col in df.columns]

        # Finish the progress bar when the file is fully loaded
        return 100, "Upload Complete", {'backgroundColor': 'green'}, columns, columns, columns
    else:
        return 0, "Failed to upload", {'backgroundColor': 'red'}, [], [], []


# Conditional analysis based on treatment selection
@app.callback(
    [Output('analysis-output', 'children'),
    Output('raw-results', 'data')],
    Input('run-analysis', 'n_clicks'),
    State('treatment-toggle', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('area-id-column', 'value'),
    State('treatment-column', 'value'),
    State('covariate-columns', 'value'),
    prevent_initial_call=True
)
def run_analysis(n_clicks, treatment_toggle, contents, filename, area_id_col, treatment_col, covariates_cols):
    if n_clicks == 0 or contents is None:
        return "", ""
    
    df = parse_contents(contents, filename)
    if df is not None and area_id_col is not None and covariates_cols:
        # Set the area ID as the index for easier referencing
        df.set_index(area_id_col, inplace=True)
        
        #Standardize Covariates
        scaler = StandardScaler()
        covariates_scaled = scaler.fit_transform(df[covariates_cols])

        if treatment_toggle:
            # Treatment areas selected: Perform propensity score matching
            if treatment_col is None:
                return "Please select a treatment column.", ""
            
            if area_id_col == treatment_col:
                return "The treatment column must be different from the label column.", ""
            
            if not set(df[treatment_col].unique()).issubset({0, 1}):
                return "Treatment column must contain a binary response. 1 = Treatment // 0 = Control.", ""
            
            #Estimate Propensity Scores
            logistic = LogisticRegression()
            logistic.fit(covariates_scaled, df[treatment_col])
            df['propensity_score'] = logistic.predict_proba(covariates_scaled)[:, 1]
            
            #Extract Treatment & Control
            treated = df[df[treatment_col] == 1]
            control = df[df[treatment_col] == 0]
            
            #Perform Matching
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(control[['propensity_score']])
            
            treated_indices = df[df[treatment_col] == 1].index
            control_indices = df[df[treatment_col] == 0].index
            
            if len(treated_indices) > 0:
                distances, indices = nn.kneighbors(treated[['propensity_score']])
                matched_data = pd.DataFrame({
                    'Treated_Area': treated_indices,
                    'Matched_Control_Area': control.iloc[indices.flatten()].index,
                    'Distance': distances.flatten()
                })
                
                # Create raw results dataframe
                raw_results = matched_data.copy()
                
                return dbc.Table.from_dataframe(matched_data, striped=True, bordered=True, hover=True), raw_results.to_dict('records')
            
            return "No treated areas to match.", ""
        
        else:
            # No treatment: Find similar areas using nearest neighbors
            nn = NearestNeighbors(n_neighbors=2)  # 2 to get the nearest neighbor excluding the point itself
            nn.fit(covariates_scaled)
            
            distances, indices = nn.kneighbors(covariates_scaled)
            
            #Filter Self Returns
            indicesResults = []
            
            for i in range(len(indices)):
                
                #If NN 1 == index select NN 2 else NN 1
                if int(indices[i][0]) == i:
                    indicesResults.append((indices[i][1], distances[i][1]))
                else:
                    indicesResults.append((indices[i][0], distances[i][0]))
                    
            indicesResults = np.array(indicesResults)
            
            
            similar_areas = pd.DataFrame({
                'Area': df.index,
                'Most_Similar_Area': df.index[indicesResults[:, 0].astype(int)],
                'Similarity_Score': indicesResults[:, 1]
            })
            
            # Create raw results dataframe
            raw_results = similar_areas.copy()
            
            return dbc.Table.from_dataframe(similar_areas, striped=True, bordered=True, hover=True), raw_results.to_dict('records')
    
    return "Please upload a valid CSV file and select the appropriate columns.", ""




# Example callback for downloading data
@app.callback(
    Output("download-data", "data"),
    Input("download-button", "n_clicks"),
    State("raw-results", "data"),
    prevent_initial_call=True
)
def download_results(n_clicks, raw_results):
    if n_clicks > 0 and raw_results is not None:
        # Convert raw results to DataFrame and then to CSV
        df_raw = pd.DataFrame(raw_results)
        
        # Get the current time
        now = datetime.now()
        current_dt = now.strftime("%Y-%m-%d_%H%M%S")

        return dcc.send_data_frame(df_raw.to_csv, "MOPAC_SpatialMatching_"+str(current_dt)+".csv")
    

#%% Quality Assurance Functions


def calculate_smd(df, treatment_col, covariates_cols, state):
    smd = []
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    for cov in covariates_cols:
        treated_mean = treated[cov].mean()
        control_mean = control[cov].mean()
        treated_std = treated[cov].std()
        control_std = control[cov].std()
        
        pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
        #smd[cov] = np.abs(treated_mean - control_mean) / pooled_std
        
        smd.append(f'SMD {state}: {np.abs(treated_mean - control_mean) / pooled_std}')
    
    return smd


def plot_covariate_distributions(df, treatment_col, covariates_cols):
    plots = []
    for cov in covariates_cols:
        fig = go.Figure()
        
        # Define colors for treatments
        treatments = df[treatment_col].unique()
        colors = ['blue', 'green']  # Customize as needed
        treatment_colors = {treatment: colors[i % len(colors)] for i, treatment in enumerate(treatments)}
        label_mapping = {1: 'Treatment', 0: 'Control'}  # Adjust as needed

        # Add a histogram trace for each treatment
        for treatment in treatments:
            subset = df[df[treatment_col] == treatment]
            color = treatment_colors[treatment]
            
            # Add histogram trace
            fig.add_trace(go.Histogram(
                x=subset[cov],
                name=label_mapping.get(treatment, str(treatment)),
                opacity=0.6,
                histnorm='probability density',
                nbinsx=30,
                bingroup=1,  # Ensure histograms are layered
                marker_color=color  # Set color of histogram bars
            ))
            
            # Calculate KDE for each subset
            kde = gaussian_kde(subset[cov].dropna())
            x = np.linspace(subset[cov].min(), subset[cov].max(), 1000)
            kde_y = kde(x)
            
            # Add KDE trace with the same color
            fig.add_trace(go.Scatter(
                x=x,
                y=kde_y,
                mode='lines',
                name=f'KDE {label_mapping.get(treatment, str(treatment))}',
                line=dict(color=color, width=2),  # Set color of KDE line
                showlegend=False  # Hide KDE lines from the legend
            ))

        # Update layout
        fig.update_layout(
            title=f'Distribution of {cov}',
            xaxis_title=cov,
            yaxis_title='Density',
            barmode='overlay',  # Layer the histograms
            legend_title='Treatment',
            bargap=0.1
        )
        
        plots.append(fig)
    
    return plots


def plot_propensity_scores(df, treatment_col):
    fig = go.Figure()

    # Define labels for treatment and control
    label_mapping = {1: 'Treatment', 0: 'Control'}
    
    # Get unique treatments
    treatments = df[treatment_col].unique()
    
    # Define a color map (you can customize these colors)
    colors = ['blue', 'green']
    
    # Ensure the number of colors matches the number of treatments
    treatment_colors = {treatment: colors[i % len(colors)] for i, treatment in enumerate(treatments)}
    
    # Add a histogram trace for each treatment
    for treatment in treatments:
        subset = df[df[treatment_col] == treatment]
        color = treatment_colors[treatment]
        
        # Add histogram trace
        fig.add_trace(go.Histogram(
            x=subset['propensity_score'],
            name=label_mapping.get(treatment, str(treatment)),  # Set custom label
            opacity=0.6,
            histnorm='probability density',  # Use density
            nbinsx=30,
            bingroup=1,  # Ensures histograms are layered
            marker_color=color  # Set the color of the histogram bars
        ))
        
        # Calculate KDE for each subset
        kde = gaussian_kde(subset['propensity_score'].dropna())
        x = np.linspace(subset['propensity_score'].min(), subset['propensity_score'].max(), 1000)
        kde_y = kde(x)
        
        # Add KDE trace with the same color
        fig.add_trace(go.Scatter(
            x=x,
            y=kde_y,
            mode='lines',
            name=f'KDE {label_mapping.get(treatment, str(treatment))}',  # Set custom label for KDE
            line=dict(color=color, width=2),  # Set the color of the KDE line
            showlegend=False  # Hide KDE lines from the legend
        ))

    # Update layout
    fig.update_layout(
        title='Distribution of Propensity Scores',
        xaxis_title='Propensity Score',
        yaxis_title='Density',  # Use 'Density' on the Y-axis
        barmode='overlay',  # Layer the histograms
        legend_title='Group',
        bargap=0.1
    )
    
    return fig


def t_test_covariates(df, treatment_col, covariates_cols):
    t_test_results = {}
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    for cov in covariates_cols:
        t_stat, p_value = stats.ttest_ind(treated[cov], control[cov], equal_var=False)
        t_test_results[cov] = {'t_stat': t_stat, 'p_value': p_value}
    
    return t_test_results


@app.callback(
    Output('output-quality-error', 'children'),
    Output('output-covariate-balance', 'children'),
    Output('ttest-output', 'children'),
    Output('propensity-distribution', 'children'),
    Input('run-quality', 'n_clicks'),
    Input('url', 'pathname'),
    Input('stored-data', 'data')

)
def quality_assurance(n_clicks, pathname, stored_data):
    
    if pathname == '/quality' and n_clicks:
        
        if stored_data is None:
            return "Error: Please Run the Analysis First.", "", "", ""
        
        #Get the data
        treatment_toggle = stored_data.get('treatment_toggle')
        file_contents    = stored_data.get('contents')
        file_name        = stored_data.get('filename')
        area_id_col      = stored_data.get('area_id_col')
        treatment_col    = stored_data.get('treatment_col')
        covariate_cols   = stored_data.get('covariate-columns')
            
        if any(value is None for value in stored_data.values()):
            return "Error: Please Run the Treatment-Control Matching First.", "", "", ""
                
        if not treatment_toggle:
            return "Error: Quality Assurance Requires a Treatment and Control Group", "", "", ""
       
        if treatment_col is None:
            return "Error: Please select a treatment column.", "", "", ""
        
        if area_id_col == treatment_col:
            return "Error: The treatment column must be different from the label column.", "", "", ""
    
        
        #Parse data
        df = parse_contents(file_contents, file_name)
        
        if not set(df[treatment_col].unique()).issubset({0, 1}):
            return "Error: Treatment column must contain a binary response. 1 = Treatment // 0 = Control.", "", "", ""
        
        #Check Treated Areas True
        if len(df[df[treatment_col] == 1].index) == 0:
            return "Error: No treated areas to match.", "", "", ""
        
        def generate_match_data(df, treatment_col, covariate_cols):

            #Get subset of data
            treatment = df[treatment_col]
            covariates = df[covariate_cols]
            
            
            #Data Transform
            scaler = StandardScaler()
            covariates_scaled = scaler.fit_transform(covariates)
            
            #Propensity Scores
            logistic = LogisticRegression()
            logistic.fit(covariates_scaled, treatment)
    
            # Get the propensity scores
            df['propensity_score'] = logistic.predict_proba(covariates_scaled)[:, 1]
    
    
            #Subset data
            treated = df[df[treatment_col] == 1]
            control = df[df[treatment_col] == 0]
            
            #NN Matching
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(control[['propensity_score']])            
            
            distances, indices = nn.kneighbors(treated[['propensity_score']])
        
            matched_control = control.iloc[indices.flatten()]
            
    
            df_matched = pd.concat([treated, matched_control])
        
            return(df_matched)
    
    
        df_matched = generate_match_data(df, treatment_col, covariate_cols)
        
        
        ## BALANCE OF COVARIATES
        plots_boc = plot_covariate_distributions(df_matched, treatment_col, covariate_cols)
        
        stats_boc_b = calculate_smd(df, treatment_col, covariate_cols, 'Before')
        stats_boc_a = calculate_smd(df_matched, treatment_col, covariate_cols, 'After')
        
        
        out_covariate_balance = []
        
        for plot, stat_b, stat_a in zip(plots_boc, stats_boc_b, stats_boc_a):
        
            out_covariate_balance.append(html.Div([dcc.Graph(figure=plot),
                                                  html.Br(),
                                                  html.P(stat_b),
                                                  html.P(stat_a)], className='grid-item')) 
        
        ## Propensity Distribution
        prop_plot = plot_propensity_scores(df, treatment_col)
        
        
        ## t-TEST RESULTS
        t_test_results = t_test_covariates(df, treatment_col, covariate_cols)
        
        out_t_test = []
        
        for main_key, sub_dict in t_test_results.items():
            
            p_value = sub_dict.get('p_value')
            t_stat = sub_dict.get('t_stat')
        
            out_t_test.append(html.Div([html.H6(main_key),
                                        html.Ul([
                                            html.Li(f'p-Value: {p_value}'),
                                            html.Li(f't-Stat: {t_stat}')
                                            ])
                                        ])) 
        
        return "", out_covariate_balance, out_t_test, html.Div([dcc.Graph(figure=prop_plot)])
    
    else:
     
        return "", "", "", ""


#%% Main

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 7860)))
