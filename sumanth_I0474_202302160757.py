#!/usr/bin/env python
# coding: utf-8

# ## This is your Downloaded Blueprint Notebook ##

# In[1]:


# tags to identify this iteration when submitted
# example: codex_tags = {'env': 'dev', 'region': 'USA', 'product_category': 'A'}

codex_tags = {
    'env': 'dev', 'region': 'USA', 'product_category': 'A'
}

from codex_widget_factory import utils
results_json=[]


# ### Custom Placeholder

# #### Grid Table

# In[2]:


irisDataGridTable = """
import pandas as pd
import json
from sqlalchemy import create_engine

APPLICATION_DB_HOST = "trainingserverbatch3.postgres.database.azure.com"
APPLICATION_DB_NAME = "Training_S3_DB"
APPLICATION_DB_USER = "Trainingadmin"
APPLICATION_DB_PASSWORD = "p%40ssw0rd"

def getLogger():
    import logging
    logging.basicConfig(filename="UIACLogger.log",
                        format='%(asctime)s %(message)s',
                        filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


logger = getLogger()


def read_database_data(sql_query, filename):
    logger.info(f"Read dataset file: {filename}")
    try:
        connection_uri = f"postgresql://{APPLICATION_DB_USER}:{APPLICATION_DB_PASSWORD}@{APPLICATION_DB_HOST}/{APPLICATION_DB_NAME}"
        engine = create_engine(connection_uri)
        connection = engine.connect()
        dframe = pd.read_sql_query(sql_query, con=connection)
        return dframe
    except Exception as error_msg:
        print(f"Error occured while reading data from database using query {query} and error info: {error_msg}")
    finally:
        if connection is not None:
            connection.close()
            

def get_filter_table(dframe, selected_filters):
    logger.info("Applying screen filters on the grid table dframe.")
    select_df = dframe.copy()
    for item in list(selected_filters):
        if isinstance(selected_filters[item], list):
            if 'All' not in selected_filters[item] and selected_filters[item]:
                select_df = select_df[select_df[item].isin(
                    selected_filters[item])]
        else:
            if selected_filters[item] != 'All':
                select_df = select_df[select_df[item]
                                      == selected_filters[item]]
    logger.info("Successfully applied screen filters on the grid table dframe.")
    return select_df


def generate_dynamic_table(dframe, name='Sales', grid_options={"tableSize": "small", "tableMaxHeight": "80vh", "quickSearch":True}, group_headers=[], grid="auto"):
    logger.info("Generate dynamic Grid table json from dframe")
    table_dict = {}
    table_props = {}
    table_dict.update({"grid": grid, "type": "tabularForm",
                      "noGutterBottom": True, 'name': name})
    values_dict = dframe.dropna(axis=1).to_dict("records")
    table_dict.update({"value": values_dict})
    col_def_list = []
    for col in list(dframe.columns):
        col_def_dict = {}
        col_def_dict.update({"headerName": col, "field": col})
        col_def_list.append(col_def_dict)
    table_props["groupHeaders"] = group_headers
    table_props["coldef"] = col_def_list
    table_props["gridOptions"] = grid_options
    table_dict.update({"tableprops": table_props})
    logger.info("Successfully generated dynamic Grid table json from dframe")
    return table_dict


def build_grid_table_json():
    logger.info("Preparing grid table json for Historical Screen")
    form_config = {}
    filename = "I0474_iris"
    sql_query = f"select * from {filename}"    
    dframe = read_database_data(sql_query, filename)
    # selected_filters = {"target": 'setosa'}  # for testing, please comment in production 
    dframe = get_filter_table(dframe, selected_filters)
    form_config['fields'] = [generate_dynamic_table(dframe)]
    grid_table_json = {}
    grid_table_json['form_config'] = form_config
    logger.info("Successfully prepared grid table json for Historical Screen")
    return grid_table_json

grid_table_json = build_grid_table_json()
dynamic_outputs = json.dumps(grid_table_json)

## For testing purposes, please remove later
# print(dynamic_outputs)

"""


# exec(irisDataGridTable)

# In[ ]:





# #### Grid Table Filter

# In[3]:


irisDataFilter = """
import pandas as pd
import json
from itertools import chain
from sqlalchemy import create_engine

APPLICATION_DB_HOST = "trainingserverbatch3.postgres.database.azure.com"
APPLICATION_DB_NAME = "Training_S3_DB"
APPLICATION_DB_USER = "Trainingadmin"
APPLICATION_DB_PASSWORD = "p%40ssw0rd"

def getLogger():
    import logging
    logging.basicConfig(filename="UIACLogger.log",
                        format='%(asctime)s %(message)s',
                        filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


logger = getLogger()


def read_database_data(sql_query, filename):
    logger.info(f"Read dataset file: {filename}")
    try:
        connection_uri = f"postgresql://{APPLICATION_DB_USER}:{APPLICATION_DB_PASSWORD}@{APPLICATION_DB_HOST}/{APPLICATION_DB_NAME}"
        engine = create_engine(connection_uri)
        connection = engine.connect()
        dframe = pd.read_sql_query(sql_query, con=connection)
        return dframe
    except Exception as error_msg:
        print(f"Error occured while reading data from database using query {query} and error info: {error_msg}")
    finally:
        if connection is not None:
            connection.close()


def get_response_filters(current_filter_params, df, default_values_selected, all_filters, multi_select_filters, extra_filters={}):
    logger.info("Preparing filter dictionary")
    # Usage
    # -----
    # >>> filter_df = pd.DataFrame(columns=[....])    # Optional operation
    # >>> filter_df = final_ADS.groupby(......)       # Optional operation
    # >>> default_values_selected = {}    # The default value to be selected for a filter, provide filter_name, filter_values
    # >>> all_option_filters = []         # Filters with an All option
    # >>> multi_select_filters = []       # Filters with an multi_select option
    # >>> more_filters = {}               # Extra filters, provide filter_names, filter_options
    # >>> final_dict_out = get_response_filters(current_filter_params, filter_df, default_values_selected, all_option_filters, multi_select_filters, more_filters)
    # >>> dynamic_outputs = json.dumps(final_dict_out)
    # Returns
    # -------
    # A dict object containing the filters JSON structure

    filters = list(df.columns)
    default_values_possible = {}
    for item in filters:
        default_possible = list(df[item].unique())
        if item in all_filters:
            default_possible = list(chain(['All'], default_possible))
        default_values_possible[item] = default_possible
    if extra_filters:
        filters.extend(list(extra_filters.keys()))
        default_values_possible.update(extra_filters)
    if current_filter_params:
        selected_filters = current_filter_params["selected"]
        # print(selected_filters)
        # current_filter = current_filter_params[selected_filters]
        # current_index = filters.index(current_filter)
        select_df = df.copy()
    final_dict = {}
    iter_value = 0
    data_values = []
    default_values = {}
    for item in filters:
        filter_dict = {}
        filter_dict["widget_filter_index"] = int(iter_value)
        filter_dict["widget_filter_function"] = False
        filter_dict["widget_filter_function_parameter"] = False
        filter_dict["widget_filter_hierarchy_key"] = False
        filter_dict["widget_filter_isall"] = True if item in all_filters else False
        filter_dict["widget_filter_multiselect"] = True if item in multi_select_filters else False
        filter_dict["widget_tag_key"] = str(item)
        filter_dict["widget_tag_label"] = str(item)
        filter_dict["widget_tag_input_type"] = "select",
        filter_dict["widget_filter_dynamic"] = True
        if current_filter_params:
            if item in df.columns:
                possible_values = list(select_df[item].unique())
                item_default_value = selected_filters[item]
                if item in all_filters:
                    possible_values = list(chain(['All'], possible_values))
                if item in multi_select_filters:
                    for value in selected_filters[item]:
                        if value not in possible_values:
                            if possible_values[0] == "All":
                                item_default_value = possible_values
                            else:
                                item_default_value = [possible_values[0]]
                else:
                    if selected_filters[item] not in possible_values:
                        item_default_value = possible_values[0]
                filter_dict["widget_tag_value"] = possible_values
                if item in multi_select_filters:
                    if 'All' not in item_default_value and selected_filters[item]:
                        select_df = select_df[select_df[item].isin(
                            item_default_value)]
                else:
                    if selected_filters[item] != 'All':
                        select_df = select_df[select_df[item]
                                              == item_default_value]
            else:
                filter_dict["widget_tag_value"] = extra_filters[item]
        else:
            filter_dict["widget_tag_value"] = default_values_possible[item]
            item_default_value = default_values_selected[item]
        data_values.append(filter_dict)
        default_values[item] = item_default_value
        iter_value = iter_value + 1
    final_dict["dataValues"] = data_values
    final_dict["defaultValues"] = default_values
    logger.info("Successfully prepared filter dictionary")
    return final_dict


def prepare_filter_json():
    logger.info(f"Preparing json for Filters in Iris Grid Table")
    # Prepare Filter json for Target in the Iris Grid Table.
    filename = "I0474_iris"
    sql_query = f"select * from {filename}"    
    dframe = read_database_data(sql_query, filename)
    dframe = dframe.groupby(['target']).sum().reset_index()
    filter_dframe = dframe[['target']]
    default_values_selected = {'target': 'setosa'}
    all_filters = []
    multi_select_filters = []
    # current_filter_params = {"selected": default_values_selected}
    final_dict_out = get_response_filters(
        current_filter_params, filter_dframe, default_values_selected, all_filters, multi_select_filters)
    logger.info(f"Successful prepared json for Filters in Iris Data Grid Table")
    return json.dumps(final_dict_out)


dynamic_outputs = prepare_filter_json()
# print(dynamic_outputs)
"""

exec(irisDataFilter)
# In[ ]:





# In[ ]:





# #### Plot - 1:  Scatterplot Matrix

# In[4]:


#BEGIN CUSTOM CODE BELOW...

#put your output in this response param for connecting to downstream widgets
irisScatterplotMatrix = """
# Below codestring is used to perform detailed analysis on quantity of sales done over time. 

import plotly.express as px
import pandas as pd
import json
import plotly.io as io
from sqlalchemy import create_engine

APPLICATION_DB_HOST = "trainingserverbatch3.postgres.database.azure.com"
APPLICATION_DB_NAME = "Training_S3_DB"
APPLICATION_DB_USER = "Trainingadmin"
APPLICATION_DB_PASSWORD = "p%40ssw0rd"


def getLogger():
    import logging
    logging.basicConfig(filename="UIACLogger.log",
                        format='%(asctime)s %(message)s',
                        filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


logger = getLogger()


def read_database_data(sql_query, filename):
    logger.info(f"Read dataset file: {filename}")
    try:
        connection_uri = f"postgresql://{APPLICATION_DB_USER}:{APPLICATION_DB_PASSWORD}@{APPLICATION_DB_HOST}/{APPLICATION_DB_NAME}"
        engine = create_engine(connection_uri)
        connection = engine.connect()
        dframe = pd.read_sql_query(sql_query, con=connection)
        return dframe
    except Exception as error_msg:
        print(f"Error occured while reading data from database using query {query} and error info: {error_msg}")
    finally:
        if connection is not None:
            connection.close()


def getGraph(dframe, filters):
    logger.info(
        "Preparing bar graph json to understand products sales over time")
    for item in filters:
        if 'All' in filters[item]:
            continue
        elif isinstance(filters[item], list):
            dframe = dframe[dframe[item].isin(filters[item])]
        else:
            dframe = dframe[dframe[item] == filters[item]]
    
    fig = px.scatter(dframe, x="sepal width (cm)", y="sepal length (cm)", color="target",
                     size='petal length (cm)', hover_data=['petal width (cm)'])
    # fig.show()
    
    logger.info(
        "Successfully prepared bar graph json to understand products sales over time")
    return io.to_json(fig)


# selected_filters = {"target": 'versicolor'}

filename = "I0474_iris"
sql_query = f"select * from {filename}"    
dframe = read_database_data(sql_query, filename)
dynamic_outputs = getGraph(dframe, selected_filters)

"""
#END CUSTOM CODE


# exec(irisScatterplotMatrix)

# #### Color Table

# In[5]:


conversion_func_code_str = '''
from plotly.io import to_json
def get_grid_table(df,
                   col_props={},
                   grid_options={"tableSize": "small", "tableMaxHeight": "80vh"},
                   group_headers=[],
                   popups={},
                   popup_col=[],
                   popup_col_props={},
                   popup_grid_options={},
                   popup_group_headers={}
                   ):
    """to get customized table
    Args:
        df (pandas dataframe): input data
        col_props (dict, optional): used to specify properties of a column. Defaults to {}.
                                        { 'country': {"sticky": True}, 'population': {'sortable': True, 'width' : '200px'}}.
        grid_options (dict, optional): used to specify the properties of the table. Defaults to {"tableSize": "small", "tableMaxHeight": "20vh"}.
        group_headers (list, optional): used to group multiples columns and specify property to it. Defaults to [].
        popups (dict, optional): used to specify the data/figure for popup screen table with respect to each value of popup column.
                                    ex: if popup column has values ['India', 'USA', 'Germany']
                                        popup_dfs = {'India':dataframe1, 'USA':dataframe2, 'Germany': dataframe3}
                                        Note: popup column should have unique values. Defaults to {}.
        popup_col (list, optional): used to define the column name which acts as trigger to popup screen. Defaults to [].
        popup_col_props (dict, optional): used to specify properties of a column present in popup screen table. Defaults to {}.
                                        { 'country': {"sticky": True}, 'population': {'sortable': True, 'width' : '200px'}}
        popup_grid_options (dict, optional): used to specify the properties of the popup screen table. Defaults to {}.
        popup_group_headers (dict, optional): used to group multiples columns and specify property of popup dataframe. Defaults to [].
    Returns:
        _type_: _description_
    """
    comp_dict = {}
    comp_dict['is_grid_table'] = True
    comp_props_dict = {}
    actual_columns = df.columns[~ ((df.columns.str.contains("_bgcolor")) | (df.columns.str.contains("_color")))]
    bg_color_columns = df.columns[df.columns.str.contains("_bgcolor")]
    color_columns = df.columns[df.columns.str.contains("_color")]
    values_dict = df[actual_columns].to_dict("records")
    row_props_list = []
    for index, row_values in enumerate(values_dict):
        row_props_dict = {}
        for col_name, row_value in row_values.items():
            row_props_dict[col_name] = row_value
            if (col_name in popup_col) or (col_name + '_bgcolor' in bg_color_columns) or (col_name + '_color' in color_columns):
                row_props_dict[col_name + '_params'] = {}
                if col_name in popup_col:
                    insights_grid_options = popup_grid_options.copy()
                    insights_grid_options.update({"tableTitle": row_value})
                    if isinstance(popups[col_name][row_value], pd.DataFrame):
                        row_props_dict[col_name + '_params'].update({"insights": {
                            "data": get_grid_table(popups[col_name][row_value], col_props=popup_col_props[col_name], grid_options=insights_grid_options[col_name],
                                                   group_headers=popup_group_headers[col_name])
                        }})
                    elif isinstance(popups[col_name][row_value], go.Figure):
                        row_props_dict[col_name + '_params'].update({"insights": {
                            "data": json.loads(to_json(popups[col_name][row_value]))
                        }})
                if col_name + '_bgcolor' in bg_color_columns:
                    row_props_dict[col_name + '_params'].update({'bgColor': df.iloc[index].to_dict()[col_name + '_bgcolor']})
                if col_name + '_color' in color_columns:
                    row_props_dict[col_name + '_params'].update({'color': df.iloc[index].to_dict()[col_name + '_color']})
        row_props_list.append(row_props_dict)
    col_props_list = []
    for col in actual_columns:
        col_props_dict = {}
        col_props_dict.update({"headerName": col, "field": col, 'cellParamsField': col + '_params'})
        if col in popup_col:
            col_props_dict.update({"enableCellInsights": True})
        if col in col_props:
            col_props_dict.update(col_props[col])
        col_props_list.append(col_props_dict)
    comp_props_dict['rowData'] = row_props_list
    comp_props_dict["coldef"] = col_props_list
    comp_props_dict["gridOptions"] = grid_options
    if group_headers:
        comp_props_dict['groupHeaders'] = group_headers
    comp_dict.update({"tableProps": comp_props_dict})
    return comp_dict
'''


# In[6]:


code_str = '''
from sklearn.datasets import load_iris
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine

APPLICATION_DB_HOST = "trainingserverbatch3.postgres.database.azure.com"
APPLICATION_DB_NAME = "Training_S3_DB"
APPLICATION_DB_USER = "Trainingadmin"
APPLICATION_DB_PASSWORD = "p%40ssw0rd"

def getLogger():
    import logging
    logging.basicConfig(filename="UIACLogger.log",
                        format='%(asctime)s %(message)s',
                        filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


logger = getLogger()

def read_database_data(sql_query, filename):
    logger.info(f"Read dataset file: {filename}")
    try:
        connection_uri = f"postgresql://{APPLICATION_DB_USER}:{APPLICATION_DB_PASSWORD}@{APPLICATION_DB_HOST}/{APPLICATION_DB_NAME}"
        engine = create_engine(connection_uri)
        connection = engine.connect()
        dframe = pd.read_sql_query(sql_query, con=connection)
        return dframe
    except Exception as error_msg:
        print(f"Error occured while reading data from database using query {query} and error info: {error_msg}")
    finally:
        if connection is not None:
            connection.close()


filename = "I0474_iris"
sql_query = f"select * from {filename}"    
main_df = read_database_data(sql_query, filename)
main_df['sepal length (cm)_bgcolor'] = main_df['sepal length (cm)'].apply(lambda x: "#0000FF" if x>5 else "#FF0000")
main_df['sepal width (cm)_bgcolor'] = '#FF00FF'
main_df['petal length (cm)_bgcolor'] = '#00FFFF'
main_df['petal width (cm)_bgcolor'] = '#800000'
main_df['target_bgcolor'] = '#00FF00'



container_dict = {}
container_dict = get_grid_table(main_df)
dynamic_outputs = json.dumps(container_dict)
print(dynamic_outputs)
'''

irisColorTable = conversion_func_code_str + code_str


# #### Graph: y = x^2

# In[7]:


#BEGIN CUSTOM CODE BELOW...

#put your output in this response param for connecting to downstream widgets

quadraticGraph = """
# Below codestring is used to perform detailed analysis on quantity of sales done over time. 

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import plotly.io as io


def getLogger():
    import logging
    logging.basicConfig(filename="UIACLogger.log",
                        format='%(asctime)s %(message)s',
                        filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


logger = getLogger()


def plotGraph():
    x = np.arange(10)
    fig = go.Figure(data=go.Scatter(x=x, y=x**2))
     
    # fig.show()
    
    logger.info(
        "Successfully prepared x=y^2")
    return io.to_json(fig)


selected_filters = {"target": 'versicolor'}

dynamic_outputs = plotGraph()

"""
#END CUSTOM CODE


# exec(irisGraph)

# #### Multi Trace Plots

# In[82]:


#BEGIN CUSTOM CODE BELOW...

#put your output in this response param for connecting to downstream widgets
irisMultiTracePlots = """
# Below codestring is used to perform detailed analysis on quantity of sales done over time. 

import plotly.graph_objects as go
import pandas as pd
import json
import plotly.io as io
from sqlalchemy import create_engine

APPLICATION_DB_HOST = "trainingserverbatch3.postgres.database.azure.com"
APPLICATION_DB_NAME = "Training_S3_DB"
APPLICATION_DB_USER = "Trainingadmin"
APPLICATION_DB_PASSWORD = "p%40ssw0rd"


def getLogger():
    import logging
    logging.basicConfig(filename="UIACLogger.log",
                        format='%(asctime)s %(message)s',
                        filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


logger = getLogger()


def read_database_data(sql_query, filename):
    logger.info(f"Read dataset file: {filename}")
    try:
        connection_uri = f"postgresql://{APPLICATION_DB_USER}:{APPLICATION_DB_PASSWORD}@{APPLICATION_DB_HOST}/{APPLICATION_DB_NAME}"
        engine = create_engine(connection_uri)
        connection = engine.connect()
        dframe = pd.read_sql_query(sql_query, con=connection)
        return dframe
    except Exception as error_msg:
        print(f"Error occured while reading data from database using query {query} and error info: {error_msg}")
    finally:
        if connection is not None:
            connection.close()


def getGraph(dframe, filters):
    logger.info(
        "Preparing bar graph json to understand products sales over time")
    for item in filters:
        if 'All' in filters[item]:
            continue
        elif isinstance(filters[item], list):
            dframe = dframe[dframe[item].isin(filters[item])]
        else:
            dframe = dframe[dframe[item] == filters[item]]
    
    
    fig = go.Figure(data=[go.Scatter(
     x=dframe['sepal width (cm)'],
     y=dframe['sepal length (cm)'],
     mode='markers',
     marker=dict(
         color='blue',
     ))])

    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=["y", "petal length (cm)"],
                        label="Petal length",
                        method="restyle"
                    ),
                    dict(
                        args=["y", "petal width (cm)"],
                        label="Petal Width",
                        method="restyle"
                    )
                ]),
                direction="down",
            ),
        ]
    )

    # fig.show()
    
    logger.info(
        "Successfully prepared bar graph json to understand products sales over time")
    return io.to_json(fig)


# selected_filters = {"target": 'versicolor'}

filename = "I0474_iris"
sql_query = f"select * from {filename}"    
dframe = read_database_data(sql_query, filename)
dynamic_outputs = getGraph(dframe, selected_filters)

"""
#END CUSTOM CODE


# exec(irisMultiTracePlots)

# In[8]:


# dynamic_metrics = {
#     'Total Units Returned': productReturn_quantity,
#     'Total Unique Product Quantity Returned': productReturn_UniqueProduct,
#     'Total Cost of Product Returned': productReturns_costOfProductReturned,
# }

dynamic_result = {
    'Iris Scatterplot': irisScatterplotMatrix,
    'Iris Dataset': irisDataGridTable,
    'Iris Color Table': irisColorTable,
    'Quadratic Graph': quadraticGraph,
    'Iris Multitrace Plots': irisMultiTracePlots,
    
}

dynamic_filter = {
    'irisDataFilter': irisDataFilter,
}

# dynamic_actions = {
#     "breadcrumb": {
#         "default": None,
#         "action_generator": historicScreenActionGen,
#         "action_handler": historicScreenActionHandler
#     },
# }

results_json.append({
    'type': 'Test Codx App',
    'name': 'Test Codx App',
    'component': 'Test Codx App',
    'dynamic_visual_results': dynamic_result,
    'dynamic_code_filters': dynamic_filter,
#     'dynamic_metrics_results': dynamic_metrics,
#     'actions': dynamic_actions

})


# ### Please save and checkpoint notebook before submitting params

# In[9]:


currentNotebook = 'sumanth_I0474_202302160757.ipynb'

get_ipython().system('jupyter nbconvert --to script {currentNotebook}')


# In[10]:


utils.submit_config_params(url='https://codex-api-stage.azurewebsites.net/codex-api/projects/upload-config-params/rqdKcDtDfAyzE3p7lA73Sg', nb_name=currentNotebook, results=results_json, codex_tags=codex_tags, args={})


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




