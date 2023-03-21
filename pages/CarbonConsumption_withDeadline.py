#!/usr/bin/env python3
"""
    Created date: 9/12/22
"""

import os
import pandas as pd
import numpy as np
import yaml
import datetime
import streamlit as st

import environment
import agent
import eval_util

import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from dateutil import parser
from glob import glob


def get_datetime(iso_str):
    dt = parser.parse(iso_str)
    dt = datetime.datetime.combine(dt.date(), datetime.time(hour=dt.hour))
    return dt

# Constants
ds_size_map = {
    "tinyimagenet": 100000,
    "imagenet": 1281167,
}

cpu_power_offset = 50


carbon_traces_path = sorted(glob("traces/*.csv"))
carbon_trace_names = [os.path.basename(trace_name) for trace_name in carbon_traces_path]
carbon_trace_names = [os.path.splitext(trace_name)[0] for trace_name in carbon_trace_names]
carbon_trace_map = {trace_name: trace_path for trace_name, trace_path in zip(carbon_trace_names, carbon_traces_path)}

profile_path = "scale_profile.yaml"
with open(profile_path, 'r') as f:
    task_profile = yaml.safe_load(f)

st.sidebar.markdown("### Policy Model")

selected_trace = st.sidebar.selectbox("Carbon Trace", options=carbon_trace_names)
carbon_trace = pd.read_csv(carbon_trace_map[selected_trace])
carbon_trace = carbon_trace[carbon_trace["carbon_intensity_avg"].notna()]
carbon_trace["hour"] = carbon_trace["zone_datetime"].apply(lambda x: parser.parse(x).hour)
carbon_trace["datetime"] = carbon_trace["zone_datetime"].apply(get_datetime)
carbon_trace["date"] = carbon_trace["zone_datetime"].apply(lambda x: parser.parse(x).date())
# selected_locations = st.sidebar.multiselect("Locations", carbon_trace_names, default=["AUS-NSW"])


selected_task = st.sidebar.selectbox("Task", options=task_profile.keys())
input_task_length = st.sidebar.number_input("Task Length (hour)", min_value=1, value=24)
#input_deadline = st.sidebar.number_input("Deadline", min_value=input_task_length, value=input_task_length)
input_max_workers = st.sidebar.number_input("Max Workers", min_value=1, max_value=8, value=8)
input_started_date = st.sidebar.date_input("Started Date", min_value=carbon_trace["date"].min(),
                                           max_value=carbon_trace["date"].max(), value=carbon_trace["date"].min())
started_datetime_df = carbon_trace[carbon_trace["date"] == input_started_date]

input_started_hour = st.sidebar.number_input("Started Hour", min_value=started_datetime_df["hour"].min(),
                                             max_value=started_datetime_df["hour"].max(),
                                             value=started_datetime_df["hour"].min())

started_hour_time = datetime.time(hour=input_started_hour)
started_datetime = datetime.datetime.combine(input_started_date, started_hour_time)

started_index = carbon_trace.index[carbon_trace["datetime"] == started_datetime][0]

st.markdown("## Carbon Footprint Analyzer for ML Tasks")

# simulation
model_profile = task_profile[selected_task]
dataset = model_profile["dataset"]
ds_size = ds_size_map[dataset]
num_profile = max(model_profile["replicas"])

tp_table = np.zeros(num_profile+1)
energy_table = np.zeros_like(tp_table)

for num_workers, profile in model_profile["replicas"].items():
    tp_table[num_workers] = profile["throughput"]
    energy_table[num_workers] = profile["gpuPower"] + (cpu_power_offset * num_workers)  # Add CPU energy offset

tp_table = tp_table / ds_size  # to epochs per hour
energy_table = energy_table * 3600. / 3.6e+6   # to Kwh per hour
num_epochs = tp_table[1] * input_task_length

reward = environment.NonLinearReward(tp_table, energy_table)
###################################################################################################################
# Carbon Consumption (g) across Carbon Scalar, Carbon Agnostic, & While A While policies for selected locations
st.markdown("Effect of completion time on the carbon footprint")
new_tp_fig = go.Figure()
list_of_completion_times = [input_task_length, int(1.5*input_task_length), 2*input_task_length, int(2.5*input_task_length), 3*input_task_length, int(3.5*input_task_length)]
#list_of_completion_times = [int(1.5*input_task_length), int(1.5*input_task_length), int(1.5*input_task_length), int(1.5*input_task_length), int(1.5*input_task_length), int(1.5*input_task_length)]
scale_table = np.zeros(6)
#print(list_of_completion_times)
count = 0
arr = np.zeros((6,2))
for len in list_of_completion_times:
    num_epochs_t = tp_table[1] * input_task_length

    reward_t = environment.NonLinearReward(tp_table, energy_table)

    # Carbon scale method
    env_t = environment.CarbonOnlyEnvironment(carbon_trace["carbon_intensity_avg"].values,
                                            reward_t, np.array([started_index]), num_epochs_t)
    carbon_scale_agent_t = agent.CarbonScaleAgent(tp_table, energy_table, input_max_workers, len)
    carbon_cost_scale_t, carbon_scale_states_t, carbon_scale_action_t, exec_time = \
        eval_util.simulate_agent(carbon_scale_agent_t, env_t, len)
    carbon_scale_action_t = carbon_scale_action_t.flatten()


    scale_table[count] = carbon_cost_scale_t[0]
    #print(carbon_cost_scale_t, carbon_cost_scale_t[0])
    arr[count] = [len, scale_table[count]]
    count = count + 1

df = pd.DataFrame(arr, columns=['length', 'scale'])

sched_fig_t = make_subplots(specs=[[{"secondary_y": True}]])
sched_fig_t.add_trace(
    go.Scatter(x=df["length"],
               y=df["scale"], 
               mode="lines+markers", name="Scale", 
               hovertemplate="%{x}<br>%{y:.2f} g"),
    secondary_y=False
)
sched_fig_t.update_yaxes( title_text="Carbon Consumption (g)", secondary_y=False)
sched_fig_t.update_xaxes(title_text="Completion Time(hours)")
st.plotly_chart(sched_fig_t)