import streamlit as st
import yaml
import pathlib
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path


def success_function(msg):
    st.success("Upload "+msg+" successful")
st.markdown("## Upload Config and Trace Files")

config = st.file_uploader("Upload a config file")

if config is not None:
    loaded_config = yaml.safe_load(config)
    data = config.getvalue().decode('utf-8')
    scale_filename = "./scale_profile.yaml"
    path_scale = Path(scale_filename)
    if path_scale.is_file() is False:
        st.session_state["upload_state2"] = "No file name present. Please change the filename to scale_profile.yaml"
        st.error("No file name present. Please change the filename")
    else:
        scale_profile_file = open(scale_filename, "w")
        scale_profile_file.write(data)
        scale_profile_file.close()
        #upload_state = st.text(config.name + " " + "Uploaded successfully.")
        #upload_state = st.text("Please refresh or continue to go to the other page for the changes to reflect")
        st.success("Upload successful.")
        #success_function(scale_filename)
   # st.write(st.session_state["upload_state2"])


st.markdown("###### Design #1")

trace = st.file_uploader("Upload trace csv files1")
def upload():
    st.empty()
    if trace is None:
        st.session_state["upload_state"] = "Upload a file first!"
    else:
        data_trace = trace.getvalue().decode('utf-8')
        filename = "./traces/"+trace.name
        #check if the file exists
        path = Path(filename)
        if path.is_file() is False:
            st.session_state["upload_state"] = "no file name present. Please change the filename"
            st.error("No file name present. Please change the filename")
        else:
            scale_trace_file = open(filename, "w")
            scale_trace_file.write(data_trace)
            scale_trace_file.close()
            st.session_state["upload_state"] = "Saved successfully"
            st.success("Upload successful.")
            #success_function(filename)
    #st.write(st.session_state["upload_state"])
st.button("Upload file to Sandbox", on_click=upload)


st.markdown("###### Design #2")

trace1 = st.file_uploader("Upload trace csv files2")

def upload1(file_location):
    if trace1 is None:
        st.session_state["upload_state1"] = "Upload a file first!"
    else:
        data_trace1 = trace1.getvalue().decode('utf-8')
        scale_trace_file1 = open(file_location, "w")
        scale_trace_file1.write(data_trace1)
        scale_trace_file1.close()
        st.session_state["upload_state1"] = "Saved successfully"
        st.success("Uploaded successfully")


if trace1 is not None:
    parent_path = pathlib.Path(__file__).parent.parent.resolve()
    data_path = os.path.join(parent_path, "traces")
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    option = st.selectbox('Pick a dataset', onlyfiles)
    file_location=os.path.join(data_path, option)
    st.button("Upload file to Sandbox 1", on_click=upload1, args = (file_location,))
