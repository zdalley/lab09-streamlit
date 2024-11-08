# Here is some starter code to get the data:
import streamlit as st
from sklearn.datasets import fetch_openml

titanic_sklearn = fetch_openml('titanic', version = 1, as_frame = True)
titanic_df = titanic_sklearn.frame
