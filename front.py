import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from back import load_data, clean_and_impute_data, encode_and_scale, compute_correlation, top_correlated_features
