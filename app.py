import io

import json

import os

import time

import base64

import tempfile

import zipfile



import numpy as np

import pandas as pd

import streamlit as st

import torch

from PIL import Image



from config import CLASS_NAMES, DEVICE

from modules.model_loader import load_model as load_model_from_checkpoint

from modules.detection import detect, annotate

from modules.video_processing import process_video

from modules.inspection_history import (

    initialize_database,

    save_inspection,

    list_inspections,

    get_inspection,

    delete_inspection,

)

from modules.report_generator import generate_inspection_pdf
from modules.asset_center import (initialize_asset_tables, list_assets, add_asset, delete_asset, list_tasks, add_task, update_task_status, asset_summary)



# Create the SQLite tables if they do not already exist.

initialize_database()
initialize_asset_tables()





# ============================================================

# PAGE

# ============================================================



st.set_page_config(

    page_title="PowerLine_AI",

    page_icon="⚡",

    layout="wide",

    initial_sidebar_state="expanded",

)





# ============================================================

# PATHS / DATA

# ============================================================



MODEL_PATH = r"D:\python\DLproject_checkpoints\checkpoint_epoch_10.pth"



# Put powerline_hero_bg.jpg beside app.py for the hero image.

HERO_IMAGE = os.path.join(

    os.path.dirname(os.path.abspath(__file__)),

    "powerline_hero_bg.jpg",

)





MAP50 = 0.8290

MAP5095 = 0.6331

MAP75 = 0.6669

MAR100 = 0.6873

VALIDATION_IMAGES = 2626



TRAIN_LOSSES = [

    0.2521, 0.166758, 0.140621, 0.090420, 0.076993,

    0.067844, 0.058481, 0.056423, 0.055148, 0.053617

]



PER_CLASS_MAP = {

    "Yoke": 0.8456,

    "Yoke Suspension": 0.8254,

    "Spacer": 0.3105,

    "Stockbridge Damper": 0.8150,

    "Lightning Rod Shackle": 0.4437,

    "Lightning Rod Suspension": 0.8854,

    "Polymer Insulator": 0.9016,

    "Glass Insulator": 0.8076,

    "Tower ID Plate": 0.9325,

    "Vari-Grip": 0.9012,

    "Polymer Insulator Lower Shackle": 0.5633,

    "Polymer Insulator Upper Shackle": 0.7941,

    "Polymer Insulator Tower Shackle": 0.3913,

    "Glass Insulator Big Shackle": 0.0727,

    "Glass Insulator Small Shackle": 0.1831,

    "Glass Insulator Tower Shackle": 0.2143,

    "Spiral Damper": 0.8755,

}







# ============================================================

# CSS

# ============================================================



st.markdown(

    """

<style>



.stApp {

    background: #07111f;

    color: #f4f7fb;

}



[data-testid="stHeader"] {

    background: #07111f;

}



.block-container {

    max-width: 1500px;

    padding: 16px 28px 35px 28px;

}



section[data-testid="stSidebar"] {

    background: linear-gradient(180deg,#091522 0%,#06111d 100%);

    border-right: 1px solid rgba(130,170,210,.12);

}



section[data-testid="stSidebar"] > div {

    padding: 18px 16px;

}



.brand {

    padding: 12px 10px 22px;

    border-bottom: 1px solid rgba(255,255,255,.08);

    margin-bottom: 20px;

}



.brand-title {

    font-size: 23px;

    font-weight: 850;

    color: #fff;

}



.brand-title span {

    color: #1598ff;

}



.brand-subtitle {

    color: #8298af;

    font-size: 11px;

    margin-top: 5px;

}



.nav-label {

    color: #6f849b;

    font-size: 9px;

    font-weight: 800;

    letter-spacing: 1.8px;

    margin: 15px 5px 8px;

}



div[role="radiogroup"] {

    gap: 7px;

}



div[role="radiogroup"] label {

    border-radius: 9px;

    padding: 6px 8px;

    border: 1px solid transparent;

}



div[role="radiogroup"] label:hover {

    background: rgba(20,135,255,.10);

}



div[role="radiogroup"] label[data-checked="true"] {

    background: linear-gradient(90deg,#087ff5,#1268dc);

    border-color: #218fff;

}



div[role="radiogroup"] label p {

    color: #dce8f5 !important;

    font-size: 13px;

    font-weight: 650;

}



.hero {

    min-height: 235px;

    border-radius: 17px;

    overflow: hidden;

    position: relative;

    padding: 34px 28px;

    background:

        linear-gradient(90deg,

            rgba(4,16,31,.99) 0%,

            rgba(5,25,45,.94) 45%,

            rgba(4,27,51,.55) 100%),

        var(--hero-bg);

    background-size: cover;

    background-position: center;

    border: 1px solid rgba(62,143,220,.15);

}



.hero-content {

    position: relative;

    z-index: 2;

    max-width: 690px;

}



.hero-kicker {

    color: #61baff;

    font-size: 10px;

    font-weight: 850;

    letter-spacing: 1.7px;

}



.hero-title {

    margin: 4px 0 0;

    font-size: 43px;

    line-height: 1;

    font-weight: 900;

    letter-spacing: -1.7px;

}



.hero-title span {

    color: #1699ff;

}



.hero-heading {

    margin-top: 10px;

    font-size: 20px;

    font-weight: 800;

    color: #f2f6fb;

}



.hero-copy {

    color: #c2d0df;

    font-size: 14px;

    line-height: 1.55;

    margin-top: 10px;

    max-width: 650px;

}



.pills {

    display: flex;

    gap: 24px;

    margin-top: 22px;

    flex-wrap: wrap;

}



.pill {

    color: #dce8f5;

    font-size: 12px;

    font-weight: 650;

}



.pill b {

    color: #1598ff;

    font-size: 19px;

    vertical-align: middle;

    margin-right: 5px;

}



.hero-right {

    position: absolute;

    right: 24px;

    top: 27px;

    z-index: 2;

    text-align: right;

    color: #edf6ff;

    font-size: 13px;

    font-weight: 750;

    line-height: 1.5;

}



.hero-right-line {

    width: 30px;

    height: 2px;

    background: #1b9bff;

    margin: 10px 0 0 auto;

}



.section-title {

    font-size: 18px;

    font-weight: 800;

    margin: 17px 0 11px;

}



.metric {

    min-height: 91px;

    padding: 14px 17px;

    border-radius: 13px;

    background: linear-gradient(145deg,#10233a,#091827);

    border: 1px solid rgba(90,155,215,.11);

    display: flex;

    align-items: center;

    gap: 13px;

}



.metric-icon {

    width: 43px;

    height: 43px;

    border-radius: 10px;

    background: rgba(30,140,255,.15);

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 21px;

}



.metric-value {

    font-size: 23px;

    font-weight: 900;

    color: #fff;

}



.metric-label {

    color: #8ea4ba;

    font-size: 12px;

    margin-top: 2px;

}



.panel {

    border-radius: 14px;

    background: linear-gradient(145deg,#0e2033,#081725);

    border: 1px solid rgba(95,155,215,.11);

    padding: 17px;

    height: 100%;

}



.panel-title {

    color: #f5f8fc;

    font-size: 16px;

    font-weight: 800;

}



.panel-subtitle {

    color: #8197ad;

    font-size: 12px;

    margin: 6px 0 12px;

}



.upload-box {

    border: 1px dashed #218fff;

    border-radius: 10px;

    padding: 22px;

    text-align: center;

    background: rgba(6,25,43,.75);

    color: #cbd9e8;

}



.result-ok {

    background: rgba(11,180,125,.13);

    border: 1px solid rgba(11,180,125,.25);

    border-radius: 9px;

    padding: 11px 13px;

    color: #6fe0b7;

    font-size: 12px;

    font-weight: 700;

    margin-top: 10px;

}



.quote-card {

    margin-top: 18px;

    padding: 22px 12px;

    text-align: center;

    border: 1px solid rgba(100,155,205,.10);

    background: rgba(15,31,48,.7);

    border-radius: 12px;

    color: #aabacc;

    font-size: 12px;

    font-style: italic;

}



.stButton > button {

    width: 100%;

    border-radius: 9px;

    min-height: 42px;

    background: linear-gradient(90deg,#087ff5,#1269dc);

    color: white;

    border: 1px solid #2496ff;

    font-weight: 800;

}



.stDownloadButton > button {

    width: 100%;

    border-radius: 9px;

}



[data-testid="stFileUploader"] {

    background: transparent;

}



[data-testid="stFileUploaderDropzone"] {

    background: rgba(6,25,43,.8);

    border: 1px dashed rgba(33,143,255,.55);

    border-radius: 10px;

}



[data-testid="stFileUploaderDropzone"] * {

    color: #cad8e6 !important;

}



.footer {

    color: #61768c;

    font-size: 10px;

    margin-top: 20px;

}




/* ===== PowerLine AI premium interface layer ===== */
:root { color-scheme: dark; }
html, body, [class*="css"] { font-family: Inter, "Segoe UI", sans-serif; }
.stApp { background: radial-gradient(circle at 80% 0%, #102b48 0, #071321 36%, #060f1b 100%); color: #e8f1fb; }
[data-testid="stHeader"] { background: rgba(6,15,27,.88); }
.block-container { max-width: 1540px; padding: 2rem 2.2rem 3rem; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0b1b2c 0%,#071321 100%); border-right: 1px solid #1c3550; }
section[data-testid="stSidebar"] > div { padding: 1.35rem 1rem; }
.brand { padding: 1.1rem .7rem 1.5rem; margin-bottom: 1.2rem; }
.brand-title { letter-spacing: -.7px; font-size: 1.45rem; }
.brand-subtitle { font-size: .72rem; letter-spacing: .2px; }
.nav-label { margin: 1.4rem .35rem .7rem; color: #7f9bb8; }
div[role="radiogroup"] { gap: .42rem; }
div[role="radiogroup"] label { padding: .55rem .65rem; border-radius: 10px; transition: all .18s ease; }
div[role="radiogroup"] label[data-checked="true"] { background: linear-gradient(100deg,#087ff5,#0756b8); box-shadow: 0 5px 18px rgba(0,111,230,.2); }
.section-title { color: #f3f7fc; font-size: 1.15rem; font-weight: 800; letter-spacing: -.25px; margin: 1.4rem 0 .9rem; }
.hero { border: 1px solid #203c59; border-radius: 18px; background: linear-gradient(115deg,rgba(13,36,59,.96),rgba(8,22,37,.94)); box-shadow: 0 14px 36px rgba(0,0,0,.16); padding: 2rem 2.1rem; margin-bottom: 1.2rem; }
.hero-title { letter-spacing: -1.2px; }
.panel, .metric { border: 1px solid #203b56; border-radius: 14px; background: linear-gradient(145deg,rgba(16,38,60,.96),rgba(10,26,42,.96)); box-shadow: 0 8px 24px rgba(0,0,0,.12); }
.panel { padding: 1.05rem 1.1rem; margin-bottom: .7rem; }
.panel-title { color: #edf5ff; font-weight: 800; font-size: .96rem; }
.panel-subtitle { color: #8eabc8; font-size: .76rem; margin-top: .35rem; }
.metric { min-height: 96px; padding: 1.1rem; display: flex; align-items: center; gap: .85rem; }
.metric-value { color: #fff; font-size: 1.55rem; font-weight: 850; letter-spacing: -.5px; }
.metric-label { color: #91abc6; font-size: .75rem; margin-top: .2rem; }
[data-testid="stMetric"] { background: linear-gradient(145deg,#102943,#0b1d30); border: 1px solid #203b56; border-radius: 13px; padding: 1rem 1.1rem; }
[data-testid="stMetricLabel"] p { color: #9bb4ce !important; font-size: .78rem; }
[data-testid="stMetricValue"] { color: #f5f9ff !important; font-weight: 800; }
.stButton > button, .stDownloadButton > button { border-radius: 10px; min-height: 2.65rem; font-weight: 750; border: 1px solid #2389ed; background: linear-gradient(100deg,#087ff5,#075ec5); box-shadow: 0 5px 14px rgba(0,100,220,.15); transition: transform .15s ease, filter .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { filter: brightness(1.12); transform: translateY(-1px); border-color: #55adff; }
[data-testid="stFileUploaderDropzone"] { background: rgba(12,34,55,.72); border: 1px dashed #277fc8; border-radius: 13px; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: #56b3ff; background: rgba(15,47,75,.8); }
[data-testid="stFileUploaderDropzone"] * { color: #c7d9eb !important; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid #243f5a; border-radius: 12px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 11px; }
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stMultiSelect div[data-baseweb="select"], .stTextArea textarea { border-radius: 9px; }
hr { border-color: rgba(130,170,210,.14); }
.footer { color: #829bb4; font-size: .75rem; margin-top: 2.2rem; padding: 1rem 0 .2rem; border-top: 1px solid rgba(130,170,210,.12); text-align: center; }
@media (max-width: 900px) { .block-container { padding: 1.1rem 1rem 2rem; } .hero { padding: 1.35rem; } .metric-value { font-size: 1.25rem; } }

/* ===== Enterprise industrial platform refinement ===== */
:root { color-scheme: dark; }
.stApp {
  background: radial-gradient(ellipse at 78% -12%, rgba(20,112,190,.18), transparent 36%),
    linear-gradient(145deg, #071321 0%, #081725 48%, #06111d 100%);
  color: #eaf2fb;
}
.block-container { max-width: 1600px; padding-top: 1.35rem; padding-bottom: 3.2rem; }
[data-testid="stHeader"] { background: rgba(5,15,26,.72); backdrop-filter: blur(14px); }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0b1c2d 0%,#071321 72%,#06101b 100%); border-right:1px solid rgba(107,164,218,.18); }
.brand { padding:1.25rem .65rem 1.5rem; }
.brand-title { font-size:1.42rem; letter-spacing:-.8px; }
.brand-subtitle { color:#91a9c2; letter-spacing:.25px; }
.nav-label { color:#6f91b3; letter-spacing:2px; }
div[role="radiogroup"] { gap:.38rem; }
div[role="radiogroup"] label { border:1px solid transparent; border-radius:11px; padding:.62rem .7rem; transition:all .18s ease; }
div[role="radiogroup"] label:hover { background:rgba(33,139,232,.11); border-color:rgba(74,163,240,.18); transform:translateX(2px); }
div[role="radiogroup"] label[data-checked="true"] { background:linear-gradient(100deg,rgba(18,126,224,.28),rgba(10,76,139,.18)); border-color:rgba(55,158,244,.42); box-shadow:inset 3px 0 0 #35a8ff; }
.page-header { display:flex; align-items:center; justify-content:space-between; gap:1.5rem; padding:1.3rem 1.55rem; margin:.15rem 0 1.35rem; border:1px solid rgba(92,151,205,.2); border-radius:16px; background:linear-gradient(112deg,rgba(16,42,66,.94),rgba(8,25,42,.88)); box-shadow:0 12px 34px rgba(0,0,0,.16); }
.page-kicker { color:#55baff; font-size:.64rem; font-weight:850; letter-spacing:2px; margin-bottom:.35rem; }
.page-heading { color:#f4f8fd; font-size:1.72rem; line-height:1.15; font-weight:850; letter-spacing:-.7px; }
.page-description { color:#9db4cb; font-size:.84rem; margin-top:.45rem; }
.system-status { white-space:nowrap; padding:.48rem .72rem; border-radius:999px; border:1px solid rgba(35,195,145,.25); background:rgba(18,153,112,.1); color:#83e2c2; font-size:.65rem; font-weight:800; letter-spacing:.7px; }
.status-dot { display:inline-block; width:7px; height:7px; margin-right:7px; border-radius:50%; background:#35d39a; box-shadow:0 0 10px rgba(53,211,154,.65); }
.hero { min-height:250px; margin-bottom:1.2rem; }
.section-title { color:#f2f7fd; font-size:1.08rem; letter-spacing:-.25px; margin:1.25rem 0 .85rem; }
.panel,.metric { box-shadow:0 10px 28px rgba(0,0,0,.13); }
.metric { min-height:98px; }
[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(17,43,68,.96),rgba(9,26,43,.96)); border:1px solid rgba(93,151,205,.2); border-radius:14px; padding:1rem 1.05rem; }
[data-testid="stMetricLabel"] p { color:#91abc5 !important; font-size:.76rem; }
[data-testid="stMetricValue"] { color:#f4f8ff !important; font-size:1.55rem; font-weight:850; }
.stButton > button,.stDownloadButton > button { border-radius:10px; min-height:2.65rem; font-weight:780; border:1px solid rgba(64,166,255,.68); background:linear-gradient(110deg,#087fe8,#0964c5 65%,#0754a6); box-shadow:0 6px 18px rgba(0,105,205,.18); }
.stButton > button:hover,.stDownloadButton > button:hover { border-color:#78c5ff; filter:brightness(1.1); }
[data-testid="stFileUploaderDropzone"] { border:1px dashed rgba(65,166,245,.62); background:rgba(9,31,51,.76); border-radius:13px; }
[data-testid="stDataFrame"],[data-testid="stTable"] { border:1px solid rgba(103,154,199,.23); border-radius:12px; overflow:hidden; }
[data-testid="stAlert"] { border-radius:12px; border:1px solid rgba(87,151,210,.2); }
[data-testid="stExpander"] { border:1px solid rgba(103,154,199,.2); border-radius:12px; background:rgba(12,31,49,.45); }
[data-baseweb="tab-list"] { gap:.35rem; border-bottom:1px solid rgba(112,159,202,.18); }
[data-baseweb="tab"][aria-selected="true"] { color:#62bcff; }
.stTextInput input,.stNumberInput input,.stTextArea textarea { background:#0c2033; border-color:#284660; color:#eaf3fc; }
hr { border-color:rgba(112,159,202,.16); }
.footer { color:#7892ab; }
@media (max-width:900px) { .page-header { align-items:flex-start; flex-direction:column; } .page-heading { font-size:1.4rem; } .block-container { padding:1rem .9rem 2rem; } }


/* ===== Aesthetic enterprise design system ===== */
:root {
  color-scheme: dark;
  --pl-bg: #07111c;
  --pl-surface: #0d1b29;
  --pl-surface-2: #102235;
  --pl-border: rgba(148, 183, 211, .14);
  --pl-text: #edf4f8;
  --pl-muted: #91a6b8;
  --pl-accent: #36c5c0;
  --pl-accent-2: #63a9ff;
}
html, body, [class*="css"] {
  font-family: Inter, "Segoe UI", system-ui, sans-serif;
  letter-spacing: .01em;
}
.stApp {
  color: var(--pl-text);
  background:
    radial-gradient(ellipse at 82% -8%, rgba(54,197,192,.10), transparent 34%),
    radial-gradient(ellipse at 6% 18%, rgba(72,139,214,.08), transparent 30%),
    linear-gradient(145deg, #07111c 0%, #081522 52%, #07111c 100%);
}
[data-testid="stHeader"] {
  background: rgba(7,17,28,.78);
  backdrop-filter: blur(16px);
}
[data-testid="stToolbar"] { right: 1.2rem; }
.block-container {
  max-width: 1580px;
  padding: 1.65rem 2.2rem 3.4rem;
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1a28 0%, #091622 65%, #07121d 100%);
  border-right: 1px solid var(--pl-border);
}
section[data-testid="stSidebar"] > div {
  padding: 1.35rem 1.05rem;
}
.brand {
  border-bottom: 1px solid var(--pl-border);
  padding: 1rem .65rem 1.5rem;
  margin-bottom: 1.25rem;
}
.brand-title { font-size: 1.38rem; letter-spacing: -.8px; }
.brand-subtitle { color: var(--pl-muted); font-size: .72rem; }
.nav-label {
  color: #7894a9;
  font-size: .64rem;
  letter-spacing: 2px;
  margin: 1.35rem .45rem .7rem;
}
div[role="radiogroup"] { gap: .42rem; }
div[role="radiogroup"] label {
  border: 1px solid transparent;
  border-radius: 11px;
  padding: .62rem .72rem;
  transition: background .18s ease, border-color .18s ease, transform .18s ease;
}
div[role="radiogroup"] label:hover {
  background: rgba(54,197,192,.08);
  border-color: rgba(54,197,192,.18);
  transform: translateX(2px);
}
div[role="radiogroup"] label[data-checked="true"] {
  background: linear-gradient(100deg, rgba(54,197,192,.17), rgba(54,169,255,.10));
  border-color: rgba(75,205,198,.34);
  box-shadow: inset 3px 0 0 var(--pl-accent);
}
div[role="radiogroup"] label p {
  color: #dce8f0 !important;
  font-weight: 650;
  font-size: .84rem;
}
.hero {
  min-height: 205px;
  padding: 2rem 2.15rem;
  margin: .1rem 0 1.45rem;
  border-radius: 20px;
  border: 1px solid rgba(113,166,203,.18);
  background:
    linear-gradient(105deg, rgba(10,29,44,.98) 0%, rgba(11,34,49,.91) 58%, rgba(8,25,39,.82) 100%),
    var(--hero-bg);
  background-size: cover;
  background-position: center;
  box-shadow: 0 18px 48px rgba(0,0,0,.18);
}
.hero-kicker, .page-kicker {
  color: #68d8d0;
  letter-spacing: 2px;
  font-size: .66rem;
  font-weight: 850;
}
.hero-title {
  color: #f3f8fb;
  font-size: clamp(2.15rem, 4vw, 3rem);
  letter-spacing: -1.7px;
  line-height: 1.06;
}
.hero-title span { color: #58b9ff; }
.hero-heading { font-size: 1.12rem; margin-top: .75rem; }
.hero-copy {
  color: #b1c2ce;
  font-size: .91rem;
  line-height: 1.7;
  max-width: 690px;
}
.hero-right { color: #d6e7ee; font-size: .78rem; }
.pills { gap: 1.35rem; margin-top: 1.3rem; }
.pill { color: #d3e2e9; font-size: .78rem; }
.pill b { color: var(--pl-accent); }
.section-title {
  color: #eff6fa;
  font-size: 1.08rem;
  font-weight: 780;
  letter-spacing: -.25px;
  margin: 1.55rem 0 .85rem;
}
.page-header {
  border-radius: 17px;
  border: 1px solid var(--pl-border);
  background: linear-gradient(115deg, rgba(16,38,54,.94), rgba(10,27,41,.92));
  box-shadow: 0 12px 34px rgba(0,0,0,.14);
  padding: 1.25rem 1.55rem;
  margin: .15rem 0 1.35rem;
}
.page-heading { font-size: 1.7rem; letter-spacing: -.8px; }
.page-description { color: #a2b6c4; line-height: 1.6; }
.system-status {
  border: 1px solid rgba(64,205,158,.24);
  background: rgba(34,158,119,.10);
  color: #8ce0bd;
  border-radius: 999px;
}
.metric, [data-testid="stMetric"] {
  background: linear-gradient(145deg, rgba(17,39,56,.96), rgba(11,27,41,.96));
  border: 1px solid var(--pl-border);
  border-radius: 15px;
  box-shadow: 0 9px 26px rgba(0,0,0,.12);
}
.metric {
  padding: 1.1rem 1.2rem;
  min-height: 94px;
  gap: .9rem;
}
.metric-icon {
  border-radius: 12px;
  color: #75ddd5;
  background: rgba(54,197,192,.12);
}
.metric-value { font-size: 1.55rem; letter-spacing: -.6px; }
.metric-label { color: #9bb0bd; font-size: .76rem; }
.panel {
  background: linear-gradient(145deg, rgba(16,37,53,.96), rgba(10,26,40,.96));
  border: 1px solid var(--pl-border);
  border-radius: 16px;
  padding: 1.15rem 1.2rem;
  box-shadow: 0 10px 28px rgba(0,0,0,.11);
}
.panel-title { font-size: .98rem; color: #edf5f8; }
.panel-subtitle { color: #92a9b8; line-height: 1.5; }
[data-testid="stMetric"] { padding: 1rem 1.15rem; }
[data-testid="stMetricLabel"] p { color: #9db2c0 !important; font-size: .78rem; }
[data-testid="stMetricValue"] { color: #f1f7fa !important; font-weight: 820; }
.stButton > button, .stDownloadButton > button {
  border-radius: 11px;
  min-height: 2.7rem;
  font-weight: 750;
  border: 1px solid rgba(83,205,198,.45);
  color: #f4fbfb;
  background: linear-gradient(105deg, #168f91, #1679b6);
  box-shadow: 0 7px 20px rgba(18,132,145,.15);
  transition: transform .18s ease, filter .18s ease, box-shadow .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  filter: brightness(1.12);
  transform: translateY(-1px);
  box-shadow: 0 10px 24px rgba(18,132,145,.22);
}
.stButton > button:focus, .stDownloadButton > button:focus {
  outline: 2px solid rgba(92,211,203,.55);
  outline-offset: 2px;
}
[data-testid="stFileUploaderDropzone"] {
  background: rgba(11,31,46,.76);
  border: 1px dashed rgba(82,190,193,.48);
  border-radius: 14px;
  transition: border-color .2s ease, background .2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: #58c9c5;
  background: rgba(16,47,62,.8);
}
[data-testid="stFileUploaderDropzone"] * { color: #c5d6df !important; }
.stTextInput input, .stNumberInput input, .stTextArea textarea {
  background: #0b1d2b;
  border: 1px solid #294354;
  color: #edf5f8;
  border-radius: 10px;
}
div[data-baseweb="select"] > div {
  background: #0b1d2b;
  border-color: #294354;
  border-radius: 10px;
}
[data-testid="stSlider"] [role="slider"] { background: var(--pl-accent); }
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--pl-border);
  border-radius: 13px;
  overflow: hidden;
}
[data-testid="stExpander"] {
  border: 1px solid var(--pl-border);
  border-radius: 13px;
  background: rgba(13,31,46,.48);
}
[data-testid="stAlert"] { border-radius: 12px; }
[data-baseweb="tab-list"] {
  gap: .35rem;
  border-bottom: 1px solid var(--pl-border);
}
[data-baseweb="tab"] { color: #a9bdc9; }
[data-baseweb="tab"][aria-selected="true"] { color: #68d8d0; }
hr { border-color: var(--pl-border); }
.footer {
  color: #8299a8;
  font-size: .75rem;
  border-top: 1px solid var(--pl-border);
  margin-top: 2.5rem;
  padding: 1.25rem 0 .4rem;
  text-align: center;
}
@media (max-width: 900px) {
  .block-container { padding: 1rem .9rem 2rem; }
  .hero { min-height: 180px; padding: 1.4rem; }
  .hero-title { font-size: 2rem; }
  .page-header { align-items: flex-start; flex-direction: column; }
  .page-heading { font-size: 1.4rem; }
  .pills { gap: .7rem; }
}

</style>

""",

    unsafe_allow_html=True,

)





def html(content):

    st.html(content)





# ============================================================

# INSPECTION HISTORY HELPERS

# ============================================================



def build_detection_records(boxes, labels, scores):

    """Convert model outputs into JSON/SQLite-friendly detection records."""

    records = []

    counts = {}



    for box, label, score in zip(boxes, labels, scores):

        class_id = int(label.item()) if hasattr(label, "item") else int(label)

        confidence = float(score.item()) if hasattr(score, "item") else float(score)

        coords = [

            float(value.item()) if hasattr(value, "item") else float(value)

            for value in box

        ]

        component = (

            CLASS_NAMES[class_id]

            if 0 <= class_id < len(CLASS_NAMES)

            else f"Class {class_id}"

        )



        counts[component] = counts.get(component, 0) + 1

        records.append({

            "component": component,

            "class_id": class_id,

            "confidence": confidence,

            "x1": coords[0],

            "y1": coords[1],

            "x2": coords[2],

            "y2": coords[3],

        })



    return records, counts





def apply_detection_filters(boxes, labels, scores, selected_classes=None,
                            min_object_area=0, max_detections=100,
                            confidence_threshold=0.25):
    """Apply UI confidence, class, area and count filters to raw candidates."""
    if len(scores) == 0:
        return boxes, labels, scores
    keep_indices = []
    allowed = set(selected_classes) if selected_classes is not None else None
    for index, (box, label) in enumerate(zip(boxes, labels)):
        class_id = int(label)
        if allowed is not None and class_id not in allowed:
            continue
        if float(scores[index]) < float(confidence_threshold):
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < float(min_object_area):
            continue
        keep_indices.append(index)



    # Highest-confidence detections are retained when a maximum is configured.

    keep_indices.sort(key=lambda idx: float(scores[idx]), reverse=True)

    if max_detections is not None:

        keep_indices = keep_indices[:int(max_detections)]



    indices = np.asarray(keep_indices, dtype=int)

    return boxes[indices], labels[indices], scores[indices]





def records_from_video_csv(csv_text):

    """Read sampled-frame detection rows from the video processor CSV."""

    if not csv_text or not csv_text.strip():

        return []

    try:

        frame = pd.read_csv(io.StringIO(csv_text))

        frame = frame.where(pd.notna(frame), None)

        return frame.to_dict(orient="records")

    except Exception:

        return []





def history_record_id(record):

    """Support either common primary-key naming convention."""

    if not record:

        return None

    return record.get("id", record.get("inspection_id"))





def dashboard_history_analytics(limit=1000):

    """Build dashboard statistics from persisted SQLite inspection records."""

    records = list_inspections(limit=limit)

    total = len(records)

    image_total = sum(r.get("inspection_type") == "Image" for r in records)

    video_total = sum(r.get("inspection_type") == "Video" for r in records)

    detection_total = sum(int(r.get("detection_count") or 0) for r in records)



    component_totals = {}

    daily_totals = {}

    for record in records:

        components = record.get("component_counts", {})

        if isinstance(components, str):

            try:

                components = json.loads(components)

            except (TypeError, ValueError):

                components = {}

        if isinstance(components, dict):

            for name, count in components.items():

                component_totals[name] = component_totals.get(name, 0) + int(count or 0)



        created = str(record.get("created_at") or "")

        day = created[:10]

        if day:

            daily_totals[day] = daily_totals.get(day, 0) + 1



    return {

        "records": records,

        "total": total,

        "images": image_total,

        "videos": video_total,

        "detections": detection_total,

        "components": component_totals,

        "daily": daily_totals,

    }





# ============================================================

# MODEL

# ============================================================



@st.cache_resource(show_spinner="Loading PowerLine_AI...")

def load_model():

    """Load the trained Faster R-CNN checkpoint through the modular loader."""

    model, model_info = load_model_from_checkpoint()

    return model


def detect_with_diagnostics(model, image):
    """Return low-threshold Faster R-CNN candidates and restore model settings."""
    roi_heads = getattr(model, "roi_heads", None)
    old_score_thresh = getattr(roi_heads, "score_thresh", None)
    old_detection_cap = getattr(roi_heads, "detections_per_img", None)
    try:
        if roi_heads is not None:
            if old_score_thresh is not None:
                roi_heads.score_thresh = 0.001
            if old_detection_cap is not None:
                roi_heads.detections_per_img = max(int(old_detection_cap), 300)
        return detect(model, image, threshold=0.0)
    finally:
        if roi_heads is not None:
            if old_score_thresh is not None:
                roi_heads.score_thresh = old_score_thresh
            if old_detection_cap is not None:
                roi_heads.detections_per_img = old_detection_cap






# ============================================================

# SIDEBAR

# ============================================================



with st.sidebar:

    html("""

    <div class="brand">

        <div class="brand-title">⚡ PowerLine<span>_AI</span></div>

        <div class="brand-subtitle">

            Inspect Smarter. Power Safer.

        </div>

    </div>

    """)



    html('<div class="nav-label">NAVIGATION</div>')



    page = st.radio(

        "",

        [

            "🏠 Dashboard",

            "🔍 Image Inspection",

            "🎥 Video Inspection",

            "🗂 Inspection History",
            "🏗 Asset Management",
            "🛠 Maintenance Center",
            "🔄 Inspection Comparison",
            "🖥 System Monitoring",
            "📊 Model Performance",

            "ℹ️ About Project",

        ],

        label_visibility="collapsed",

    )



    st.markdown("---")



    if DEVICE.type == "cuda":

        gpu = torch.cuda.get_device_name(0)



        html(f"""

        <div style="

            padding:12px;

            border-radius:10px;

            background:rgba(20,140,255,.07);

            border:1px solid rgba(20,140,255,.13);

        ">

            <div style="

                color:#5dbaff;

                font-size:10px;

                font-weight:800;

            ">● GPU ACCELERATION</div>

            <div style="

                color:#9eb2c7;

                font-size:11px;

                margin-top:5px;

            ">{gpu}</div>

        </div>

        """)

    else:

        html("""

        <div class="result-ok">

            ● Running on CPU

        </div>

        """)



    html("""

    <div class="quote-card">

        “Reliable Grids<br>

        Stronger Tomorrows”<br>

        <span style="color:#1598ff;">⚡</span>

    </div>



    <div class="footer">

        © 2026 PowerLine_AI<br>

        All rights reserved.

    </div>

    """)





# ============================================================

# PAGE HEADER / DASHBOARD HERO

if page != "🏠 Dashboard":
    page_meta = {
        "🔍 Image Inspection": ("INSPECTION WORKSPACE", "Image Inspection", "Analyze an inspection image and review component-level detections."),
        "🎥 Video Inspection": ("VIDEO ANALYTICS", "Video Inspection", "Process inspection footage and export annotated results."),
        "🗂 Inspection History": ("RECORDS & REPORTING", "Inspection History", "Review saved inspections, inspect individual records and generate reports."),
        "🏗 Asset Management": ("ASSET REGISTER", "Asset Management", "Maintain a structured register of towers and power-line assets."),
        "🛠 Maintenance Center": ("OPERATIONS", "Maintenance Center", "Create and track maintenance work orders and follow-up actions."),
        "🔄 Inspection Comparison": ("CHANGE REVIEW", "Inspection Comparison", "Compare component detection summaries across two saved inspections."),
        "🖥 System Monitoring": ("SYSTEM HEALTH", "System Monitoring", "Review runtime, hardware availability and inspection confidence signals."),
        "📊 Model Performance": ("MODEL INTELLIGENCE", "Model Performance", "Explore validation metrics, training progress and class-wise performance."),
        "ℹ️ About Project": ("PLATFORM OVERVIEW", "About PowerLine AI", "A deep-learning platform for power-line component inspection."),
    }
    kicker, title, description = page_meta.get(page, ("POWERLINE AI", "Workspace", ""))
    html(f"""
    <div class="page-header">
      <div class="page-header-main">
        <div class="page-kicker">{kicker}</div>
        <div class="page-heading">{title}</div>
        <div class="page-description">{description}</div>
      </div>
      <div class="system-status"><span class="status-dot"></span> SYSTEM READY</div>
    </div>
    """)

if page == "🏠 Dashboard":
    # HERO

    # ============================================================

    hero_b64 = ""



    if os.path.exists(HERO_IMAGE):

        with open(HERO_IMAGE, "rb") as f:

            hero_b64 = base64.b64encode(f.read()).decode("utf-8")

    html(f"""

    <div class="hero">



        <img

            src="data:image/jpeg;base64,{hero_b64}"

            style="

                position:absolute;

                inset:0;

                width:100%;

                height:100%;

                object-fit:cover;

                object-position:center;

                opacity:0.48;

                z-index:0;

            "

        >



        <div style="

            position:absolute;

            inset:0;

            background:

                linear-gradient(

                    90deg,

                    rgba(4,16,31,0.98) 0%,

                    rgba(5,25,45,0.88) 43%,

                    rgba(5,25,45,0.45) 100%

                );

            z-index:1;

        "></div>



        <div class="hero-right">

            AI for a Safer,<br>

            Stronger Tomorrow

            <div class="hero-right-line"></div>

        </div>



        <div class="hero-content">



            <div class="hero-kicker">

                AI-POWERED POWER-LINE INSPECTION

            </div>



            <div class="hero-title">

                PowerLine<span>_AI</span>

            </div>



            <div class="hero-heading">

                AI-Powered Power-Line Component Inspection

            </div>



            <div class="hero-copy">

                Detect and identify critical power-line components

                from inspection images using deep learning.

                Faster, safer, and more reliable infrastructure

                monitoring.

            </div>



            <div class="pills">

                <div class="pill">

                    <b>⚡</b>AI Vision

                </div>



                <div class="pill">

                    <b>◆</b>Smart Inspection

                </div>



                <div class="pill">

                    <b>▮</b>Reliable Results

                </div>



                <div class="pill">

                    <b>◷</b>Safer Future

                </div>

            </div>



        </div>

    </div>

    """)  





    # ============================================================

    # METRICS

    # ============================================================



    c1, c2, c3, c4 = st.columns(4)



    metric_data = [

        ("🖼", f"{MAP50 * 100:.2f}%", "mAP@50"),

        ("▥", f"{MAP5095 * 100:.2f}%", "mAP@50:95"),

        ("▰", "18", "Component Classes"),

        ("🗃", f"{VALIDATION_IMAGES:,}", "Validation Images"),

    ]



    for col, (icon, value, label) in zip(

        [c1, c2, c3, c4],

        metric_data

    ):

        with col:

            html(f"""

            <div class="metric">

                <div class="metric-icon">{icon}</div>

                <div>

                    <div class="metric-value">{value}</div>

                    <div class="metric-label">{label}</div>

                </div>

            </div>

            """)





    # ============================================================



# DASHBOARD

# ============================================================



if page == "🏠 Dashboard":



    st.markdown(

        '<div class="section-title">Inspection Dashboard</div>',

        unsafe_allow_html=True

    )



    # Live operational metrics are calculated from saved SQLite records.

    try:

        live_stats = dashboard_history_analytics()

        live_cols = st.columns(4)

        live_metrics = [

            ("Total Inspections", f'{live_stats["total"]:,}'),

            ("Image Inspections", f'{live_stats["images"]:,}'),

            ("Video Inspections", f'{live_stats["videos"]:,}'),

            ("Components Detected", f'{live_stats["detections"]:,}'),

        ]

        for metric_col, (metric_label, metric_value) in zip(live_cols, live_metrics):

            with metric_col:

                st.metric(metric_label, metric_value)

        if live_stats["total"] == 1000:

            st.caption("Dashboard analytics are based on the latest 1,000 saved inspections.")

    except Exception as exc:

        live_stats = {"records": [], "components": {}, "daily": {}, "total": 0,

                      "images": 0, "videos": 0, "detections": 0}

        st.warning(f"Live inspection statistics are temporarily unavailable: {exc}")



    st.info(

        "Run inspections from the Image Inspection or Video Inspection pages. "

        "Completed results are automatically saved here for dashboard analytics."

    )



    # Bottom charts

    st.markdown(

        '<div class="section-title">Inspection Analytics</div>',

        unsafe_allow_html=True

    )



    chart1, chart2, recent = st.columns(

        [1.15, 1.05, .9],

        gap="medium"

    )



    with chart1:

        html("""

        <div class="panel">

            <div class="panel-title">▥ Component Distribution</div>

            <div class="panel-subtitle">

                Components detected in saved inspections

            </div>

        </div>

        """)



        component_counts = pd.DataFrame(

            list(live_stats["components"].items()),

            columns=["Component", "Detections"],

        ).sort_values("Detections", ascending=False) if live_stats["components"] else pd.DataFrame(

            columns=["Component", "Detections"]

        )



        if not component_counts.empty:

            st.bar_chart(

                component_counts.head(10).set_index("Component"),

                height=250

            )

        else:

            st.info("Component analytics will appear after your first saved inspection.")



    with chart2:

        html("""

        <div class="panel">

            <div class="panel-title">⌁ Inspection Trend</div>

            <div class="panel-subtitle">

                Daily saved inspection count

            </div>

        </div>

        """)



        daily_activity = pd.DataFrame(

            sorted(live_stats["daily"].items()),

            columns=["Date", "Inspections"],

        )

        if not daily_activity.empty:

            st.line_chart(

                daily_activity.set_index("Date"),

                height=250

            )

        else:

            st.info("Daily inspection trends will appear after inspections are saved.")



    with recent:

        html("""

        <div class="panel">

            <div class="panel-title">◷ Recent Inspections</div>

            <div class="panel-subtitle">Latest saved inspection activity</div>

        </div>

        """)

        recent_records = live_stats["records"][:5]

        if recent_records:

            recent_rows = []

            for record in recent_records:

                recent_rows.append({

                    "File": record.get("source_filename", "—"),

                    "Type": record.get("inspection_type", "—"),

                    "Detections": int(record.get("detection_count") or 0),

                    "Status": record.get("status", "Completed"),

                })

            st.dataframe(

                pd.DataFrame(recent_rows),

                hide_index=True,

                use_container_width=True,

                height=250,

            )

        else:

            st.info("No inspections saved yet.")





# ============================================================

# IMAGE INSPECTION

# ============================================================



elif page == "🔍 Image Inspection":



    st.markdown(

        '<div class="section-title">Image Inspection</div>',

        unsafe_allow_html=True

    )

    st.caption(

        "Upload a power-line image, run the trained Faster R-CNN detector, "

        "and review the annotated output and component-level results."

    )



    upload, settings = st.columns([2, 1], gap="medium")



    with upload:

        html("""

        <div class="panel">

            <div class="panel-title">⬆ Upload Inspection Image</div>

            <div class="panel-subtitle">

                Upload JPG, JPEG or PNG power-line imagery.

            </div>

        </div>

        """)

        file = st.file_uploader(

            "Choose image",

            type=["jpg", "jpeg", "png"],

            label_visibility="collapsed",

            key="inspection_file"

        )



    with settings:

        html("""

        <div class="panel">

            <div class="panel-title">⚙ Detection Settings</div>

            <div class="panel-subtitle">

                Minimum confidence for displayed detections.

            </div>

        </div>

        """)

        threshold = st.slider(

            "Confidence threshold",

            min_value=0.10,

            max_value=0.95,

            value=0.25,

            step=0.05,

            key="image_inspection_confidence"

        )

        selected_component_names = st.multiselect(

            "Component classes",

            options=CLASS_NAMES[1:],

            default=CLASS_NAMES[1:],

            help="Select the component classes to keep in the displayed results.",

            key="image_component_class_filter"

        )

        min_object_area = st.number_input(

            "Minimum bounding-box area (pixels²)",

            min_value=0,

            max_value=2000000,

            value=0,

            step=100,

            help="Set to 0 to disable size filtering.",

            key="image_min_object_area"

        )

        max_detections = st.slider(

            "Maximum detections per image",

            min_value=1,

            max_value=100,

            value=100,

            step=1,

            key="image_max_detections"

        )

        show_boxes = st.checkbox(

            "Show bounding boxes and labels",

            value=True,

            key="image_show_boxes"

        )



    if file is not None:

        image = Image.open(file).convert("RGB")

        st.markdown("#### Uploaded Image")

        st.image(image, caption=file.name, width="stretch")



        if st.button(

            "⚡ Run AI Detection",

            key="inspection_run",

            type="primary",

            use_container_width=True

        ):

            with st.spinner("Analyzing inspection image..."):

                model = load_model()

                raw_boxes, raw_labels, raw_scores, elapsed = detect_with_diagnostics(model, image)
                raw_candidate_records, raw_candidate_counts = build_detection_records(
                    raw_boxes, raw_labels, raw_scores
                )

                selected_class_ids = [

                    CLASS_NAMES.index(name)

                    for name in selected_component_names

                ]

                boxes, labels, scores = apply_detection_filters(
                    raw_boxes, raw_labels, raw_scores,
                    selected_classes=selected_class_ids,
                    min_object_area=min_object_area,
                    max_detections=max_detections,
                    confidence_threshold=threshold,
                )



            detection_records, component_counts = build_detection_records(

                boxes, labels, scores

            )

            inspection_id = save_inspection(

                inspection_type="Image",

                source_filename=file.name,

                confidence_threshold=float(threshold),

                detection_count=len(detection_records),

                processing_seconds=float(elapsed),

                component_counts=component_counts,

                detections=detection_records,

            )



            result = (annotate(image, boxes, labels, scores) if show_boxes else image.copy())



            st.session_state["last_image_inspection_id"] = inspection_id

            st.session_state["last_image_result"] = result

            st.session_state["last_image_original"] = image.copy()

            st.session_state["last_image_records"] = detection_records

            st.session_state["last_image_counts"] = component_counts
            st.session_state["last_image_raw_records"] = raw_candidate_records
            st.session_state["last_image_raw_counts"] = raw_candidate_counts

            st.session_state["last_image_elapsed"] = float(elapsed)

            st.session_state["last_image_filename"] = file.name



            st.success(f"Inspection #{inspection_id} saved to history.")



    # Keep the latest completed inspection visible across Streamlit reruns.

    if "last_image_result" in st.session_state:

        st.markdown(

            '<div class="section-title">Inspection Results</div>',

            unsafe_allow_html=True

        )



        result_col, summary_col = st.columns([1.5, 1], gap="medium")

        with result_col:

            st.markdown("#### Annotated Image")

            st.image(

                st.session_state["last_image_result"],

                caption="Detected components with bounding boxes",

                width="stretch"

            )



        records = st.session_state.get("last_image_records", [])

        counts = st.session_state.get("last_image_counts", {})

        elapsed = st.session_state.get("last_image_elapsed", 0.0)



        with summary_col:

            st.markdown("#### Inspection Summary")

            m1, m2 = st.columns(2)

            m1.metric("Detections", len(records))

            m2.metric("Processing Time", f"{elapsed:.2f}s")

            st.metric("Unique Components", len(counts))



            if counts:

                counts_df = pd.DataFrame(

                    [{"Component": name, "Count": count}

                     for name, count in sorted(

                         counts.items(), key=lambda item: item[1], reverse=True

                     )]

                )

                st.markdown("**Component-wise Summary**")

                st.dataframe(

                    counts_df,

                    hide_index=True,

                    use_container_width=True,

                    height=260

                )

            else:

                st.info("No components detected at the selected confidence threshold.")



        raw_records = st.session_state.get("last_image_raw_records", [])
        raw_counts = st.session_state.get("last_image_raw_counts", {})
        with st.expander("🔎 Raw model prediction diagnostics", expanded=True):
            st.caption(
                "Candidates returned by Faster R-CNN before the app's confidence, "
                "class, area and maximum-count filters. RPN and NMS processing "
                "inside the model still applies."
            )
            st.metric("Raw candidates returned by model", len(raw_records))
            if raw_counts:
                raw_count_df = pd.DataFrame([
                    {"Component": name, "Raw candidates": count}
                    for name, count in sorted(
                        raw_counts.items(), key=lambda item: item[1], reverse=True
                    )
                ])
                st.dataframe(raw_count_df, hide_index=True, use_container_width=True)
            if raw_records:
                raw_details_df = pd.DataFrame([
                    {
                        "Component": record["component"],
                        "Confidence": f'{record["confidence"] * 100:.2f}%',
                        "X1": round(record["x1"]), "Y1": round(record["y1"]),
                        "X2": round(record["x2"]), "Y2": round(record["y2"]),
                    }
                    for record in sorted(
                        raw_records, key=lambda item: item["confidence"], reverse=True
                    )
                ])
                st.dataframe(raw_details_df, hide_index=True, use_container_width=True)
            else:
                st.info(
                    "The model returned no candidates even after lowering its ROI "
                    "score threshold. The issue is not just the app confidence filter."
                )

        st.markdown("#### Detection Details")

        if records:

            details_df = pd.DataFrame([

                {

                    "#": index,

                    "Component": record["component"],

                    "Confidence": f'{record["confidence"] * 100:.2f}%',

                    "X1": round(record["x1"]),

                    "Y1": round(record["y1"]),

                    "X2": round(record["x2"]),

                    "Y2": round(record["y2"]),

                }

                for index, record in enumerate(records, start=1)

            ])

            st.dataframe(

                details_df,

                hide_index=True,

                use_container_width=True

            )

        else:

            st.warning("No component was detected above the selected confidence threshold.")



        download_buffer = io.BytesIO()

        st.session_state["last_image_result"].save(

            download_buffer, format="PNG"

        )

        st.download_button(

            "⬇ Download Annotated Image",

            data=download_buffer.getvalue(),

            file_name="powerline_detection.png",

            mime="image/png",

            use_container_width=True,

            key="download_image_inspection_result"

        )

        st.caption(

            f'Latest saved record: Inspection #{st.session_state.get("last_image_inspection_id", "—")} '

            f'· File: {st.session_state.get("last_image_filename", "—")}'

        )





    # ========================================================

    # BATCH IMAGE INSPECTION

    # ========================================================



    st.markdown("---")

    st.markdown(

        '<div class="section-title">Batch Image Inspection</div>',

        unsafe_allow_html=True

    )

    st.caption(

        "Upload multiple JPG, JPEG or PNG images and inspect them in one run. "

        "Each image is saved separately in Inspection History."

    )



    batch_files = st.file_uploader(

        "Choose multiple inspection images",

        type=["jpg", "jpeg", "png"],

        accept_multiple_files=True,

        key="batch_inspection_files"

    )



    if batch_files:

        st.info(f"{len(batch_files)} image(s) selected.")

        if st.button(

            "⚡ Run Batch Inspection",

            key="run_batch_inspection",

            type="primary",

            use_container_width=True

        ):

            batch_rows = []

            batch_details = []

            zip_buffer = io.BytesIO()

            model = load_model()



            progress = st.progress(0, text="Starting batch inspection...")

            with zipfile.ZipFile(

                zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED

            ) as result_zip:

                for index, uploaded in enumerate(batch_files, start=1):

                    try:

                        original = Image.open(uploaded).convert("RGB")

                        raw_boxes, raw_labels, raw_scores, elapsed = detect_with_diagnostics(model, original)

                        selected_class_ids = [

                            CLASS_NAMES.index(name)

                            for name in selected_component_names

                        ]

                        boxes, labels, scores = apply_detection_filters(
                    raw_boxes, raw_labels, raw_scores,
                    selected_classes=selected_class_ids,
                    min_object_area=min_object_area,
                    max_detections=max_detections,
                    confidence_threshold=threshold,
                )

                        records, counts = build_detection_records(

                            boxes, labels, scores

                        )

                        annotated_image = (

                            annotate(original, boxes, labels, scores)

                            if show_boxes else original.copy()

                        )



                        inspection_id = save_inspection(

                            inspection_type="Image",

                            source_filename=uploaded.name,

                            confidence_threshold=float(threshold),

                            detection_count=len(records),

                            processing_seconds=float(elapsed),

                            component_counts=counts,

                            detections=records,

                        )



                        safe_stem = os.path.splitext(

                            os.path.basename(uploaded.name)

                        )[0]

                        image_bytes = io.BytesIO()

                        annotated_image.save(image_bytes, format="PNG")

                        result_zip.writestr(

                            f"{index:03d}_{safe_stem}_annotated.png",

                            image_bytes.getvalue()

                        )



                        batch_rows.append({

                            "Image": uploaded.name,

                            "Detections": len(records),

                            "Unique Components": len(counts),

                            "Processing Time (s)": round(float(elapsed), 2),

                            "Inspection ID": inspection_id,

                            "Status": "Completed",

                        })

                        for record in records:

                            batch_details.append({

                                "Image": uploaded.name,

                                "Component": record["component"],

                                "Confidence": f'{record["confidence"] * 100:.2f}%',

                                "X1": round(record["x1"]),

                                "Y1": round(record["y1"]),

                                "X2": round(record["x2"]),

                                "Y2": round(record["y2"]),

                            })



                    except Exception as batch_error:

                        batch_rows.append({

                            "Image": uploaded.name,

                            "Detections": 0,

                            "Unique Components": 0,

                            "Processing Time (s)": 0.0,

                            "Inspection ID": "—",

                            "Status": f"Failed: {batch_error}",

                        })



                    progress.progress(

                        index / len(batch_files),

                        text=f"Processed {index} of {len(batch_files)} images"

                    )



            st.session_state["last_batch_rows"] = batch_rows

            st.session_state["last_batch_details"] = batch_details

            st.session_state["last_batch_zip"] = zip_buffer.getvalue()

            st.success(

                f"Batch inspection finished: "

                f"{sum(row['Status'] == 'Completed' for row in batch_rows)} "

                f"of {len(batch_rows)} image(s) processed successfully."

            )



    # Persist batch output across Streamlit reruns.

    if st.session_state.get("last_batch_rows"):

        batch_rows = st.session_state["last_batch_rows"]

        batch_df = pd.DataFrame(batch_rows)

        successful_rows = [

            row for row in batch_rows if row["Status"] == "Completed"

        ]

        total_batch_detections = sum(

            row["Detections"] for row in successful_rows

        )

        unique_batch_components = sorted({

            detail["Component"]

            for detail in st.session_state.get("last_batch_details", [])

        })



        st.markdown("#### Batch Summary")

        metric_cols = st.columns(3)

        metric_cols[0].metric("Images Processed", len(successful_rows))

        metric_cols[1].metric("Total Detections", total_batch_detections)

        metric_cols[2].metric(

            "Unique Components", len(unique_batch_components)

        )

        st.dataframe(

            batch_df,

            hide_index=True,

            use_container_width=True

        )



        if st.session_state.get("last_batch_details"):

            st.markdown("#### Batch Detection Details")

            st.dataframe(

                pd.DataFrame(st.session_state["last_batch_details"]),

                hide_index=True,

                use_container_width=True

            )

        else:

            st.info("No components were detected in the successfully processed images.")



        st.download_button(

            "⬇ Download All Annotated Images (ZIP)",

            data=st.session_state["last_batch_zip"],

            file_name="powerline_batch_annotated_results.zip",

            mime="application/zip",

            use_container_width=True,

            key="download_batch_annotated_zip"

        )







# ============================================================

# VIDEO INSPECTION

# ============================================================



elif page == "🎥 Video Inspection":



    st.markdown(

        '<div class="section-title">Video Inspection</div>',

        unsafe_allow_html=True

    )



    st.markdown(

        """

        Upload a power-line inspection video. The existing Faster R-CNN

        model will detect components in selected frames and generate an

        annotated video with downloadable detection records.

        """

    )



    upload_col, settings_col = st.columns([2, 1], gap="medium")



    with upload_col:

        html("""

        <div class="panel">

            <div class="panel-title">🎥 Upload Inspection Video</div>

            <div class="panel-subtitle">

                Supported formats: MP4, AVI, MOV and MKV.

            </div>

        </div>

        """)

        video_file = st.file_uploader(

            "Choose video",

            type=["mp4", "avi", "mov", "mkv"],

            key="video_inspection_file",

            label_visibility="collapsed"

        )



    with settings_col:

        html("""

        <div class="panel">

            <div class="panel-title">⚙ Video Detection Settings</div>

            <div class="panel-subtitle">

                Adjust confidence and frame sampling.

            </div>

        </div>

        """)

        video_threshold = st.slider(

            "Minimum confidence",

            min_value=0.10,

            max_value=0.95,

            value=0.50,

            step=0.05,

            key="video_confidence"

        )

        frame_interval = st.select_slider(

            "Run detection every N frames",

            options=[ 5, 10, 15, 20, 30],

            value=5,

            key="video_frame_interval",

            help="For example, 5 means the model runs on every fifth frame. Other frames are copied unchanged."

        )



    if video_file is not None:

        st.video(video_file)



        st.caption(

            f"File: {video_file.name} · "

            f"Size: {video_file.size / (1024 * 1024):.2f} MB"

        )



        if st.button(

            "⚡ Run Video Inspection",

            key="run_video_inspection",

            type="primary",

            use_container_width=True

        ):

            input_suffix = os.path.splitext(video_file.name)[1].lower() or ".mp4"



            input_path = None

            output_path = None



            try:

                with tempfile.NamedTemporaryFile(

                    delete=False, suffix=input_suffix

                ) as input_temp:

                    input_temp.write(video_file.getvalue())

                    input_path = input_temp.name



                with tempfile.NamedTemporaryFile(

                    delete=False, suffix=".mp4"

                ) as output_temp:

                    output_path = output_temp.name



                model = load_model()

                progress_bar = st.progress(0, text="Preparing video...")

                status_text = st.empty()



                def update_video_progress(progress, message):

                    progress_bar.progress(

                        min(max(float(progress), 0.0), 1.0),

                        text=message

                    )

                    status_text.caption(message)



                with st.spinner("Processing video frames with Faster R-CNN..."):

                    video_result = process_video(

                        input_path=input_path,

                        output_path=output_path,

                        model=model,

                        confidence_threshold=video_threshold,

                        frame_interval=frame_interval,

                        progress_callback=update_video_progress

                    )



                with open(output_path, "rb") as output_file:

                    annotated_video_bytes = output_file.read()



                csv_text = video_result.get("csv_text", "")

                csv_bytes = csv_text.encode("utf-8")

                summary = video_result["summary"]

                video_detections = records_from_video_csv(csv_text)



                video_inspection_id = save_inspection(

                    inspection_type="Video",

                    source_filename=video_file.name,

                    confidence_threshold=float(video_threshold),

                    detection_count=int(summary["total_detections"]),

                    processing_seconds=float(summary["processing_seconds"]),

                    component_counts=summary.get("component_counts", {}),

                    detections=video_detections,

                    frame_interval=int(frame_interval),

                    total_frames=int(summary["total_frames"]),

                    analyzed_frames=int(summary["analyzed_frames"]),

                )



                st.session_state["video_inspection_result"] = {

                    "video_bytes": annotated_video_bytes,

                    "csv_bytes": csv_bytes,

                    "summary": summary,

                    "filename": os.path.splitext(video_file.name)[0],

                    "inspection_id": video_inspection_id,

                }



                progress_bar.progress(1.0, text="Video inspection completed.")

                status_text.success("Video inspection completed successfully.")



            except Exception as exc:

                st.error(f"Video processing failed: {exc}")

                st.exception(exc)



            finally:

                for temporary_path in (input_path, output_path):

                    if temporary_path and os.path.exists(temporary_path):

                        try:

                            os.remove(temporary_path)

                        except OSError:

                            pass



    if "video_inspection_result" in st.session_state:

        saved_video = st.session_state["video_inspection_result"]

        summary = saved_video["summary"]



        st.markdown(

            '<div class="section-title">Video Inspection Results</div>',

            unsafe_allow_html=True

        )



        metric_columns = st.columns(4)

        video_metrics = [

            ("Frames in video", f'{summary["total_frames"]:,}'),

            ("Frames analyzed", f'{summary["analyzed_frames"]:,}'),

            ("Detections", f'{summary["total_detections"]:,}'),

            ("Processing time", f'{summary["processing_seconds"]:.2f}s'),

        ]



        for metric_col, (metric_label, metric_value) in zip(

            metric_columns, video_metrics

        ):

            with metric_col:

                st.metric(metric_label, metric_value)



        st.success(

            f'Video inspection #{saved_video["inspection_id"]} saved to history.'

        )



        preview_tab, details_tab = st.tabs(

            ["🎬 Annotated Video", "📊 Detection Details"]

        )

        with preview_tab:

            st.video(saved_video["video_bytes"])

            st.caption(

                "The annotated output contains detections on sampled frames. "

                "Frames between samples are retained without fresh detections."

            )



        with details_tab:

            counts_data = summary.get("component_counts", {})

            if counts_data:

                st.markdown("**Component-wise detection summary**")

                counts_frame = pd.DataFrame(

                    [

                        {"Component": name, "Detection Count": count}

                        for name, count in counts_data.items()

                    ]

                ).sort_values("Detection Count", ascending=False)

                st.dataframe(

                    counts_frame,

                    hide_index=True,

                    use_container_width=True

                )



            if saved_video.get("csv_bytes"):

                try:

                    detail_frame = pd.read_csv(

                        io.BytesIO(saved_video["csv_bytes"])

                    )

                    if not detail_frame.empty:

                        st.markdown("**Frame-wise detection records**")

                        st.caption(

                            f'{len(detail_frame):,} detection record(s) from '

                            f'{summary["analyzed_frames"]:,} analyzed frame(s).'

                        )

                        st.dataframe(

                            detail_frame,

                            hide_index=True,

                            use_container_width=True

                        )

                    else:

                        st.info("The CSV contains no detection rows.")

                except Exception as table_error:

                    st.warning(

                        f"Could not display the CSV details: {table_error}"

                    )



        download_col1, download_col2 = st.columns(2)

        with download_col1:

            st.download_button(

                "⬇ Download Annotated Video",

                data=saved_video["video_bytes"],

                file_name=f'{saved_video["filename"]}_annotated.mp4',

                mime="video/mp4",

                use_container_width=True,

                key="download_annotated_video"

            )



        with download_col2:

            st.download_button(

                "⬇ Download Detection CSV",

                data=saved_video["csv_bytes"],

                file_name=f'{saved_video["filename"]}_detections.csv',

                mime="text/csv",

                use_container_width=True,

                key="download_video_csv"

            )



        if summary["total_detections"] == 0:

            st.info(

                "No components were detected above the selected confidence threshold. "

                "Try lowering the confidence threshold and inspect the video again."

            )









# ============================================================

# INSPECTION HISTORY

# ============================================================



elif page == "🗂 Inspection History":



    st.markdown(

        '<div class="section-title">Inspection History</div>',

        unsafe_allow_html=True

    )

    st.caption(

        "Previously completed image and video inspections stored in the local SQLite database."

    )



    try:

        history = list_inspections(limit=500)

    except Exception as exc:

        history = []

        st.error(f"Could not load inspection history: {exc}")



    if not history:

        st.info(

            "No saved inspections yet. Run an Image or Video Inspection to create your first history record."

        )

    else:

        history_df = pd.DataFrame(history)



        # Keep the main table readable while retaining the full record for PDF generation.

        preferred_columns = [

            "id", "inspection_id", "created_at", "inspection_type",

            "source_filename", "confidence_threshold", "detection_count",

            "processing_seconds", "frame_interval", "total_frames",

            "analyzed_frames",

        ]

        visible_columns = [

            column for column in preferred_columns

            if column in history_df.columns

        ]

        if visible_columns:

            st.dataframe(

                history_df[visible_columns],

                hide_index=True,

                use_container_width=True,

            )

        else:

            st.dataframe(history_df, hide_index=True, use_container_width=True)



        available_ids = [

            history_record_id(record)

            for record in history

            if history_record_id(record) is not None

        ]



        if available_ids:

            selected_id = st.selectbox(

                "Select an inspection to view or download its report",

                options=available_ids,

                format_func=lambda value: f"Inspection #{value}",

                key="history_selected_id",

            )



            try:

                selected_record = get_inspection(selected_id)

            except Exception as exc:

                selected_record = None

                st.error(f"Could not load inspection details: {exc}")



            if selected_record:

                st.markdown("#### Inspection Details")

                detail_cols = st.columns(4)

                detail_values = [

                    ("Type", selected_record.get("inspection_type", "—")),

                    ("File", selected_record.get("source_filename", "—")),

                    ("Detections", selected_record.get("detection_count", 0)),

                    (

                        "Processing Time",

                        f'{float(selected_record.get("processing_seconds") or 0):.2f}s'

                    ),

                ]

                for col, (label, value) in zip(detail_cols, detail_values):

                    with col:

                        st.metric(label, value)



                component_data = selected_record.get("component_counts", {})

                if isinstance(component_data, str):

                    try:

                        import json

                        component_data = json.loads(component_data)

                    except (ValueError, TypeError):

                        component_data = {}



                if component_data:

                    st.markdown("**Detected component counts**")

                    component_df = pd.DataFrame(

                        list(component_data.items()),

                        columns=["Component", "Count"],

                    ).sort_values("Count", ascending=False)

                    st.dataframe(

                        component_df,

                        hide_index=True,

                        use_container_width=True,

                    )



                try:

                    pdf_bytes = generate_inspection_pdf(selected_record)

                    st.download_button(

                        "⬇ Download PDF Inspection Report",

                        data=pdf_bytes,

                        file_name=f"PowerLine_AI_Inspection_{selected_id}.pdf",

                        mime="application/pdf",

                        use_container_width=True,

                        key=f"download_history_pdf_{selected_id}",

                    )

                except Exception as exc:

                    st.error(f"PDF report generation failed: {exc}")



                with st.expander("Delete this inspection record"):

                    st.warning(

                        "This removes the selected record from the local inspection history. "

                        "It does not delete the original image or video."

                    )

                    confirm_delete = st.checkbox(

                        "I understand that this history record will be deleted.",

                        key=f"confirm_delete_{selected_id}",

                    )

                    if st.button(

                        "Delete Inspection Record",

                        disabled=not confirm_delete,

                        key=f"delete_history_{selected_id}",

                    ):

                        try:

                            delete_inspection(selected_id)

                            st.success(f"Inspection #{selected_id} deleted.")

                            st.rerun()

                        except Exception as exc:

                            st.error(f"Could not delete inspection: {exc}")





# ============================================================

# PERFORMANCE

# ============================================================




# ============================================================
# ASSET MANAGEMENT
# ============================================================

elif page == "🏗 Asset Management":
    st.markdown('<div class="section-title">Registered Infrastructure Assets</div>', unsafe_allow_html=True)
    assets = list_assets()
    top = st.columns(3)
    summary = asset_summary()
    top[0].metric("Registered Assets", summary["assets"])
    top[1].metric("Open Work Orders", summary["open_tasks"])
    top[2].metric("High-Priority Work Orders", summary["high_priority_tasks"])
    with st.expander("＋ Register a new tower / asset", expanded=not bool(assets)):
        with st.form("asset_registration_form", clear_on_submit=True):
            c1,c2 = st.columns(2)
            asset_code = c1.text_input("Asset / Tower ID", placeholder="e.g. TOWER-001")
            asset_name = c2.text_input("Asset name", placeholder="e.g. Feeder 11 - Tower 24")
            location = st.text_input("Location / route")
            c3,c4,c5 = st.columns(3)
            latitude = c3.number_input("Latitude", value=0.0, format="%.6f")
            longitude = c4.number_input("Longitude", value=0.0, format="%.6f")
            voltage = c5.selectbox("Voltage level", ["Not specified","11 kV","33 kV","66 kV","132 kV","220 kV","400 kV","765 kV"])
            notes = st.text_area("Notes")
            submitted = st.form_submit_button("Save Asset", use_container_width=True)
            if submitted:
                if not asset_code.strip() or not asset_name.strip():
                    st.error("Asset ID and asset name are required.")
                else:
                    try:
                        add_asset(asset_code, asset_name, location,
                                  latitude if latitude != 0 else None,
                                  longitude if longitude != 0 else None,
                                  "" if voltage == "Not specified" else voltage, notes)
                        st.success(f"Asset {asset_code.strip()} registered.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not register asset. Check for a duplicate Asset ID. Details: {exc}")
    if assets:
        st.dataframe(pd.DataFrame(assets)[["asset_code","name","location","voltage_level","latitude","longitude","created_at"]],
                     hide_index=True, use_container_width=True)
        selected_asset = st.selectbox("Select asset to remove", options=[a["id"] for a in assets],
                                      format_func=lambda v: next(f'{a["asset_code"]} — {a["name"]}' for a in assets if a["id"]==v))
        confirm_asset_delete = st.checkbox("Confirm deletion of selected asset")
        if st.button("Delete selected asset", disabled=not confirm_asset_delete):
            delete_asset(selected_asset)
            st.success("Asset deleted. Linked work orders will remain, without an asset link.")
            st.rerun()
    else:
        st.info("No assets registered yet. Add your first tower or infrastructure asset above.")
    st.caption("Asset records are operational metadata. Registering an asset does not automatically associate historical detections with it.")

# ============================================================
# MAINTENANCE CENTER
# ============================================================

elif page == "🛠 Maintenance Center":
    st.markdown('<div class="section-title">Maintenance Work Orders</div>', unsafe_allow_html=True)
    assets = list_assets()
    tasks = list_tasks()
    summary = asset_summary()
    c1,c2,c3 = st.columns(3)
    c1.metric("Open", summary["open_tasks"])
    c2.metric("High Priority", summary["high_priority_tasks"])
    c3.metric("Total Work Orders", len(tasks))
    with st.expander("＋ Create maintenance work order", expanded=not bool(tasks)):
        with st.form("maintenance_task_form", clear_on_submit=True):
            asset_options = {None:"Unassigned asset"}
            asset_options.update({a["id"]:f'{a["asset_code"]} — {a["name"]}' for a in assets})
            asset_id = st.selectbox("Related asset", options=list(asset_options), format_func=lambda v:asset_options[v])
            title = st.text_input("Work order title", placeholder="e.g. Schedule field verification")
            description = st.text_area("Work description / observation")
            c1,c2 = st.columns(2)
            priority = c1.selectbox("Priority", ["Low","Medium","High"], index=1)
            due_date = c2.date_input("Target date").isoformat()
            submitted = st.form_submit_button("Create Work Order", use_container_width=True)
            if submitted:
                if not title.strip():
                    st.error("Work order title is required.")
                else:
                    add_task(asset_id,title,description,priority,due_date)
                    st.success("Work order created.")
                    st.rerun()
    if tasks:
        st.dataframe(pd.DataFrame(tasks)[["id","asset_code","asset_name","title","priority","status","due_date","created_at"]],
                     hide_index=True, use_container_width=True)
        task_id = st.selectbox("Update work order", options=[t["id"] for t in tasks],
                               format_func=lambda v: next(f'#{t["id"]} · {t["title"]}' for t in tasks if t["id"]==v))
        current_task = next(t for t in tasks if t["id"]==task_id)
        new_status = st.selectbox("Work order status", ["Open","In Progress","Completed","Cancelled"],
                                  index=["Open","In Progress","Completed","Cancelled"].index(current_task["status"]))
        if st.button("Update status", use_container_width=True):
            update_task_status(task_id,new_status)
            st.success("Work order status updated.")
            st.rerun()
    else:
        st.info("No work orders yet. Create a task to start tracking maintenance activity.")
    st.warning("Maintenance priority is manually assigned by the operator. AI component detections do not diagnose physical defects or certify electrical safety.")

# ============================================================
# INSPECTION COMPARISON
# ============================================================

elif page == "🔄 Inspection Comparison":
    st.markdown('<div class="section-title">Compare Saved Inspections</div>', unsafe_allow_html=True)
    records = list_inspections(limit=500)
    if len(records) < 2:
        st.info("At least two saved inspections are required for comparison.")
    else:
        labels = {r["id"]:f'#{r["id"]} · {r["created_at"][:19]} · {r["source_filename"]}' for r in records}
        ids = [r["id"] for r in records]
        c1,c2 = st.columns(2)
        first_id = c1.selectbox("Baseline inspection", ids, index=min(1,len(ids)-1), format_func=lambda v:labels[v])
        second_id = c2.selectbox("Newer / comparison inspection", ids, index=0, format_func=lambda v:labels[v])
        if first_id == second_id:
            st.warning("Choose two different inspection records.")
        else:
            first = get_inspection(first_id) or {}
            second = get_inspection(second_id) or {}
            a = first.get("component_counts") or {}
            b = second.get("component_counts") or {}
            all_classes = sorted(set(a)|set(b))
            comp = pd.DataFrame([{"Component":k,"Baseline":int(a.get(k,0)),"Comparison":int(b.get(k,0)),
                                  "Change":int(b.get(k,0))-int(a.get(k,0))} for k in all_classes])
            metrics = st.columns(3)
            metrics[0].metric("Baseline detections", int(first.get("detection_count",0)))
            metrics[1].metric("Comparison detections", int(second.get("detection_count",0)))
            metrics[2].metric("Detection count change", int(second.get("detection_count",0))-int(first.get("detection_count",0)))
            st.dataframe(comp, hide_index=True, use_container_width=True)
            if not comp.empty:
                st.bar_chart(comp.set_index("Component")[["Baseline","Comparison"]], use_container_width=True)
            st.caption("This compares model detection counts only. Differences can result from camera angle, image quality, confidence threshold or sampling; they are not proof of component damage or disappearance.")

# ============================================================
# SYSTEM MONITORING
# ============================================================

elif page == "🖥 System Monitoring":
    st.markdown('<div class="section-title">Runtime & Model Monitoring</div>', unsafe_allow_html=True)
    records = list_inspections(limit=1000)
    total = len(records)
    avg_time = sum(float(r.get("processing_seconds") or 0) for r in records)/total if total else 0
    avg_detections = sum(int(r.get("detection_count") or 0) for r in records)/total if total else 0
    gpu_available = bool(torch.cuda.is_available())
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Compute Device", "CUDA GPU" if gpu_available else "CPU")
    c2.metric("GPU", torch.cuda.get_device_name(0) if gpu_available else "Not detected")
    c3.metric("Avg. processing / record", f"{avg_time:.2f}s")
    c4.metric("Avg. detections / record", f"{avg_detections:.1f}")
    if gpu_available:
        try:
            props = torch.cuda.get_device_properties(0)
            allocated = torch.cuda.memory_allocated(0)/(1024**3)
            reserved = torch.cuda.memory_reserved(0)/(1024**3)
            g1,g2,g3 = st.columns(3)
            g1.metric("GPU memory capacity", f"{props.total_memory/(1024**3):.1f} GB")
            g2.metric("Allocated memory", f"{allocated:.2f} GB")
            g3.metric("Reserved memory", f"{reserved:.2f} GB")
        except Exception as exc:
            st.caption(f"GPU memory details unavailable: {exc}")
    else:
        st.info("CUDA is not available in the current runtime. Inference will use the configured CPU device.")
    st.markdown("#### Low-confidence detections for operator review")
    low_conf = []
    for record in records:
        for detection in (record.get("detections") or []):
            try:
                confidence = float(detection.get("confidence",0))
            except (TypeError,ValueError):
                continue
            if confidence < 0.50:
                low_conf.append({"Inspection ID":record["id"],"Date":record["created_at"],
                                 "File":record["source_filename"],"Component":detection.get("component","Unknown"),
                                 "Confidence":round(confidence,3)})
    if low_conf:
        st.dataframe(pd.DataFrame(low_conf).sort_values("Confidence"), hide_index=True, use_container_width=True)
    else:
        st.success("No saved detections below 0.50 confidence were found in the inspected records.")
    st.caption("Runtime metrics are measured from the current application session and saved inspection records. Low confidence is a review signal, not a defect classification.")


elif page == "📊 Model Performance":



    html("""

    <div class="hero">

        <div class="hero-content">

            <div class="hero-kicker">MODEL ANALYTICS</div>

            <div class="hero-title">

                Model <span>Performance</span>

            </div>

            <div class="hero-copy">

                Validation metrics and training behaviour of

                the PowerLine_AI Faster R-CNN detector.

            </div>

        </div>

    </div>

    """)



    st.markdown(

        '<div class="section-title">Validation Metrics</div>',

        unsafe_allow_html=True

    )



    a,b,c,d = st.columns(4)



    for col, value, label in [

        (a, MAP50, "mAP @ 50"),

        (b, MAP5095, "mAP @ 50:95"),

        (c, MAP75, "mAP @ 75"),

        (d, MAR100, "mAR @ 100"),

    ]:

        with col:

            html(f"""

            <div class="metric">

                <div class="metric-icon">▥</div>

                <div>

                    <div class="metric-value">

                        {value*100:.2f}%

                    </div>

                    <div class="metric-label">

                        {label}

                    </div>

                </div>

            </div>

            """)



    st.markdown(

        '<div class="section-title">Training Progress</div>',

        unsafe_allow_html=True

    )



    loss_df = pd.DataFrame({

        "Epoch": range(1,11),

        "Training Loss": TRAIN_LOSSES

    })



    st.line_chart(

        loss_df.set_index("Epoch"),

        height=350

    )



    st.markdown(

        '<div class="section-title">Per-Class mAP</div>',

        unsafe_allow_html=True

    )



    class_df = pd.DataFrame({

        "Component": list(PER_CLASS_MAP.keys()),

        "mAP": [

            v * 100

            for v in PER_CLASS_MAP.values()

        ]

    })



    st.bar_chart(

        class_df.set_index("Component"),

        height=500

    )





# ============================================================

# ABOUT

# ============================================================



elif page == "ℹ️ About Project":



    html("""

    <div class="hero">

        <div class="hero-content">

            <div class="hero-kicker">POWERLINE_AI</div>

            <div class="hero-title">

                About <span>the Project</span>

            </div>

            <div class="hero-copy">

                AI-powered inspection of power-line infrastructure

                using deep-learning object detection.

            </div>

        </div>

    </div>

    """)



    st.markdown(

        '<div class="section-title">Technology Stack</div>',

        unsafe_allow_html=True

    )



    stack = pd.DataFrame({

        "Component": [

            "Language",

            "Framework",

            "Architecture",

            "Backbone",

            "Dataset",

            "Frontend",

            "Hardware"

        ],

        "Technology": [

            "Python",

            "PyTorch + Torchvision",

            "Faster R-CNN",

            "ResNet-50 + FPN",

            "InsPLAD",

            "Streamlit",

            "NVIDIA RTX 2050"

        ]

    })



    st.dataframe(

        stack,

        hide_index=True,

        width="stretch"

    )



    st.markdown(

        '<div class="section-title">System Workflow</div>',

        unsafe_allow_html=True

    )



    workflow = [

        ("01","Input","UAV inspection image"),

        ("02","Preprocess","Image normalization"),

        ("03","Features","ResNet-50 + FPN"),

        ("04","Detect","Faster R-CNN"),

        ("05","Classify","18 component classes"),

        ("06","Output","Boxes + confidence"),

    ]



    cols = st.columns(6)



    for col, (n,t,d) in zip(cols, workflow):

        with col:

            html(f"""

            <div class="metric"

                 style="display:block;min-height:130px;">

                <div style="

                    color:#1598ff;

                    font-size:11px;

                    font-weight:850;

                ">{n}</div>

                <div style="

                    margin-top:10px;

                    font-size:14px;

                    font-weight:800;

                ">{t}</div>

                <div style="

                    margin-top:7px;

                    color:#8197ad;

                    font-size:10px;

                ">{d}</div>

            </div>

            """)



html("""
<div class="footer">
    ⚡ <strong>PowerLine_AI</strong> &nbsp;•&nbsp; AI-Powered Power-Line Component Inspection
    <span style="display:block;margin-top:5px;">Intelligent infrastructure inspection · 2026</span>
</div>
""")
