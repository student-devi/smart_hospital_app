import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Smart Hospital Patient Navigator", page_icon="🏥", layout="wide")

@st.cache_resource
def load_model():
    with open('hospital_model.pkl', 'rb') as f:
        return pickle.load(f)

bundle        = load_model()
model         = bundle['model']
scaler        = bundle['scaler']
features      = bundle['features']
cols_to_scale = bundle['cols_to_scale']
dept_map_inv  = bundle['dept_map_inv']
gender_map    = bundle['gender_map']
temp_map      = bundle['temp_map']
hr_map        = bundle['hr_map']
dur_map       = bundle['dur_map']
cc_map        = bundle['cc_map']

DEPT_INFO = {
    'Respiratory Medicine': {
        'icon':'🫁','color':'#0284c7','bg':'#e0f2fe','border':'#7dd3fc',
        'desc':'Specialises in conditions affecting the lungs and airways.',
        'next':['Visit Level 2, Wing B','Estimated wait: 15–25 min','Please wear a mask']
    },
    'Cardiology': {
        'icon':'❤️','color':'#dc2626','bg':'#fee2e2','border':'#fca5a5',
        'desc':'Specialises in heart and cardiovascular conditions.',
        'next':['Visit Level 3, Wing A','Estimated wait: 20–30 min','Bring any previous ECG reports']
    },
    'Gastroenterology': {
        'icon':'🫃','color':'#d97706','bg':'#fef3c7','border':'#fcd34d',
        'desc':'Specialises in digestive system and abdominal conditions.',
        'next':['Visit Level 1, Wing C','Estimated wait: 10–20 min','Avoid eating before consultation']
    },
    'Neurology': {
        'icon':'🧠','color':'#7c3aed','bg':'#ede9fe','border':'#c4b5fd',
        'desc':'Specialises in brain, spine, and nervous system conditions.',
        'next':['Visit Level 4, Wing A','Estimated wait: 25–35 min','Bring list of current medications']
    },
    'General Medicine': {
        'icon':'🩺','color':'#059669','bg':'#d1fae5','border':'#6ee7b7',
        'desc':'Handles general health concerns and non-specialist conditions.',
        'next':['Visit Level 1, Wing A','Estimated wait: 10–15 min','Registration desk is open 24/7']
    },
    'Dermatology': {
        'icon':'🔬','color':'#b45309','bg':'#fef9c3','border':'#fde68a',
        'desc':'Specialises in skin, hair, and nail conditions.',
        'next':['Visit Level 2, Wing D','Estimated wait: 15–20 min','Bring photos of affected area if possible']
    },
}

st.markdown("""
<div style="background-color: #1e3a8a; text-align: center;">
    <div style="color: #ffffff; font-size: 20px;">
        Smart Hospital Patient Navigator
    </div>
</div>
""", unsafe_allow_html=True)

with st.form("triage_form"):
    st.markdown("""
    <div style="background:#f0f9ff; border:1px solid #bae6fd;">
        <div style="display:flex; align-items:center;">
            <span style="color: #1e3a8a;">1</span>
            <span style="color: #1e3a8a;">What are your main symptoms?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        fever = st.checkbox("Fever")
        cough = st.checkbox("Cough")
    with c2:
        headache = st.checkbox("Headache")
        chest_pain = st.checkbox("Chest Pain")
    with c3:
        stomach_pain = st.checkbox("Stomach Pain")
        shortness_breath = st.checkbox("Shortness Breath")
    with c4:
        nausea_vomitting = st.checkbox("Nausea Vommiting")
        dizziness = st.checkbox("Dizziness")
    c5, _, _, _ = st.columns(4)
    with c5:
        skin_rash = st.checkbox("Skin Rash")
    st.markdown("""
    <div style="background:#fdf4ff;border:1px solid #e9d5ff;border-radius:14px;
                padding:20px 24px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="background:#7c3aed;color:white;border-radius:8px;
                         padding:4px 10px;font-size:12px;font-weight:600;">2</span>
            <span style="font-size:16px;font-weight:600;color:#3b0764;">How long have you had these symptoms?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_cc, col_dur = st.columns(2)
    with col_cc:
        chief_complaint = st.selectbox("Chief complaint", options=list(cc_map.keys()))
    with col_dur:
        duration = st.selectbox("Duration", options=list(dur_map.keys()), index=1)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:14px;
                padding:20px 24px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="background:#ea580c;color:white;border-radius:8px;
                         padding:4px 10px;font-size:12px;font-weight:600;">3</span>
            <span style="font-size:16px;font-weight:600;color:#7c2d12;">How would you rate the severity?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_temp, col_hr = st.columns(2)
    with col_temp:
        temperature_level = st.selectbox("Temperature", options=list(temp_map.keys()), index=1)
    with col_hr:
        heart_rate_level  = st.selectbox("Heart rate", options=list(hr_map.keys()), index=1)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:14px;
                padding:20px 24px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <span style="background:#059669;color:white;border-radius:8px;
                         padding:4px 10px;font-size:12px;font-weight:600;">4</span>
            <span style="font-size:16px;font-weight:600;color:#064e3b;">Do you have any of the following?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ch1, ch2, ch3, _ = st.columns(4)
    with ch1: hypertension  = st.checkbox("🩺 High Blood Pressure")
    with ch2: heart_disease = st.checkbox("❤️ Heart Disease")
    with ch3: asthma        = st.checkbox("💨 Asthma")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;
                padding:20px 24px;margin-bottom:24px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="background:#475569;color:white;border-radius:8px;
                         padding:4px 10px;font-size:12px;font-weight:600;">5</span>
            <span style="font-size:16px;font-weight:600;color:#1e293b;">Patient Information</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_age, col_gen = st.columns(2)
    with col_age:
        age    = st.number_input("Age", min_value=1, max_value=120, value=35)
    with col_gen:
        gender = st.selectbox("Gender", options=['Female', 'Male'])

    submitted = st.form_submit_button("Get AI Recommendation →")
    


