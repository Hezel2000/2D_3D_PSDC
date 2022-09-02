#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import TMP_MAX
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd

##-----------------------------------------------------------------------------##
##--- Fitting a distribution to the 2D chd and calculating their mu & sigma ---##
##-----------------------------------------------------------------------------##


def calcMuSigma(sample):
    shape, loc, scale = stats.lognorm.fit(
        sample, floc=0)  # hold location to 0 while fitting
    m, s = np.log(scale), shape  # mu, sigma
    return (round(m, 3), round(s, 3))


#---------------------------------#
#------ Fitting  -----------------#
#---------------------------------#

st.write('Producing a log-normal fit from uploaded data')


uploaded_file = st.file_uploader('')
if uploaded_file is not None:
    st.session_state.dfRaw = pd.read_excel(uploaded_file)

sel2D3DData = st.session_state.dfRaw.columns.drop_duplicates().tolist()

col1, col2 = st.columns(2)
with col1:
    st.session_state.sel2DData = st.multiselect('Select 2D data', sel2D3DData)
with col2:
    st.session_state.sel3DData = st.multiselect('Select 3D data', sel2D3DData)

st.table(np.array([st.session_state.sel2DData,
         st.session_state.sel3DData]).tolist())

setMaxChdDia = st.number_input(
    'Set the maximum chondrule diameter', value=2000)


nrOfPlots = len(st.session_state.sel2DData)

fig = plt.figure(figsize=(5, 4 * nrOfPlots), dpi=150)

for i in range(nrOfPlots):

    extract2D = st.session_state.sel2DData[i]
    extract3D = st.session_state.sel3DData[i]

    mu2D, sigma2D = calcMuSigma(
        st.session_state.dfRaw[extract2D][st.session_state.dfRaw[extract2D] < setMaxChdDia].dropna().tolist())

    mu3D, sigma3D = calcMuSigma(
        st.session_state.dfRaw[extract3D][st.session_state.dfRaw[extract3D] < setMaxChdDia].dropna().tolist())

    plotData2D = st.session_state.dfRaw[extract2D][st.session_state.dfRaw[extract2D]
                                                   < setMaxChdDia].dropna().tolist()
    plotData3D = st.session_state.dfRaw[extract3D][st.session_state.dfRaw[extract3D]
                                                   < setMaxChdDia].dropna().tolist()

    meteorite = extract2D.split('-')[0]

    xpdf = np.linspace(.000001, 2500, 1000)
    y2Dpdf = (1/(xpdf*sigma2D*(2*np.pi)**.5)) * \
        np.e**(-(np.log(xpdf)-mu2D)**2/(2*sigma2D**2))
    y3Dpdf = (1/(xpdf*sigma3D*(2*np.pi)**.5)) * \
        np.e**(-(np.log(xpdf)-mu3D)**2/(2*sigma3D**2))

    ax = fig.add_subplot(nrOfPlots, 1, i + 1)
    ax.hist(plotData3D, 50, density=True, alpha=.4, label='3D distribution')
    ax.hist(plotData2D, 50, density=True, alpha=.4, label='2D distribution')
    ax.plot(xpdf, y3Dpdf, color='Blue', label='3D calc. distribution')
    ax.plot(xpdf, y2Dpdf, color='Red', label='2D calc. distribution')

    ax.vlines(np.e**(mu3D + 0.5 * sigma3D**2), 0, .0025,
              linestyles='dashed', colors='blue')
    ax.vlines(np.e**(mu2D + 0.5 * sigma2D**2), 0, .0025,
              linestyles='dashed', colors='red')

    ax.text(.02, .93, meteorite, horizontalalignment='left',
            transform=ax.transAxes)
    ax.set_xlabel('chd diameter')
    ax.set_ylabel('frequency')
    ax.legend()
    ax.set_xlim(0, 2000)

    textBox2D = '\n'.join((
        '2D', r'$\sigma=%.3f$' % (sigma2D, ), r'$\mu=%.3f$' % (mu2D, ), r'$\mathrm{mean}=%.0f$' % (np.e**(mu2D + .5 * sigma2D**2)), r'$\mathrm{med}=%.0f$' % (np.e**mu2D, ), r'$\mathrm{mod}=%.0f$' % (np.e**(mu2D - sigma2D**2), )))
    textBox3D = '\n'.join((
        '3D', r'$\sigma=%.3f$' % (sigma3D, ), r'$\mu=%.3f$' % (mu3D, ), r'$\mathrm{mean}=%.0f$' % (np.e**(mu3D + .5 * sigma3D**2), ), r'$\mathrm{med}=%.0f$' % (np.e**mu3D, ), r'$\mathrm{mod}=%.0f$' % (np.e**(mu3D - sigma3D**2), )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.503, 0.61, textBox2D, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    ax.text(0.762, 0.61, textBox3D, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)

st.pyplot(fig)


dataParameters = []
for i in st.session_state.dfRaw.columns:
    ms = calcMuSigma(
        st.session_state.dfRaw[i][st.session_state.dfRaw[i] < setMaxChdDia].dropna().tolist())
    dataParameters.append([i, ms[0], ms[1]])


st.table(dataParameters)

st.write(st.session_state.dfRaw)
