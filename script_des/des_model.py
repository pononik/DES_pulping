#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 10:56:11 2026

@author: ponoman1
"""
# Modeling of deep eutectic solvent (DES) pulping.
# According to literature data Smink2020 and Perez2025.
# Nikolai P. Ponomarev
# Use 'process_diagram_des.pdf' for numbering of streams.
#%% Libraries
import pandas as pd
import numpy as np
from pathlib import Path
import math
import matplotlib.pyplot as plt
import random, numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy.stats import linregress
import olca_ipc as ipc
import olca_schema as o
import olca_ipc.utree as utree
import time
import warnings
###
from ax.service.managed_loop import optimize # for machine learning, takes time to upload to kernel
###
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import random as rand
import statistics as stats
from scipy.stats import norm
from PIL import Image
#%% Given data
yP = 0.57 # (%) Pulp Yield [40-70%] Smink2020
ltow = 10 # liqour to wood ratio
lig = 0.271 # lignin content: hardwood 18-25%, softwood 25-33%
HBD = 10 # hydrogen bond donor: Lactic Acid (LA) (parts of ratio) (Perez2025) HBD:HBA = 10:1
HBA = 1 # hydrogen bond acceptor: Choline Chloride (ChCl) or NaCl (parts of ratio) (Perez2025) HBD:HBA = 10:1
la = (ltow * HBD) / (HBD + HBA) #g/g wood HBD
chcl = (ltow * HBA) / (HBD + HBA) # g/gwood HBA (Scenario1)
nacl = (ltow * HBA) / (HBD + HBA) # g/gwood HBA (Scenario2)
des = la + chcl # total amount of DES that equals actually ltow
#hemi = 0.25 # hemicelluloses content: hardwood 20-30%, softwood 20-35%
WoodH2O = 0.45 # water content of wood is 45% (Gustafson2011, p.279)[40-50%]
sw_wood_logs = 0.5 #specific weight of wooden logs [t/m3]
PulpH2O = 0.55 # assumed high consistency pulp 45%, i.e. water 55%. Do not increase it.
Pulp_dried = 0.12 # assumed water content in the resultant pulp (%) 
LigH2O = 0.4 # assumed water content in the resultant lignin 40%
Loss_DES = 0.05 # assumed 5% losses of DES [0-100%]
Loss_H2O = 0.05 # assumed 10% losses of H2O [0-100%]
R = 8.3145 # (J / mol * K), gas constant 
MCO2 = 44 # molar mass of carbon dioxide (g/mol) 
MC = 12 # molar mass of carbon g/mol
M_LA = 90.08 # g/mol
M_ChCl = 139.62 # g/mol
M_NaCl = 58.44 # g/mol 
Head = 50 # meters, head of pump for ultrafiltration, i.e. 5 bar (Servaes2017)
g_981 = 9.81 # m/s2
eff_pump = 0.7 # efficiency of the pump and motor, % [0.5-0.7]
eff_boiler = 0.8 # boiler efficiency, % (85-90%)
ED = 0.75 # Evaporation degree of MVR evaporator, % [0.5-0.9]
SP = 0.9 # Separation performance of the membrane, % (Servaes2017) [0.75-0.94]
Tdl = 130 # temperature of delignification 120-180, degrees Celsius
Tdef = 125 # defibration stage temperature 160-170, degrees Celsius
Tcinf = 10 # temperature of cold influent, degress Celcius
Thinf = 80 # temperature of hot influent, degress Celcius
t1 = 4.5 # delignification time in hours, 2-4.5 hours
Edef = 0.050 # energy for defibration at consistency 30-35%, MWh/tp, i.e. product pulp. (Gustafson2011, p.252)  
Ewh = 0.045 # electricuty for wood handling (FRAM2011)
Emvr_per_tH2O = 0.012 # energy for MVR [MWh/tH2Oevap] (Parviainen2008, P = 2.2 * dT, dT = 5-10 C)
heat_loss = 0.04 # 4% heat losses (Gustafson2011)
C_H2O = 4.19 # heat capacity of water (kJ/C*kg)
C_w = 1.53 # heat capacity of wood (kJ/C*kg)
C_des = 2.53 # heat capacity of DES (kJ/kg*oC) (Smink2020)
C_la = 2.5 #heat capacity of LA [2.4-2.6] (kJ/kg*oC)
C_chcl = 1.2 # ChCl (kJ/kg*oC)
C_nacl = 0.9 # NaCl (kJ/kg*oC)
hev = 2256 # Enthalpy of vaporization of water, i.e latent heat (kJ/kg)
biomass_cal = 11 # lignin or wood LHV (MJ/kg) [FRAM11]
Kb = 0.512 # ebullioscopic constant (boiling-point elevation constant) of water (i.e. solvent) (K*kg/mol) 
wood_carbon = 0.5 # Stoichiometric carbon content in wood (%)
# Creating a dataframe
data_given_des = [
    ["Description", "Value", "Variation limit", "Unit", "Variable"],
    ["Pulp yield (fraction)", yP, "[0.40-0.70]", "-", "yP"],
    ["Liquor to wood ratio", ltow, None, "g/g", "ltow"],
    ["Lignin content (fraction)", lig, "HW[0.18-0.25]; SW[0.25-0.33]", "-", "lig"],
    ["Hydrogen bond donor parts (LA)", HBD, "HBD:HBA=10:1", "-", "HBD"],
    ["Hydrogen bond acceptor parts (ChCl/NaCl)", HBA, "HBD:HBA=10:1", "-", "HBA"],
    ["Lactic acid dose per wood", la, None, "g/g wood", "la"],
    ["Choline chloride dose per wood", chcl, None, "g/g wood", "chcl"],
    ["Sodium chloride dose per wood", nacl, None, "g/g wood", "nacl"],
    ["Water content of fresh wood (fraction)", WoodH2O, "[0.40-0.50]", "-", "WoodH2O"],
    ["Specific weight of wood logs", sw_wood_logs, None, "t/m^3", "sw_wood_logs"],
    ["Pulp water content (fraction)", PulpH2O, "-", "-", "PulpH2O"],
    ["Pulp dried water content (fraction)", Pulp_dried, None, "-", "Pulp_dried"],
    ["Lignin water content (fraction)", LigH2O, "[0.70-0.80]", "-", "LigH2O"],
    ["DES loss (fraction)", Loss_DES, "[0-1.0]", "-", "Loss_DES"],
    ["Water loss (fraction)", Loss_H2O, "[0-1.0]", "-", "Loss_H2O"],
    ["Gas constant", R, None, "J/mol*K", "R"],
    ["Molar mass CO2", MCO2, None, "g/mol", "MCO2"],
    ["Molar mass C", MC, None, "g/mol", "MC"],
    ["Molar mass LA", M_LA, None, "g/mol", "M_LA"],
    ["Molar mass ChCl", M_ChCl, None, "g/mol", "M_ChCl"],
    ["Molar mass NaCl", M_NaCl, None, "g/mol", "M_NaCl"],
    ["Pump head", Head, None, "m", "Head"],
    ["Gravitational acceleration", g_981, None, "m/s^2", "g"],
    ["Pump+motor efficiency", eff_pump, "[0.5-0.7]", "-", "eff_pump"],
    ["Boiler efficiency", eff_boiler, "[0.85-0.90]", "-", "eff_boiler"],
    ["Evaporation degree (MVR)", ED, "[0.5-0.9]", "-", "ED"],
    ["Membrane separation performance", SP, "[0.75-0.94]", "-", "SP"],
    ["Delignification temperature", Tdl, "[120-180]", "C", "Tdl"],
    ["Defibration temperature", Tdef, "[160-170]", "C", "Tdef"],
    ["Cold influent temperature", Tcinf, None, "C", "Tcinf"],
    ["Hot influent temperature", Thinf, None, "C", "Thinf"],
    ["Delignification time", t1, "[2-4.5]", "h", "t1"],
    ["Defibration energy", Edef, None, "MWh/tp", "Edef"],
    ["Electricity for wood handling", Ewh, None, "MWh/t", "Ewh"],
    ["MVR energy per t H2O evaporated", Emvr_per_tH2O, None, "MWh/tH2O_evap", "Emvr_per_tH2O"],
    ["Heat loss (fraction)", heat_loss, None, "-", "heat_loss"],
    ["Heat capacity of water", C_H2O, None, "kJ/kg*C", "C_H2O"],
    ["Heat capacity of wood", C_w, None, "kJ/kg*C", "C_w"],
    ["Heat capacity of DES", C_des, None, "kJ/kg*C", "C_des"],
    ["Heat capacity of LA", C_la, "[2.4-2.6]", "kJ/kg*C", "C_la"],
    ["Heat capacity of ChCl", C_chcl, None, "kJ/kg*C", "C_chcl"],
    ["Heat capacity of NaCl", C_nacl, None, "kJ/kg*C", "C_nacl"],
    ["Enthalpy of vaporization of water", hev, None, "kJ/kg", "hev"],
    ["Biomass LHV", biomass_cal, None, "MJ/kg", "biomass_cal"],
    ["Ebullioscopic constant of water", Kb, None, "K*kg/mol", "Kb"],
    ["Stoichiometric carbon in wood (fraction)", wood_carbon, None, "-", "wood_carbon"],
]
# Build DataFrame using the header row
df_gd_des = pd.DataFrame(data_given_des)#(rows, columns=columns)
#%% Creating one master function to unite all functions
# Make indent for the minor functions to make working DES master function
def DES(gd):
    # Unpack the given data
    ltow      = gd["ltow"]        # liquor-to-wood ratio (g/g)
    HBD       = gd["HBD"]         # HBD parts (e.g., LA)
    HBA       = gd["HBA"]         # HBA parts (e.g., ChCl or NaCl)
    la        = gd["la"]          # lactic acid dose per wood (g/g wood)
    chcl      = gd["chcl"]        # choline chloride dose per wood (g/g wood)
    #nacl      = gd["nacl"]        # sodium chloride dose per wood (g/g wood)
    lig       = gd["lig"]         # lignin fraction in wood (dry)
    yP        = gd["yP"]          # pulp yield (fraction)
    WoodH2O   = gd["WoodH2O"]     # water fraction in fresh wood
    PulpH2O   = gd["PulpH2O"]     # water fraction in pulp (high-consistency)
    Pulp_dried= gd["Pulp_dried"]  # water fraction of dried pulp
    LigH2O    = gd["LigH2O"]      # water fraction in separated lignin
    Tdl       = gd["Tdl"]         # delignification temperature (C)
    #Tdef      = gd["Tdef"]        # defibration temperature (C)
    Tcinf     = gd["Tcinf"]       # cold influent temperature (C)
    Thinf     = gd["Thinf"]       # hot influent temperature (C)
    t1        = gd["t1"]          # delignification time (h)
    C_H2O     = gd["C_H2O"]       # kJ/kg*C
    C_w       = gd["C_w"]         # kJ/kg*C
    C_des     = gd["C_des"]       # kJ/kg*C (bulk DES)
    #C_la      = gd["C_la"]        # kJ/kg*C
    #C_chcl    = gd["C_chcl"]      # kJ/kg*C
    #C_nacl    = gd["C_nacl"]      # kJ/kg*C
    heat_loss = gd["heat_loss"]   # fraction
    Ewh       = gd["Ewh"]         # MWh/t (wood handling)
    Edef      = gd["Edef"]        # MWh/tp (defibration)
    ED        = gd["ED"]          # evaporation degree (MVR)
    SP        = gd["SP"]          # membrane separation performance
    hev       = gd["hev"]         # kJ/kg (latent heat of water)
    Loss_H2O  = gd["Loss_H2O"]    # fraction
    Loss_DES  = gd["Loss_DES"]    # fraction (replaces Loss_EG)
    Head      = gd["Head"]        # m
    g_981         = gd["g_981"]           # m/s^2
    eff_pump  = gd["eff_pump"]    # -
    eff_boiler= gd["eff_boiler"]  # -
    sw_wood_logs = gd["sw_wood_logs"] # t/m3
    #R         = gd["R"]           # J/mol*K
    MC        = gd["MC"]          # g/mol (carbon)
    M_LA      = gd["M_LA"]        # g/mol
    #M_ChCl    = gd["M_ChCl"]      # g/mol
    #M_NaCl    = gd["M_NaCl"]      # g/mol
    Kb        = gd["Kb"]          # K*kg/mol
    biomass_cal = gd["biomass_cal"]  # MJ/kg
    wood_carbon = gd["wood_carbon"]  # fraction
    def DL(lig, yP, la, chcl, des, ltow,HBD, HBA, WoodH2O, PulpH2O, Tdl, Tcinf, Thinf, C_H2O, C_w,C_des, heat_loss,Ewh,Edef):
        """
        Compute mass balance for the Delignification (DL) stage.
        Including Defibration (DF) and Solid-Liquid Separation (SLS)
        Returns:
          - rounded dataframe of streams
          - dict with key totals (inf_DL, eff_DL)
        """
        # Convenience terms
        carb = 1 - lig                  # carbohydrates in wood
        wood11 = 1 / yP                 # total dry wood needed for process (carb+lig = 1)
        la = (ltow * HBD) / (HBD + HBA) #g/g wood HBD
        chcl = (ltow * HBA) / (HBD + HBA) # g/gwood HBA (Scenario1)
        # nacl = (ltow * HBA) / (HBD + HBA) # g/gwood HBA (Scenario2)
        des = la + chcl # total amount of DES that equals actually ltow
        # Stream 1.1 (influent from wood)
        carb11 = carb / yP
        lig11  = lig / yP
        h2o11  = (wood11 * WoodH2O) / (1 - WoodH2O)
        tot11  = wood11 + h2o11
        # Stream 2.1 (chemicals/water make-up)
        des21  = wood11 * des # amount of des
        # h2o21 = h2o11
        tot21 = des21# + h2o21
        inf_DL = tot11 + tot21 #+ tot22
        # Effluent
        carb12 = (carb11 + lig11) * yP * carb
        lig12  = (carb11 + lig11) * yP * lig
        h2o12  = ((carb12 + lig12) * PulpH2O) / (1 - PulpH2O)
        pulpDL   = carb12 + lig12
        tot12  = pulpDL + h2o12
        # Streams 2.2 and 2.3 do not exist in this process, but make minimal changes to script numbering has not changed
        # Stream 2.4 (liquor)
        carb24 = (carb / yP) - ((carb11 + lig11) * yP * carb) #carb11 - carb12: 1st term in brackets - 2nd term in brackets
        lig24  = (lig / yP) - ((carb11 + lig11) * yP * lig) #lig11 - lig12: 1st term in brackets - 2nd term in brackets
        h2o24  = h2o11 - h2o12
        des24   = des21
        tot24  = carb24 + lig24 + h2o24 + des24
        # Total effluent
        eff_DL = tot12 + tot24 # + tot23
        # Build dataframe (use NaN for missing to keep numeric dtype)
        data_mass_bal_DL = {
            "stream": ["1.1", "1.2", "2.1", "2.2", "2.3", "2.4"],
            "carbs":  [carb11, carb12, np.nan, np.nan, np.nan, carb24],
            "lignin": [lig11,  lig12,  np.nan, np.nan, np.nan, lig24],
            "H2O":    [h2o11,  h2o12,  np.nan,  np.nan, np.nan, h2o24],
            "DES":     [np.nan, np.nan, des21,   np.nan, np.nan, des24],
            "CO2":    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "inf":    [inf_DL, np.nan, np.nan, np.nan, np.nan, np.nan],
            "eff":    [eff_DL, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
        mass_bal_DL = pd.DataFrame(data_mass_bal_DL)
        # Round numeric values to 3 decimals
        df_mat_bal_DL = mass_bal_DL.round(3)
        """
        Compute heat balance in MJ/t or GJ/t for the Delignification (DL) stage.
        Sensible heat, latent heat and heat losses.
        """
        #Sensible heat
        Q_sens_DL = (
                C_H2O * h2o11 * (Tdl - Tcinf) + 
                C_w * wood11 * (Tdl - Tcinf) + 
                C_des * des21 * (Tdl - Thinf)
                )
        #Latent heat i.e. heat of pulping accrding to Courchene2005
        k1 = math.exp(43.2 - (16115 / (Tdl + 273.15)))  # Arrenius type constant "k", "e" means energy
        H1 = k1 * (t1 - 0) # H factor
        if H1 < 100:
            Y1 = 144 * 10 ** (-0.00523 * H1) * 4.19  # at H < 100 and/or for the initial phase
        else:
            Y1 = 60.84 * 10 ** (-0.00149 * H1) * 4.19  # at H > 100 and/or for the bulk phase
        Q_lat_DL = Y1 * (carb11 + lig11)
        #Heat losses
        Q_loss_DL = Q_sens_DL * heat_loss
        # Total heat and converted to GJ/t
        Q_DL = (Q_sens_DL + Q_lat_DL + Q_loss_DL) * 1e-3
        data_heat_bal_DL = [
            ["Item", "Value", "Unit"],
            ["Sensible heat", round(Q_sens_DL * 1e-3, 3), "GJ/t"],
            ["Latent heat", round(Q_lat_DL * 1e-3, 3), "GJ/t"],
            ["Heat losses", round(Q_loss_DL * 1e-3, 3), "GJ/t"],
            ["Total heat for DL", round(Q_DL, 3), "GJ/t"]
            ]
        df_heat_bal_DL = pd.DataFrame(data_heat_bal_DL[1:], columns=data_heat_bal_DL[0])
        # Electricity for wood handling
        E_wh = Ewh * 1e+3
        # Electricity for defibration
        E_df = Edef * 1e+3    
        return (df_mat_bal_DL,df_heat_bal_DL,carb24,lig24,h2o24,des24,tot24,tot12,
                h2o12,carb12,lig12,tot21,des21,h2o11,Q_DL,carb11,lig11,
                tot11,wood11,E_wh,E_df)
    #Calling variables from function using parantheses:
    (df_mat_bal_DL, df_heat_bal_DL, carb24, lig24, h2o24, des24, tot24, tot12,
     h2o12, carb12, lig12, tot21, des21, h2o11, Q_DL, carb11, lig11,
     tot11, wood11, E_wh, E_df) = DL(lig, yP, la, chcl, des, ltow, HBD, HBA, WoodH2O, PulpH2O, Tdl, Tcinf, Thinf, C_H2O, C_w, C_des, heat_loss, Ewh, Edef)
    #print("\n Material balance of delignification (DL) (t/t)\n", df_mat_bal_DL)
    #print("\n Heat balance of DL\n", df_heat_bal_DL)
    # print("\n Electrical energy consumption for wood handling (WH) (kWh/t)\n", round(E_wh, 3))
    # print("\n Electrical energy consumption for defibration (DF) (kWh/t)\n", round(E_df, 3))
    #Membrane separation (MEM)
    def MEM(carb24, lig24, h2o24, des24, tot24, SP, LigH2O, Loss_H2O, Loss_DES, g_981, Head, eff_pump):
        """
        Compute mass balance for the Membrane separation (MEM) stage.
        Returns:
          - rounded dataframe of streams
          - energy consumption E_mem
          - key stream values
        """
        # Stream 2.6 (lignin product)
        lig26 = lig24 * SP
        h2o26 = (lig26 * LigH2O) / (1 - LigH2O)
        tot26 = h2o26 + lig26
        # Stream 2.7 (losses)
        lig27 = lig24 * (1 - SP)
        carb27 = carb24 * (1 - SP)
        h2o27 = h2o24 * Loss_H2O
        des27 = des24 * Loss_DES
        tot27 = des27 + h2o27 + lig27 + carb27
        # Stream 2.5 (retentate to next step)
        des25 = des24 - des27
        carb25 = carb24 * SP
        h2o25 = h2o24 - h2o27 - h2o26
        tot25 = des25 + h2o25 + carb25
        inf_MEM = tot24
        eff_MEM = tot26 + tot27 + tot25
        data_mass_bal_MEM = {
            "stream": ["2.4", "2.5", "2.6", "2.7"],
            "carbs":  [carb24,  carb25,  np.nan,  carb27],
            "lignin": [lig24,   np.nan,  lig26,   lig27],
            "H2O":    [h2o24,   h2o25,   h2o26,   h2o27],
            "DES":    [des24,   des25,   np.nan,  des27],
            "CO2":    [np.nan,  np.nan,  np.nan,  np.nan],
            "inf":    [inf_MEM, np.nan,  np.nan,  np.nan],
            "eff":    [eff_MEM, np.nan,  np.nan,  np.nan],
        }
        mass_bal_MEM = pd.DataFrame(data_mass_bal_MEM)
        df_mat_bal_MEM = mass_bal_MEM.round(3)
        # Pumping energy (kWh/t)
        E_mem = tot24 * g_981 * Head * 2.777778e-4 * eff_pump**-1
        return df_mat_bal_MEM, E_mem, h2o25, des25, carb25, tot25, h2o27, h2o26, lig26, carb27, lig27, des27, tot26, tot27
    (df_mat_bal_MEM, E_mem, h2o25, des25, carb25, tot25, h2o27, h2o26, lig26,
     carb27, lig27, des27, tot26, tot27) = MEM(carb24, lig24, h2o24, des24, tot24, SP, LigH2O, Loss_H2O, Loss_DES, g_981, Head, eff_pump)
    #print("\n Material balance of membrane separation (MEM) (t/t)\n", df_mat_bal_MEM)
    # print("\n Electrical energy consumption for MEM (kWh/t)\n", round(E_mem, 3))
    def MVR(h2o25, des25, carb25, ED, M_LA, Kb):
        """
        Mass and energy balance for Mechanical Vapor Recompression (MVR).
        ED: fraction of water evaporated from stream 2.5 to 2.9
        M_DES: molar mass of DES (e.g., MEG)
        Kb: ebullioscopic constant (K·kg/mol)
        des_feed_for_BPE, h2o_feed_for_BPE: composition used to estimate BPE
        """
        # Mass balance
        h2o28 = h2o25 * (1 - ED)
        des28 = des25
        carb28 = carb25
        tot28 = h2o28 + des28 + carb28
        h2o29 = h2o25 * ED
        tot29 = h2o29
        inf_MVR = des25 + h2o25 + carb25  # equals tot25
        eff_MVR = tot28 + tot29
        data_mass_bal_MVR = {
            "stream": ["2.5", "2.8", "2.9"],
            "carbs":  [carb25,  carb28,  np.nan],
            "lignin": [np.nan,  np.nan,  np.nan],
            "H2O":    [h2o25,   h2o28,   h2o29],
            "DES":    [des25,   des28,   np.nan],
            "CO2":    [np.nan,  np.nan,  np.nan],
            "inf":    [inf_MVR, np.nan,  np.nan],
            "eff":    [eff_MVR, np.nan,  np.nan],
        }
        mass_bal_MVR = pd.DataFrame(data_mass_bal_MVR)
        df_mat_bal_MVR = mass_bal_MVR.round(3)
        # BPE and energy estimate
        # Using your original form: m = des / (M_DES * (1-ED) * h2o_feed)
        # Ensure units: des in t, convert to kg with 1e3; h2o in t
        m = des25 / (M_LA * (1 - ED) * h2o25)
        dT = Kb * m
        E_mvr = 2.2 * dT * h2o29  # kWh/t
        return df_mat_bal_MVR, E_mvr, carb28, h2o28, des28, tot28, tot29, h2o29,m, dT
    # Call (use des24, h2o24 for the BPE basis to match your prior logic)
    df_mat_bal_MVR, E_mvr, carb28, h2o28, des28, tot28, tot29, h2o29,m, dT = MVR(h2o25, des25, carb25, ED, M_LA, Kb)
    # print("\n Material balance of mechanical vapor recompression (MVR) (t/t)\n", df_mat_bal_MVR)
    # print("\n Electrical energy consumption for MVR (kWh/t)\n", round(E_mvr, 3))
    # Ultrafiltration (UF)
    def UF(carb28, h2o28, des28, tot28, g, Head, eff_pump):
        # Mass balance
        carb210 = carb28
        h2o210 = h2o28
        tot210 = carb28 + h2o28
        des211 = des28
        tot211 = des211
        inf_UF = tot28
        eff_UF = tot210 + tot211
        # Dataframe
        data_mass_bal_UF = {
            "stream": ["2.8", "2.10", "2.11"],
            "carbs":  [carb28, carb210, np.nan],
            "lignin": [np.nan,  np.nan,  np.nan],
            "H2O":    [h2o28,  h2o210,  np.nan],
            "DES":    [des28,  np.nan,  des211],
            "CO2":    [np.nan, np.nan,  np.nan],
            "inf":    [inf_UF, np.nan,  np.nan],
            "eff":    [eff_UF, np.nan,  np.nan],
        }
        mass_bal_UF = pd.DataFrame(data_mass_bal_UF)
        df_mat_bal_UF = mass_bal_UF.round(3)
        # Energy consumption
        E_uf = tot28 * g * Head * 2.777778e-4 * eff_pump**-1
        return df_mat_bal_UF, E_uf, tot211, des211, h2o210, carb210, tot210
    df_mat_bal_UF, E_uf, tot211, des211, h2o210, carb210, tot210 = UF(carb28, h2o28, des28, tot28, g_981, Head, eff_pump)
    # print("\n Material balance of ultrafiltration (UF)(t/t)\n", df_mat_bal_UF)
    # print("\n Electrical energy consumption for UF (kWh/t)\n", round(E_uf, 3))
    # Drying (DR)
    def DR(h2o12, carb12, lig12):
        # Mass balance
        carb13 = carb12
        lig13 = lig12
        h2o13 = h2o12 * Pulp_dried
        pulpDR = carb13 + lig13
        tot13 = carb13 + lig13 + h2o13
        h2o212 = h2o12 * (1 - Pulp_dried)
        tot212 = h2o212
        inf_DR = tot12
        eff_DR = tot13 + tot212
        # Dataframe
        data_mass_bal_DR = {
            "stream": ["1.2", "1.3", "2.12"],
            "carbs":  [carb12, carb13, np.nan],
            "lignin": [lig12,  lig12,  np.nan],
            "H2O":    [h2o12,  h2o13,  h2o212],
            "DES":    [np.nan, np.nan, np.nan],
            "CO2":    [np.nan, np.nan, np.nan],
            "inf":    [inf_DR, np.nan,  np.nan],
            "eff":    [eff_DR, np.nan,  np.nan],
        }
        mass_bal_DR = pd.DataFrame(data_mass_bal_DR)
        df_mat_bal_DR = mass_bal_DR.round(3)
        # Energy balance
        Q_sens_DR = (
            C_H2O * h2o12 * (100 - Thinf) +
            C_w * (carb12 + lig12) * (100 - Thinf)
        )
        Q_lat_DR = hev * (h2o12 - h2o13)
        Q_loss_DR = (Q_sens_DR + Q_lat_DR) * heat_loss
        Q_DR = (Q_sens_DR + Q_lat_DR + Q_loss_DR) * 1e-3
    
        data_heat_bal_DR = [
            ["Item", "Value", "Unit"],
            ["Sensible heat", round(Q_sens_DR * 1e-3, 3), "GJ/t"],
            ["Latent heat",   round(Q_lat_DR * 1e-3, 3), "GJ/t"],
            ["Heat losses",   round(Q_loss_DR * 1e-3, 3), "GJ/t"],
            ["Total heat for DR", round(Q_DR, 3), "GJ/t"]
        ]
        df_heat_bal_DR = pd.DataFrame(data_heat_bal_DR[1:], columns=data_heat_bal_DR[0])
        return df_mat_bal_DR, df_heat_bal_DR, pulpDR, h2o13, tot212, h2o212, Q_DR, carb13, lig13, tot13
    df_mat_bal_DR, df_heat_bal_DR, pulpDR, h2o13, tot212, h2o212, Q_DR, carb13, lig13, tot13 = DR(h2o12, carb12, lig12)
    #print("\n Material balance of drying (DR) (t/t)\n", df_mat_bal_DR)
    #print("\n Heat balance of DR\n", df_heat_bal_DR)
    # Buffer Tank (BT)
    def BT(h2o29, h2o212, des211, des21,  h2o11, Loss_DES, #h2o13,
           h2o26, h2o27, h2o210, tot29, tot211, tot212, tot21):
        # Mass balance
        des213 = des21 * Loss_DES       # DES make-up
        #h2o213 = h2o13 + h2o26 + h2o27 + h2o210 - h2o11  # Water make-up
        tot213 = des213 #+ h2o213
        inf_BT = tot213 + tot211# + tot212 + tot29
        eff_BT = tot21
        # Data frame
        data_mass_bal_BT = {
            "stream": ["2.1", "2.9", "2.11", "2.12", "2.13"],
            "carbs":  [np.nan, np.nan, np.nan, np.nan, np.nan],
            "lignin": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "H2O":    [np.nan, h2o29,  np.nan,  h2o212, np.nan],
            "DES":    [des21, np.nan, des211,  np.nan,  des213],
            "CO2":    [np.nan, np.nan, np.nan, np.nan, np.nan],
            "inf":    [inf_BT, np.nan, np.nan, np.nan, np.nan],
            "eff":    [eff_BT, np.nan, np.nan, np.nan, np.nan],
        }
        mass_bal_BT = pd.DataFrame(data_mass_bal_BT)
        df_mat_bal_BT = mass_bal_BT.round(3)
        return df_mat_bal_BT,  des213, tot213 #h2o213,
    df_mat_bal_BT, des213, tot213 = BT(h2o29, h2o212, des211, des21, h2o11, Loss_DES, h2o26, h2o27, h2o210, tot29, tot211, tot212, tot21)
    #print("\n Material balance of Buffer Tank (BT) (t/t)\n", df_mat_bal_BT)
    # Boiler (BLR)
    def BLR (lig26, biomass_cal, eff_boiler, heat_loss):
        Q_BLR = lig26 * biomass_cal * eff_boiler * (1 - heat_loss)
        biomass_egco2 = (Q_DL + Q_DR) * 1e+3 / (biomass_cal * eff_boiler * (1 - heat_loss))
        co2_261 = biomass_egco2 * wood_carbon * MCO2 * 1e-3 / MC
        tot261 = co2_261
        return Q_BLR, co2_261, tot261
    Q_BLR, co2_261, tot261 = BLR(lig26, biomass_cal, eff_boiler, heat_loss)
    #print("\nFrom the boiler (BLR) you can get:")
    #print("Heat from lignin (GJ/t) \n", round(Q_BLR, 3))
    #print("CO2 non-fossil while covering entire heat demand (t/t) \n", round(co2_261, 3))
    # Overall material balance of DES pulping
    data_mass_bal_DES = {
        "stream": ["1.1","1.2","1.3","2.1","2.2","2.3","2.4","2.5","2.6","2.6.1","2.7","2.8","2.9","2.1o","2.11","2.12","2.13"],
        "carbs":  [carb11, carb12, carb13, np.nan, np.nan, np.nan, carb24, carb25, np.nan, np.nan, carb27, carb28, np.nan, carb210, np.nan, np.nan, np.nan],
        "lignin": [lig11,  lig12,  lig13,  np.nan, np.nan, np.nan, lig24,  np.nan, lig26,  np.nan, lig27,  np.nan, np.nan, np.nan,  np.nan, np.nan, np.nan],
        "H2O":    [h2o11,  h2o12,  h2o13,  np.nan, np.nan, np.nan, h2o24, h2o25, h2o26, np.nan, h2o27, h2o28, h2o29, h2o210, np.nan, h2o212, np.nan],
        "DES":     [np.nan, np.nan, np.nan, des21,  np.nan, np.nan, des24,  des25,  np.nan, np.nan, des27,  des28,  np.nan, np.nan,  des211, np.nan,  des213],
        "CO2":    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, co2_261, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "total":  [tot11, tot12, tot13, tot21, np.nan, np.nan, tot24, tot25, tot26, tot261, tot27, tot28, tot29, tot210, tot211, tot212, tot213],
        "influent":[(tot11 + tot213)] + [np.nan]*16,
        "effluent":[(tot13 + tot26 + tot27 + tot210 + tot212 + tot29)] + [np.nan]*16,
    }
    mass_bal_DES = pd.DataFrame(data_mass_bal_DES)
    df_mat_bal_DES = mass_bal_DES.round(3).set_index("stream")
    #print("\n Overall material balance of DES pulping (t/t)\n", df_mat_bal_DES)
    #Summirizing total inputs and outputs for the process
    def summary_DES(wood11, sw_wood_logs, des213, Q_DL, Q_DR, E_wh, E_df, E_mem, E_mvr, E_uf,
                    pulpDR, lig26, Q_BLR, carb210, co2_261, tot27): 
        wood_logs_des = wood11 / sw_wood_logs  # Wood logs (m3/t) 
        la_des = des213 * la * 1e-1# Lactic Acid (t/t)
        chcl_des = des213 * chcl * 1e-1 # Choline Chloiride (t/t)                  
        h2o_des = 0                     # Process water (t/t)
        heat_des = Q_DL + Q_DR                 # Heat demand (GJ/t)
        elect_des = E_wh + E_df + E_mem + E_mvr + E_uf  # Electricity demand (kWh/t) 
        pulp_des = pulpDR                      # Produced pulp (t/t)
        lignin_des = lig26                     # Lignin (t/t)
        lignin_heat_des = Q_BLR                # Lignin as heat (GJ/t)
        hemi_des = carb210                     # Hemicelluloses (t/t)   
        co2_non_fos_des = co2_261              # Non-fossil CO2 (t/t)
        wastewater_des = tot27                 # Wastewater (t/t)  
        return (wood_logs_des,
                la_des,
                chcl_des,
                h2o_des,
                heat_des,
                elect_des,
                pulp_des,
                lignin_des,
                co2_non_fos_des,
                wastewater_des,
                lignin_heat_des,
                hemi_des)
    (wood_logs_des,
    la_des,
    chcl_des,
    h2o_des,
    heat_des,
    elect_des,
    pulp_des,
    lignin_des,
    co2_non_fos_des,
    wastewater_des,
    lignin_heat_des,
    hemi_des) = summary_DES(wood11, sw_wood_logs, des213, 
                                   Q_DL, Q_DR, E_wh, E_df, E_mem, E_mvr, E_uf, pulpDR, lig26, Q_BLR, carb210, co2_261, tot27)
    return (
        wood_logs_des,
        la_des,
        chcl_des,
        h2o_des,
        heat_des,
        elect_des,
        pulp_des,
        lignin_des,
        co2_non_fos_des,
        wastewater_des,
        lignin_heat_des,
        hemi_des)#,
        #df_mat_bal_DES,
# used parameters from given data
giv_dat = {
    "ltow": ltow, "la": la, "chcl": chcl, "HBD":HBD, "HBA": HBA,
    "lig": lig, "yP": yP, "WoodH2O": WoodH2O, "PulpH2O": PulpH2O, "MCO2": MCO2, "R": R,
    "Tdl": Tdl, "t1":t1, "Tcinf": Tcinf, "Thinf": Thinf, "C_H2O": C_H2O, "C_w": C_w, "C_des": C_des,
    "heat_loss": heat_loss, "Ewh": Ewh, "Edef": Edef,
    "SP": SP, "LigH2O": LigH2O, "Loss_H2O": Loss_H2O, "Loss_DES": Loss_DES, "g_981": g_981, "Head": Head, "eff_pump": eff_pump,
    "ED": ED, "M_LA": M_LA, "Kb": Kb,
    "Pulp_dried": Pulp_dried, "hev": hev,
    "biomass_cal": biomass_cal, "eff_boiler": eff_boiler,
    "sw_wood_logs": sw_wood_logs, "wood_carbon": wood_carbon, "MC": MC
}
# Using the results from function des (formerly egco2), with CO2 and H2O removed from returns
(wood_logs_des,
la_des,
chcl_des,
h2o_des,
heat_des,
elect_des,
pulp_des,
lignin_des,
co2_non_fos_des,
wastewater_des,
lignin_heat_des,
hemi_des,
) = DES(giv_dat)
# df_mat_bal_DES,
# sum inputs for LCA in one list (without CO2 and H2O streams previously present)
inp_base = (wood_logs_des,
            la_des,
            chcl_des,
            h2o_des,
            heat_des,
            elect_des,
            co2_non_fos_des,
            wastewater_des,
            lignin_heat_des,
            hemi_des)
#%% Data summary
data_summary = {
    "Category": [
        "given data","properties","properties",
        "input", "input", "input", "input", "input", "input",
        "output", "output", "output", "output",
        "waste", "waste"
    ],
    "Item": [
        "Pulp yield","Kappa Number","Fibre length",
        "Wood logs", "Lactic Acid","Choline Chloride", "Process water", "Heat demand", "Electricity demand",
        "Pulp", "Lignin", "Lignin as heat", "Hemicelluloses",
        "Non-fossil CO2", "Wastewater"
    ],
    "Value": [
        yP*100,"110",1.9,
        wood_logs_des, la_des, chcl_des, 0, heat_des, elect_des,
        pulp_des, lignin_des, lignin_heat_des, hemi_des,
        co2_non_fos_des, wastewater_des
    ],
    "Unit": [
        "%","-","mm",
        "m3/t", "t/t","t/t", "t/t", "GJ/t", "kWh/t",
        "t/t", "t/t", "GJ/t", "t/t",
        "t/t", "t/t"
    ],
}
df_summary_des = pd.DataFrame(data_summary).round(3)
#%% SKIP IF NOTHING CHANGED TO OPEN_LCA 
#LCA with OpenLCA using olca-ipc
# For DES
# The main idea is that global parameters (GB) are set in the LCA database,
# and equal 1. Then these GBs are multiplied by corresponding values obtained
# in material and energy balances, and in OVAT part of the program.
# Process should be created in advance in the existing database
# Kraft process is also added for comparison purposes
# Start timing
t0 = time.perf_counter()
print("\nConnecting to openLCA IPC server on port 8080 ...", end=" ")
try:
    # Try to connect
    client = ipc.Client(8080)
    # Light check that the server responds (a simple call)
    _ = client.find(o.ImpactMethod, "EF v3.1")
    print("connected.\n")
except Exception as e:
    warnings.warn(f"Could not connect to openLCA IPC server on port 8080: {e}")
    # Stop here if not connected
    elapsed = time.perf_counter() - t0
    total_seconds = int(elapsed)
    centi = int((elapsed - total_seconds) * 100)
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    print(f"Execution aborted. Elapsed time: {h:02d}:{m:02d}:{s:02d}:{centi:02d}")
    raise SystemExit(1)
# Find process of DES
# Adjust the name string to match the process name in your DB
model = client.find(o.Process, "DES_v2.0")
# To get kraft pulping
model_kraft = client.find(
    o.Process,
    "sulfate pulp production, from softwood, unbleached | sulfate pulp, unbleached | Cutoff, U"
)
# Create a method for both processes
method = client.find(o.ImpactMethod, "EF v3.1")  # or 'EF v3.1 (CO2only)' if available
# To get global parameters
params = client.get_all(o.Parameter)
def g(name):
    for p in params:
        if p.name == name and getattr(p, "context", None) is None:
            return float(p.value)
    raise KeyError(f"Global parameter '{name}' not found")
# Recalculated values (DES uses LA and ChCl; CO2 and H2O removed)
des_wood_r         = g("des_wood") * wood_logs_des
des_la_r           = g("des_la") * la_des
des_chcl_r         = g("des_chcl") * chcl_des
des_h2o_r = g("des_h2o") * h2o_des
des_heat_r         = g("des_heat") * heat_des
des_elect_r        = g("des_elect") * elect_des
des_co2_non_fos_r  = g("des_co2_non_fos") * co2_non_fos_des
des_wastewater_r   = g("des_wastewater") * wastewater_des
des_lignin_heat_r  = g("des_lignin_heat") * lignin_heat_des
des_hemi_r         = g("des_hemi") * hemi_des
# Setup calculation method
setup = o.CalculationSetup(
    target=model,
    impact_method=method,
    parameters=[
        o.ParameterRedef(name="des_wood",          value=des_wood_r),
        o.ParameterRedef(name="des_la",            value=des_la_r),
        o.ParameterRedef(name="des_chcl",          value=des_chcl_r),
        o.ParameterRedef(name="des_h2o",          value=des_h2o_r),
        o.ParameterRedef(name="des_heat",          value=des_heat_r),
        o.ParameterRedef(name="des_elect",         value=des_elect_r),
        o.ParameterRedef(name="des_co2_non_fos",   value=des_co2_non_fos_r),
        o.ParameterRedef(name="des_wastewater",    value=des_wastewater_r),
        o.ParameterRedef(name="des_lignin_heat",   value=des_lignin_heat_r),
        o.ParameterRedef(name="des_hemi",          value=des_hemi_r),
    ],
)
print("Getting values has started... Wait...\n")
# To get only climate change
result = client.calculate(setup)
result.wait_until_ready()
climate_change = next(
    float(i.amount) for i in result.get_total_impacts()
    if i.impact_category.name == "Climate change"
)
result.dispose()
print(f"Climate change (EFv3.1): {climate_change:.3f} kg CO2 eq per kg of pulp\n")
print("Wait...\n")
# To get all results
result = client.calculate(setup)
result.wait_until_ready()
impacts = result.get_total_impacts()
des_dict = {i.impact_category.name: (i.amount, i.impact_category.ref_unit) for i in impacts}
result.dispose()
# Setup only for Kraft
setup_kraft = o.CalculationSetup(
    target=model_kraft,
    impact_method=method
)
# Run calculations for kraft
result_kraft = client.calculate(setup_kraft)
result_kraft.wait_until_ready()
impacts_kraft = result_kraft.get_total_impacts()
kraft_dict = {i.impact_category.name: (i.amount, i.impact_category.ref_unit) for i in impacts_kraft}
result_kraft.dispose()
# Build a unified list of categories
all_categories = sorted(set(des_dict.keys()) | set(kraft_dict.keys()))
# Assemble rows: Impact category, Amount (DES pulping), Amount (Kraft pulping), Unit
rows = []
for cat in all_categories:
    des_amt, des_unit = des_dict.get(cat, (float('nan'), ""))
    kraft_amt, kraft_unit = kraft_dict.get(cat, (float('nan'), ""))
    unit = des_unit if des_unit else kraft_unit
    rows.append([cat, des_amt, kraft_amt, unit])
df_des_lca = pd.DataFrame(
    rows,
    columns=["Impact category", "DES pulping", "Kraft pulping", "Unit"]
)
# Print timing summary
elapsed = time.perf_counter() - t0
total_seconds = int(elapsed)
centi = int((elapsed - total_seconds) * 100)
h, rem = divmod(total_seconds, 3600)
m, s = divmod(rem, 60)
print(f"Execution completed. Total elapsed time (hh:mm:ss:cs): {h:02d}:{m:02d}:{s:02d}:{centi:02d}\n")
# print("\nLCA impacts (EFv3.1) per kg of pulp")
# print(df_des_lca)
#%% SKIP IF CONTRIBUTION TREE IS NOT NEEDED
# LCA: Contribution tree
# Optimized upstream-tree extraction for DES
# Tuning parameters for the upstream tree expansion
MAX_EXPAND_LEVELS = 1  # maximum recursion depth
MAX_EXPAND_NODES = 8   # max number of children per node to traverse
def expand(node: utree.Node, level: int, rows: list, impact_idx: int, impact_name: str,
           unit: str, process_label: str):
    """Recursively expands an upstream tree and collects rows."""
    # Append to rows for DataFrame
    rows.append([
        process_label,
        impact_idx,
        impact_name,
        unit,
        level,
        node.provider.name if node.provider else "",
        node.result
    ])
    # Recurse
    if level < MAX_EXPAND_LEVELS:
        for c in node.childs[0:MAX_EXPAND_NODES]:
            expand(c, level + 1, rows, impact_idx, impact_name, unit, process_label)
def collect_upstream_for_process(client, process_ref, method_ref, process_label: str):
    """
    Runs an OpenLCA calculation, iterates all impact categories of the method,
    expands the upstream tree.
    """
    setup = o.CalculationSetup(
        target=process_ref,
        impact_method=method_ref,
        parameters=[
            o.ParameterRedef(name="des_wood",         value=des_wood_r),
            o.ParameterRedef(name="des_la",           value=des_la_r),
            o.ParameterRedef(name="des_chcl",         value=des_chcl_r),
            o.ParameterRedef(name="des_h2o",          value=des_h2o_r),
            o.ParameterRedef(name="des_heat",         value=des_heat_r),
            o.ParameterRedef(name="des_elect",        value=des_elect_r),
            o.ParameterRedef(name="des_co2_non_fos",  value=des_co2_non_fos_r),
            o.ParameterRedef(name="des_wastewater",   value=des_wastewater_r),
            o.ParameterRedef(name="des_lignin_heat",  value=des_lignin_heat_r),
            o.ParameterRedef(name="des_hemi",         value=des_hemi_r),
        ]
    )
    result = client.calculate(setup)
    result.wait_until_ready()
    rows = []
    cats = result.get_impact_categories()
    for idx, ref in enumerate(cats):
        root = utree.of(result, ref)
        expand(root, 0, rows, idx, ref.name, ref.ref_unit, process_label)
    result.dispose()
    df_interim = pd.DataFrame(rows, columns=[
        "process",
        "impact_index",
        "impact_name",
        "unit",
        "level",
        "provider",
        "result"
    ])
    return df_interim
def get_pulping_upstream_dfs():
    """
    Runs upstream-tree calculations for DES pulping
    and returns DataFrame: (df_des_lca_tree).
    """
    print("Getting values for contribution tree has started... Wait...\n")
    t0 = time.perf_counter()
    client = ipc.Client(8080)
    # Resolve process by name (adjust if you use UUIDs instead)
    proc_des = client.find(o.Process, "DES_v2.0")
    # Impact method
    method = client.find(o.ImpactMethod, "EF v3.1")  # or EF v3.1 (CO2only) or TRACI 2.1
    df_des_lca_tree = collect_upstream_for_process(client, proc_des, method, process_label="DES_v2.0")
    # Timing summary
    elapsed = time.perf_counter() - t0
    total_seconds = int(elapsed)
    centi = int((elapsed - total_seconds) * 100)
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    print(f"Execution for contribution tree completed. Total elapsed time (hh:mm:ss:cs): {h:02d}:{m:02d}:{s:02d}:{centi:02d}")
    return df_des_lca_tree, method.name
df_des_lca_tree, method_name = get_pulping_upstream_dfs()
#%% SKIP IF SANKEY IS NOT NEEDED
# LCA: Sankey diagram
# Only Climate Change
# Tuning parameters for the upstream tree expansion
MAX_EXPAND_LEVELS = 3  # number of subprocesses 
MAX_EXPAND_NODES = 6   # number of stages
def expand(node: utree.Node, level: int, rows: list, impact_idx: int, impact_name: str,
           unit: str, process_label: str):
    """Recursively expands an upstream tree and collects rows."""
    # Append to rows for DataFrame
    rows.append([
        process_label,
        impact_idx,
        impact_name,
        unit,
        level,
        node.provider.name if node.provider else "",
        node.result
    ])
    # Recurse
    if level < MAX_EXPAND_LEVELS:
        for c in node.childs[0:MAX_EXPAND_NODES]:
            expand(c, level + 1, rows, impact_idx, impact_name, unit, process_label)
def collect_upstream_for_process(client, process_ref, method_ref, process_label: str):
    """
    Runs an OpenLCA calculation, iterates all impact categories of the method,
    expands the upstream tree.
    """
    setup = o.CalculationSetup(
        target=process_ref,
        impact_method=method_ref,
        parameters=[
            o.ParameterRedef(name="des_wood",         value=des_wood_r),
            o.ParameterRedef(name="des_la",           value=des_la_r),
            o.ParameterRedef(name="des_chcl",         value=des_chcl_r),
            o.ParameterRedef(name="des_h2o",          value=des_h2o_r),
            o.ParameterRedef(name="des_heat",         value=des_heat_r),
            o.ParameterRedef(name="des_elect",        value=des_elect_r),
            o.ParameterRedef(name="des_co2_non_fos",  value=des_co2_non_fos_r),
            o.ParameterRedef(name="des_wastewater",   value=des_wastewater_r),
            o.ParameterRedef(name="des_lignin_heat",  value=des_lignin_heat_r),
            o.ParameterRedef(name="des_hemi",         value=des_hemi_r),
        ]
    )
    result = client.calculate(setup)
    result.wait_until_ready()
    rows = []
    cats = result.get_impact_categories()
    for idx, ref in enumerate(cats):
        root = utree.of(result, ref)
        expand(root, 0, rows, idx, ref.name, ref.ref_unit, process_label)
    result.dispose()
    df_interim = pd.DataFrame(rows, columns=[
        "process",
        "impact_index",
        "impact_name",
        "unit",
        "level",
        "provider",
        "result"
    ])
    return df_interim
def get_pulping_upstream_dfs():
    """
    Runs upstream-tree calculations for DES pulping
    and returns DataFrame: (df_des_lca_sankey).
    """
    print("Getting values for Sankey diagram has started... Wait...\n")
    t0 = time.perf_counter()
    client = ipc.Client(8080)
    # Resolve process by name (adjust if you use UUIDs instead)
    proc_des = client.find(o.Process, "DES_v2.0")
    # Impact method (Climate Change only)
    method = client.find(o.ImpactMethod, "EF v3.1 (CO2only)")  # or TRACI 2.1 if desired
    df_des_lca_sankey = collect_upstream_for_process(client, proc_des, method, process_label="DES_v2.0")
    # Timing summary
    elapsed = time.perf_counter() - t0
    total_seconds = int(elapsed)
    centi = int((elapsed - total_seconds) * 100)
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    print(f"Execution for contribution tree completed (Sankey). Total elapsed time (hh:mm:ss:cs): {h:02d}:{m:02d}:{s:02d}:{centi:02d}")
    return df_des_lca_sankey, method.name
df_des_lca_sankey, method_name = get_pulping_upstream_dfs()
#%% SKIP IF NOTHING CHANGED TO OPEN_LCA 
# Linear Regression of Climate Change from OpenLCA
# Skip it if you have df_weights...csv and nothing chnaged. 
# To avoid interaction with OpenLCA for OVAT and MonteCarlo.
# Validate linearity and extract regression coefficients.
# See detailed explanation in Backup.
# Start timing
t0 = time.perf_counter()
# Connect to openLCA IPC
client = ipc.Client(8080)
# Find model and method
model = client.find(o.Process, "DES_v2.0")
method = client.find(o.ImpactMethod, "EF v3.1 (CO2only)")
# Get global parameters once
print("\nGetting values for regression coefficients has started... Wait... \n")
params = client.get_all(o.Parameter)
def g(name: str) -> float:
    for p in params:
        if p.name == name and getattr(p, "context", None) is None:
            return float(p.value)
    raise KeyError(f"Global parameter '{name}' not found")
# Map global parameter names in the LCA model to i_des variable names
# CO2 is removed; include water as des_h2o <-> h2o_des
param_map = [
    ("des_la",             "la_des"),
    ("des_chcl",           "chcl_des"),
    ("des_heat",           "heat_des"),
    ("des_wood",           "wood_logs_des"),
    ("des_h2o",            "h2o_des"),
    ("des_elect",          "elect_des"),
    ("des_lignin_heat",    "lignin_heat_des"),
    ("des_hemi",           "hemi_des"),
]
# Alter one variable while others = 0
base_multipliers = {v: 0.0 for _, v in param_map}
def build_setup(mult_overrides):
    mults = base_multipliers.copy()
    mults.update(mult_overrides)
    redefs = []
    for gp_name, mult_key in param_map:
        redefs.append(
            o.ParameterRedef(
                name=gp_name,
                value=g(gp_name) * mults[mult_key]
            )
        )
    return o.CalculationSetup(
        target=model,
        impact_method=method,
        parameters=redefs,
    )
# Values to alter for each i_des
test_values = [0.1, 1, 10]  # keep "1" always in the middle of the array!!!
# Run "OVAT" for each variable and collect DataFrames
dfs = {}  # dict: key=i_des variable, value=DataFrame with columns ["i_des", "Climate Change"]
for _, var_key in param_map:
    results = []
    for val in test_values:
        setup = build_setup({var_key: val})
        result = client.calculate(setup)
        result.wait_until_ready()
        climate_change = next(
            float(i.amount)
            for i in result.get_total_impacts()
            if i.impact_category.name == "Climate change"
        )
        results.append(climate_change)
        result.dispose()
    df = pd.DataFrame({
        "i_des": test_values,
        "Climate Change": results
    })
    dfs[var_key] = df
# Combined long-form data frame: columns = ["variable", "i_des", "Climate Change"]
df_for_lre = pd.concat(
    [df.assign(variable=k) for k, df in dfs.items()],
    ignore_index=True
)
# Create individual variables like df_la_des, df_chcl_des, ... for each dataframe
for k, df in dfs.items():
    globals()[f"df_{k}"] = df
# Statistical validation (linearity)
for name, df in dfs.items():
    res = linregress(df["i_des"], df["Climate Change"])
    slope = res.slope
    intercept = res.intercept
    r2 = res.rvalue ** 2
    print(f"{name}: Slope={slope:.6f}, Intercept={intercept:.3f}, RSQ={r2:.6g}")
# Extract regression coefficients (weights) using slopes at i_des = 1
var_map = [
    ("la_des",           "w_la"),
    ("chcl_des",         "w_chcl"),
    ("heat_des",         "w_heat"),
    ("wood_logs_des",    "w_wood"),
    ("h2o_des",          "w_h2o"),
    ("elect_des",        "w_elect"),
    ("lignin_heat_des",  "w_lignin_heat"),
    ("hemi_des",         "w_hemi"),
]
# Compute slopes once using linregress
slope_map = {}
for name, df in dfs.items():
    res = linregress(df["i_des"], df["Climate Change"])
    slope_map[name] = res.slope  # reuse this slope
# Assign slopes to the requested variables
for df_key, var_name in var_map:
    globals()[var_name] = float(slope_map[df_key])
# Optional: print results
for _, var_name in var_map:
    print(f"{var_name} = {globals()[var_name]:.6f}")
# to avoid "undefined names" (explicit assignments)
w_la           = float(slope_map["la_des"])
w_chcl         = float(slope_map["chcl_des"])
w_heat         = float(slope_map["heat_des"])
w_wood         = float(slope_map["wood_logs_des"])
w_h2o          = float(slope_map["h2o_des"])
w_elect        = float(slope_map["elect_des"])
w_lignin_heat  = float(slope_map["lignin_heat_des"])
w_hemi         = float(slope_map["hemi_des"])
# Data frame for weights
df_weights = pd.DataFrame([{
    "w_la": w_la,
    "w_chcl": w_chcl,
    "w_heat": w_heat,
    "w_wood": w_wood,
    "w_h2o": w_h2o,
    "w_elect": w_elect,
    "w_lignin_heat": w_lignin_heat,
    "w_hemi": w_hemi,
}])
# Execution time summary
elapsed = time.perf_counter() - t0
ts = int(elapsed); cs = int((elapsed - ts) * 100)
h, rem = divmod(ts, 3600); m, s = divmod(rem, 60)
print(f"\nExecution completed for regression coefficients. Total elapsed time (hh:mm:ss:cs): {h:02d}:{m:02d}:{s:02d}:{cs:02d}")
#%% STAR USING HERE IF NOTHING CHANGED TO OPEN_LCA  
#Linear Regression Equation (LRE) for Climate Change (CC).
# Update if need if something changed in OpenLCA.
#Validation that Climate Change is similar to that obtained from OpenLCA
# Path to weights CSV
csv_path6 = Path("/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/tables_des/df_weights_for_lre.csv")
# Load into DataFrame
df_weights = pd.read_csv(csv_path6)
# Extract weights as a dict (supports wide or long format)
expected_cols = ["w_la", "w_chcl", "w_heat", "w_wood", "w_h2o", "w_elect", "w_lignin_heat", "w_hemi"]
# Assign variables for LRE
w_la          = float(df_weights.at[0, "w_la"])
w_chcl        = float(df_weights.at[0, "w_chcl"])
w_heat        = float(df_weights.at[0, "w_heat"])
w_wood        = float(df_weights.at[0, "w_wood"])
w_h2o         = float(df_weights.at[0, "w_h2o"])
w_elect       = float(df_weights.at[0, "w_elect"])
w_lignin_heat = float(df_weights.at[0, "w_lignin_heat"])
w_hemi        = float(df_weights.at[0, "w_hemi"])
# The Climate Change linear regression is
cc = (w_la * la_des + 
      w_chcl * chcl_des +
      w_heat * heat_des +
      w_wood * wood_logs_des + 
      w_h2o * h2o_des +
      w_elect * elect_des + 
      w_lignin_heat * lignin_heat_des +
      w_hemi * hemi_des)
print("\nValue of Climate Change obtained by linear regression equation (kgCO2eq/kgPulp)", round(cc, 3))
# To print value obtained through the interaction with OpenLCA run the DES LCA cell.
# The values should be identical.
#%% LCA: OVAT without interaction with OpenLCA (DES version)
# to see the old version, see backup files
# creating and collecting set of data for simulation
# min and max variation
min_var = 0.5
max_var = 1.5
# Define variables ranges (DES: use Loss_DES instead of Loss_EG; keep H2O; remove CO2 loss)
Loss_DES_i   = [0.02,        Loss_DES,    0.10]
yP_i         = [0.40,        yP,          0.70]
ltow_i        = [5.0,         ltow,         20.0]
Ewh_i        = [Ewh*min_var, Ewh,         Ewh*max_var]
Edef_i       = [Edef*min_var, Edef,       Edef*max_var]
eff_boiler_i = [0.60,        eff_boiler,  0.95]
lig_i        = [0.20,        lig,         0.35]
SP_i         = [0.75,        SP,          0.94]
# Build a DataFrame with min, base, max for each OVAT variable
data_ovat_ranges = [
    {"variable": "Loss_DES",  "min": Loss_DES_i[0],  "base": Loss_DES_i[1],  "max": Loss_DES_i[2]},
    {"variable": "yP",        "min": yP_i[0],        "base": yP_i[1],        "max": yP_i[2]},
    {"variable": "ltow",       "min": ltow_i[0],       "base": ltow_i[1],       "max": ltow_i[2]},
    {"variable": "Ewh",       "min": Ewh_i[0],       "base": Ewh_i[1],       "max": Ewh_i[2]},
    {"variable": "Edef",      "min": Edef_i[0],      "base": Edef_i[1],      "max": Edef_i[2]},
    {"variable": "eff_boiler","min": eff_boiler_i[0],"base": eff_boiler_i[1],"max": eff_boiler_i[2]},
    {"variable": "lig",       "min": lig_i[0],       "base": lig_i[1],       "max": lig_i[2]},
    {"variable": "SP",        "min": SP_i[0],        "base": SP_i[1],        "max": SP_i[2]},
]
df_des_ovat_ranges = pd.DataFrame(data_ovat_ranges, columns=["variable", "min", "base", "max"])
# print(df_des_ovat_ranges.round(3).to_string(index=False))
# Helper: compute Climate Change from a DES tuple
# Expected tuple (in this order):
# (la_des, chcl_des, heat_des, wood_logs_des, h2o_des, elect_des, lignin_heat_des, hemi_des)
def cc_from_inp(inp):
    wood, la, chcl, h2o, heat, elect, pulp, lignin, co2_non_fos, wastewater, lignin_heat, hemi = inp
    return (w_la * la +
            w_chcl * chcl +
            w_heat * heat +
            w_wood * wood +
            w_h2o * h2o +
            w_elect * elect +
            w_lignin_heat * lignin_heat +
            w_hemi * hemi)
# Base tuple and its CC
inp_base = DES(giv_dat)
cc_base = cc_from_inp(inp_base)
# Minimum effects (OVAT)
in_DES_min         = DES({**giv_dat, "Loss_DES":   Loss_DES_i[0]})
in_yP_min          = DES({**giv_dat, "yP":         yP_i[0]})
in_ltow_min         = DES({**giv_dat, "ltow":        ltow_i[0]})
in_Ewh_min         = DES({**giv_dat, "Ewh":        Ewh_i[0]})
in_Edef_min        = DES({**giv_dat, "Edef":       Edef_i[0]})
in_eff_boiler_min  = DES({**giv_dat, "eff_boiler": eff_boiler_i[0]})
in_lig_min         = DES({**giv_dat, "lig":        lig_i[0]})
in_SP_min          = DES({**giv_dat, "SP":         SP_i[0]})
cc_DES_min         = cc_from_inp(in_DES_min)
cc_yP_min          = cc_from_inp(in_yP_min)
cc_ltow_min         = cc_from_inp(in_ltow_min)
cc_Ewh_min         = cc_from_inp(in_Ewh_min)
cc_Edef_min        = cc_from_inp(in_Edef_min)
cc_eff_boiler_min  = cc_from_inp(in_eff_boiler_min)
cc_lig_min         = cc_from_inp(in_lig_min)
cc_SP_min          = cc_from_inp(in_SP_min)
min_effect = [cc_DES_min, cc_yP_min, cc_ltow_min, cc_Ewh_min, cc_Edef_min,
              cc_eff_boiler_min, cc_lig_min, cc_SP_min]
# No effect (baseline CC for each slot)
no_effect = [cc_base] * len(min_effect)  # repeat cc_base per altered parameter
# Maximum effects (OVAT)
in_DES_max         = DES({**giv_dat, "Loss_DES":   Loss_DES_i[2]})
in_yP_max          = DES({**giv_dat, "yP":         yP_i[2]})
in_ltow_max         = DES({**giv_dat, "ltow":        ltow_i[2]})
in_Ewh_max         = DES({**giv_dat, "Ewh":        Ewh_i[2]})
in_Edef_max        = DES({**giv_dat, "Edef":       Edef_i[2]})
in_eff_boiler_max  = DES({**giv_dat, "eff_boiler": eff_boiler_i[2]})
in_lig_max         = DES({**giv_dat, "lig":        lig_i[2]})
in_SP_max          = DES({**giv_dat, "SP":         SP_i[2]})
cc_DES_max         = cc_from_inp(in_DES_max)
cc_yP_max          = cc_from_inp(in_yP_max)
cc_ltow_max         = cc_from_inp(in_ltow_max)
cc_Ewh_max         = cc_from_inp(in_Ewh_max)
cc_Edef_max        = cc_from_inp(in_Edef_max)
cc_eff_boiler_max  = cc_from_inp(in_eff_boiler_max)
cc_lig_max         = cc_from_inp(in_lig_max)
cc_SP_max          = cc_from_inp(in_SP_max)
max_effect = [cc_DES_max, cc_yP_max, cc_ltow_max, cc_Ewh_max, cc_Edef_max,
              cc_eff_boiler_max, cc_lig_max, cc_SP_max]
# Build DataFrame 
labels = ["DES losses", "yP", "LtoW", "Ewh", "Edef", "eff_boiler", "lig", "SP"]
df_des_ovat_results = pd.DataFrame({
    "varied parameter": labels,
    "min effect": min_effect,
    "no effect": no_effect,
    "max effect": max_effect,
})
print("\nOVAT simulation has been completed (no interaction with OpenLCA). The results are: \n")
print(df_des_ovat_results.round(3))
#%% Tornado plot for OVAT
# If df was not uploaded, you can simply upload it withouth running previous cell.
# csv_path_tornado = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/tables_des/df_des_ovat_results.csv')
# # Load CC ranges from CSV
# df_des_ovat_results = pd.read_csv(csv_path_tornado)  # expects columns: varied parameter, min effect, no effect, max effect
# Names of the parameters (use order from CSV to stay consistent)
parameters = ["DES losses (2, 5, 10, %)", "Pulp yield (40, 57, 70, %)",
      "LtoW (5, 10, 20)", "WH energy (±50%, kWh)", "DF energy (±50%, kWh)",
      "BLR efficiency (60, 80, 95, %)", "Lignin of wood (20, 27, 35, %)", "Membrane SP (75, 90, 94, %)"]
# Values for tornado (Climate Change)
cc_min  = df_des_ovat_results["min effect"].to_numpy(dtype=float)
cc_zero = df_des_ovat_results["no effect"].to_numpy(dtype=float)
cc_max  = df_des_ovat_results["max effect"].to_numpy(dtype=float)
# Calculate the differences from the central point as NumPy arrays
low_arr  = np.asarray(cc_zero - cc_min)
high_arr = np.asarray(cc_max  - cc_zero)
# Calculate the percentage of "dmin" and "dmax" effect
dmin = ((cc_min - cc_zero) / cc_zero) * 100
dmax = ((cc_max - cc_zero) / cc_zero) * 100
# Create a DataFrame using the arrays
df = pd.DataFrame({
    "Parameter": parameters,
    "Min effect on CC": cc_min,
    "No effect on CC": cc_zero,
    "Max effect on CC": cc_max,
    "Low": low_arr,
    "High": high_arr,
    "dmin": dmin,
    "dmax": dmax
})
# Sort the DataFrame by "dmin" and "dmax" in ascending order
df["mag"] = np.maximum(np.abs(df["dmin"]), np.abs(df["dmax"]))
df = df.sort_values(by="mag", ascending=True).drop(columns="mag")
# Plotting the tornado plot
fig_tornado, ax = plt.subplots(figsize=(12/2.54, 8/2.54), dpi=300)  # Convert cm to inches for figsize
# --- Gradient setup (currently unused, kept for future use) ---
grad = np.linspace(0, 1, 256).reshape(1, -1)  # horizontal gradient
cmap_pos = LinearSegmentedColormap.from_list('skyblue_grad', ['white', 'skyblue'])
cmap_neg = LinearSegmentedColormap.from_list('salmon_grad',  ['white', 'salmon'])
# Parameters for which labels are centered differently
special_params = {
    'Pulp yield (40, 57, 70, %)',
    'BLR efficiency (60, 80, 95, %)',
    'Lignin of wood (20, 27, 35, %)',
    'Membrane SP (75, 90, 94, %)'
}
# Plot bars and annotations
for i, row in df.reset_index(drop=True).iterrows():
    param  = row['Parameter']
    left   = row['No effect on CC']
    high_i = row['High']    # > 0
    low_i  = row['Low']     # > 0
    dmax_  = row['dmax']
    dmin_  = row['dmin']
    # Draw bars first to establish categorical y-axis
    ax.barh(param,  high_i, left=left, color='salmon',  edgecolor='black')
    ax.barh(param, -low_i,  left=left, color='skyblue', edgecolor='black')
    # Bar endpoints for annotations
    x_right = left + high_i
    x_left  = left - low_i
    # Annotations
    if param in special_params:
        ax.text(x_right - 0.015, param, f'{dmax_:.1f}%', va='center', ha='left',  color='black', fontsize=7)
        ax.text(x_left  + 0.015, param, f'{dmin_:.1f}%', va='center', ha='right', color='black', fontsize=7)
    else:
        ax.text(x_right + 0.0025, param, f'{dmax_:.1f}%', va='center', ha='left',  color='black', fontsize=7)
        ax.text(x_left  - 0.0025, param, f'{dmin_:.1f}%', va='center', ha='right', color='black', fontsize=7)
# Vertical reference lines
for x in df["No effect on CC"].unique():
    ax.axvline(x=x, color='black', linewidth=1, linestyle='--')
# Change font size of y-axis na x-axis labels
ax.tick_params(axis='y', labelsize=9)
ax.tick_params(axis='x', labelsize=9)
# Adjust x-axis limits (optional: keep or adjust based on your data)
plt.xlim(0.1, 0.2)
# Set labels and title with increased font size
ax.set_xlabel('Effect on Climate Change (t CO$_2$eq / tp)', fontsize=8)
# Legend using proxy handles
legend_handles = [
    Patch(facecolor='salmon',  edgecolor='black', label='Max effect'),
    Patch(facecolor='skyblue', edgecolor='black', label='Min effect'),
]
ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(0.6, 0.15), fontsize=7)
plt.tight_layout() # optional, if not used bars occupy less space
# Save plot as png image with specified characteristics
# save_tornado_plot = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/figures_des/des_tornado_cc.png')
# fig_tornado.savefig(save_tornado_plot, dpi=300, bbox_inches='tight')
# Show plot
plt.show()
#%% LCA, Monte Carlo: no interaction with OpenLCA
print("\nMonte Carlo simulation has been started (no interaction with OpenLCA).\n")
def cc_from_inp(inp):
    wood, la, chcl, h2o, heat, elect, pulp, lignin, co2_non_fos, wastewater, lignin_heat, hemi = inp
    return (w_la * la +
            w_chcl * chcl +
            w_heat * heat +
            w_wood * wood +
            w_h2o * h2o +
            w_elect * elect +
            w_lignin_heat * lignin_heat +
            w_hemi * hemi)
# LCA, Monte Carlo (no interaction with OpenLCA)
np.random.seed(42)  # deterministic and reproducible
CI = 0.95
num_samples = 10000
var = 0.5
# Draws (DES): Loss_DES, yP, ltow (formerly H2O), Ewh, Edef, eff_boiler, lig, SP
Loss_DES_j   = np.random.triangular(Loss_DES_i[0],   Loss_DES_i[1],   Loss_DES_i[2],   size=num_samples)
yP_j         = np.random.triangular(yP_i[0],         yP_i[1],         yP_i[2],         size=num_samples)
ltow_j       = np.random.triangular(ltow_i[0],       ltow_i[1],       ltow_i[2],       size=num_samples)
Ewh_j        = np.random.normal(loc=Ewh_i[1],  scale=Ewh_i[1]*var,   size=num_samples)
Edef_j       = np.random.normal(loc=Edef_i[1], scale=Edef_i[1]*var,  size=num_samples)
eff_boiler_j = np.random.triangular(eff_boiler_i[0], eff_boiler_i[1], eff_boiler_i[2], size=num_samples)
lig_j        = np.random.triangular(lig_i[0],        lig_i[1],        lig_i[2],        size=num_samples)
SP_j         = np.random.triangular(SP_i[0],         SP_i[1],         SP_i[2],         size=num_samples)
# Run DES per sample; collect Climate Change values
cc_mc = np.empty(num_samples, dtype=float)
print_every = 500  # print progress every N iterations
t0 = time.perf_counter()
for k in range(num_samples):
    gd = {
        **giv_dat,
        "Loss_DES":   float(Loss_DES_j[k]),
        "yP":         float(yP_j[k]),
        "ltow":       float(ltow_j[k]),    # note: was "H2O"
        "Ewh":        float(Ewh_j[k]),
        "Edef":       float(Edef_j[k]),
        "eff_boiler": float(eff_boiler_j[k]),
        "lig":        float(lig_j[k]),
        "SP":         float(SP_j[k]),
    }
    inp = DES(gd)                 # DES returns 12/10/9/8-tuple (cc_from_inp handles all)
    cc_mc[k] = cc_from_inp(inp)   # scalar Climate Change
    # progress print
    if (k + 1) % print_every == 0 or k == 0 or (k + 1) == num_samples:
        elapsed = time.perf_counter() - t0
        total_seconds = int(elapsed)
        centi = int((elapsed - total_seconds) * 100)  # centiseconds
        h, rem = divmod(total_seconds, 3600)
        m, s = divmod(rem, 60)
        print(
            f"Iteration {k+1}/{num_samples} | elapsed {h:02d}:{m:02d}:{s:02d}:{centi:02d} | "
            f"CC={cc_mc[k]:.3f}"
        )
# Summary and total time
elapsed_total = time.perf_counter() - t0
mean_cc = float(np.mean(cc_mc))
low, high = np.percentile(cc_mc, [(1 - CI) / 2 * 100, (1 + CI) / 2 * 100])
ts = int(elapsed_total)
centi = int((elapsed_total - ts) * 100)
h, rem = divmod(ts, 3600)
m, s = divmod(rem, 60)
print("\nMonte Carlo simulation has been completed (no connection to OpenLCA).\n")
print(f"MC Climate Change: mean={mean_cc:.3f}, {int(CI*100)}% CI=({low:.3f}, {high:.3f})\n")
print(f"Total elapsed time: {h:02d}:{m:02d}:{s:02d}:{centi:02d}\n")
# DataFrame
df_mc = pd.DataFrame({"cc_mc": cc_mc})
print(df_mc.head())
#%% Histogram plot for Monte Carlo
# If df_mc does not exist in memory already.
# csv_path_monte_carlo = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/tables_des/df_mc.csv')  # change path/name if needed
# Load CC ranges from CSV
# df_mc = pd.read_csv(csv_path_monte_carlo)  # expects columns: varied parameter, min effect, no effect, max effect
cc_mc  = df_mc["cc_mc"].to_numpy(dtype=float)
# Calculate statistics
CI = 0.95
num_samples = 10_000
# Calculation empirical quantilies
low, high = np.percentile(cc_mc, [(1 - CI) / 2 * 100, (1 + CI) / 2 * 100])
# For Gausian curve assuming normal distribution only to show for comaprision purpose on figure
mean_cc = np.mean(cc_mc)
min_cc = np.min(cc_mc)
max_cc = np.max(cc_mc)
std_cc = np.std(cc_mc)
# Calculate percentages
percent_below_0 = np.mean(cc_mc < 0) * 100.0
percent_above_0 = np.mean(cc_mc > 0) * 100.0
plt.figure(figsize=(12/2.54, 8/2.54))
# Plot histogram using bins
count, bins, ignored = plt.hist(cc_mc, bins=15, density=True, edgecolor='black', alpha=0.6, label='Histogram')
# Plot Gaussian curve
x = np.linspace(min_cc, max_cc, 1000)
plt.plot(x, norm.pdf(x, mean_cc, std_cc), 'r--', label='Gaussian Fit')
# Plot vertical line at CC = mean
plt.axvline(x=mean_cc, color='black', linestyle='--', linewidth=1.5, label='Mean of CC')
ax = plt.gca()
ci_color = 'tab:blue'
plt.axvline(x=low,  color=ci_color, linestyle=':', linewidth=1.5, label=f'{int(CI*100)}% CI bounds')
plt.axvline(x=high, color=ci_color, linestyle=':', linewidth=1.5)
# Annotate counts below and above zero
plt.annotate(f"Climate Change\n(t CO$_2$eq / tp): \nMean={mean_cc:.3f}, \n{int(CI*100)}% CI=({low:.3f}, {high:.3f}), \nIterations={num_samples}",xy=(0.025, 0.975), xycoords='axes fraction', fontsize=8, verticalalignment='top')
#plt.annotate(f"\nSTD: {std_cc:.3f} kgCO2eq", xy=(0.7, 0.65), xycoords='axes fraction', fontsize=8, verticalalignment='top')
# plt.title('EG-CO2')
plt.xlabel('Climate Change (t CO$_2$eq / tp)')
plt.xlim(0.1, 0.2) # Adjust x-axis limits
plt.ylabel('Density')
plt.legend(fontsize=8)
plt.grid(False)
plt.tight_layout()
# Save plot as png image with specified characteristics
# save_histogram_plot = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/figures_des/des_histogram_cc.png')
# plt.savefig(save_histogram_plot, dpi=300, bbox_inches='tight')
plt.show()
#%% Linear regression and ANOVA of experimental data
# similar to R script but adopted to python (des_model.R)
# Path to your CSV
csv_path0 = "/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/tables_des/exp_data_des.csv"
# Read data
df_exp = pd.read_csv(csv_path0)
# Mask Run10 because it is actually uncooked pulp (row index 9 if zero-based)
#df_exp.iloc[9] = np.nan
# Linear regression for Yield with all interactions up to 3-way
# y = b0 + btemp*temp + btime*time + bltow*ltow + bint1*temp*time + bint2*temp*ltow + bint3*time*ltow + bint4*time*temp*ltow
yield_formula = "Yield_prcnt ~ Temperature_C * Time_hrs * LtoW"
# Alternative (main effects only):
# yield_formula = "Yield_prcnt ~ Temperature_C + Time_hrs + LtoW"
yield_model = smf.ols(yield_formula, data=df_exp).fit()
print("\n=== Yield model summary ===")
print(yield_model.summary())
# ANOVA table (Type II; use typ=3 if you need Type III with a properly coded design)
# print("\n=== Yield model ANOVA (Type II) ===")
# print(anova_lm(yield_model, typ=2))
# Experimental vs fitted for Yield
yield_results = pd.DataFrame({
    "expt Yield": df_exp["Yield_prcnt"],
    "pred Yield": yield_model.fittedvalues,
    "residual error": yield_model.resid
})
print("\n=== Yield: experimental vs predicted and residuals ===")
print(yield_results)
# To extract regression coefficients for the model of Yield
params_yld  = yield_model.params
b0_yld      = params_yld.get("Intercept")
btemp_yld   = params_yld.get("Temperature_C")
btime_yld   = params_yld.get("Time_hrs")
bltow_yld   = params_yld.get("LtoW")
btt_yld     = params_yld.get("Temperature_C:Time_hrs")
btempl_yld  = params_yld.get("Temperature_C:LtoW")
btimel_yld  = params_yld.get("Time_hrs:LtoW")
bttl_yld    = params_yld.get("Temperature_C:Time_hrs:LtoW")
# Linear regression for Fibre length with interactions
#fibre_formula = "Fibre_mm ~ Temperature_C * Time_hrs * LtoW"
# Alternative (main effects only):
# fibre_formula = "Fibre_mm ~ Temperature_C + Time_hrs + LtoW"
# Quadratic model
# fibre_formula = """Fibre_mm ~ Temperature_C + Time_hrs + LtoW +
# I(Temperature_C**2) +
# I(Time_hrs**2) +
# I(LtoW**2) +
# Temperature_C:Time_hrs +
# Temperature_C:LtoW +
# Time_hrs:LtoW
# """
fibre_formula = """
Fibre_mm ~
Temperature_C +
Time_hrs +
LtoW +
Temperature_C:Time_hrs
"""
fibre_model = smf.ols(fibre_formula, data=df_exp).fit()
print("\n=== Fibre length model summary ===")
print(fibre_model.summary())
# print("\n=== Fibre length model ANOVA (Type II) ===")
# print(anova_lm(fibre_model, typ=2))
fibre_results = pd.DataFrame({
    "expt Fibre Length": df_exp["Fibre_mm"],
    "pred Fibre Length": fibre_model.fittedvalues,
    "residual error": fibre_model.resid
})
print("\n=== Fibre: experimental vs predicted and residuals ===")
print(fibre_results)
# Extract regression coefficients for the fibre model
params_fbr  = fibre_model.params
b0_fbr      = params_fbr.get("Intercept", 0)
btemp_fbr   = params_fbr.get("Temperature_C", 0)
btime_fbr   = params_fbr.get("Time_hrs", 0)
bltow_fbr   = params_fbr.get("LtoW", 0)
btt_fbr     = params_fbr.get("Temperature_C:Time_hrs", 0)
# Optional: compute fitted values from the extracted coefficients
fibre_calc = (
    b0_fbr
    + btemp_fbr * df_exp["Temperature_C"]
    + btime_fbr * df_exp["Time_hrs"]
    + bltow_fbr * df_exp["LtoW"]
    + btt_fbr   * df_exp["Temperature_C"] * df_exp["Time_hrs"]
)
print("\nUse the equation within the range of the experimental data!")
#%% To get varibles from the experimental data for Bayesian optimization
T = df_exp["Temperature_C"]
t = df_exp["Time_hrs"]
L = df_exp["LtoW"]
# Fibre Length
def fibre_length(T, t, L):
    b0 = b0_fbr
    bT = btemp_fbr
    bt = btime_fbr
    bL = bltow_fbr
    bTt = btt_fbr
    return (
        b0
        + bT*T
        + bt*t
        + bL*L
        + bTt*T*t
    )
# Climate Change
def cc_from_inp(inp):
    wood, la, chcl, h2o, heat, elect, pulp, lignin, co2_non_fos, wastewater, lignin_heat, hemi = inp
    return (w_la * la +
            w_chcl * chcl +
            w_heat * heat +
            w_wood * wood +
            w_h2o * h2o +
            w_elect * elect +
            w_lignin_heat * lignin_heat +
            w_hemi * hemi)
# Adopting given data
def make_gd(T, t, L):
    # Important: pass scalars for optimization, not full Series.
    # If DES expects numpy scalars/arrays, wrap as needed:
    # e.g., float(T) or np.array([T]) depending on DES API.
    gd = {
        **giv_dat,
        "Tdl": float(T),
        "t1": float(t),
        "ltow": float(L),
    }
    return gd
#CC from parameters
def cc_from_params(T, t, L):
    gd = make_gd(T, t, L)
    flows = DES(gd)
    # Normalize DES output to the tuple expected by cc_from_inp.
    # Handle multiple possible return formats (tuple/list/dict/DataFrame).
    if isinstance(flows, dict):
        # Map keys to required order; adjust keys if your DES dict uses different names.
        tup = (
            flows["wood"],
            flows["la"],
            flows["chcl"],
            flows["h2o"],
            flows["heat"],
            flows["elect"],
            flows["pulp"],
            flows["lignin"],
            flows["co2_non_fos"],
            flows["wastewater"],
            flows["lignin_heat"],
            flows["hemi"],
        )
    elif hasattr(flows, "__iter__") and not isinstance(flows, (str, bytes)):
        tup = tuple(flows)  # assume order matches cc_from_inp
    else:
        raise ValueError("DES() returned an unsupported type; expected dict or tuple/list")
    # Convert to float in case they are numpy scalars/Series
    tup = tuple(float(x) for x in tup)
    return cc_from_inp(tup)
# Build a DataFrame of inputs and computed outputs
df_fibre_cc = df_exp[["Temperature_C", "Time_hrs", "LtoW"]].copy()
# Use experimental fibre values directly (replace "Fibre_mm" if your column name differs)
df_fibre_cc["Fibre_mm"] = df_exp["Fibre_mm"].astype(float)
# OR Compute Fibre from regression
# df_cc["Fibre_mm"] = df_exp.apply(
#     lambda r: fibre_length(r["Temperature_C"], r["Time_hrs"], r["LtoW"]),
#     axis=1
# )
# Compute CC via DES-backed function (per-row scalars), i.e. from LRE
df_fibre_cc["climate_change"] = df_exp.apply(
    lambda r: cc_from_params(r["Temperature_C"], r["Time_hrs"], r["LtoW"]),
    axis=1
)
#%% Scalarized Bayesian optimization with Ax This minimizes cc − λ · fibre_length + penalty. 
LAMBDA = 0.2 # (0.1-1.0)
PENALTY_W = 50  # (10-200)
trials = 300 # more trails => more Pareto points
RNG = 42 # random generator to have same randomness
# Observed fibre range from experimental data (use your column name)
fbr_obs_min = float(df_exp["Fibre_mm"].min())
fbr_obs_max = float(df_exp["Fibre_mm"].max())
# Trade-off weight and penalty strength
def objective_evaluator(params):
    T = params["Temperature_C"]
    t = params["Time_hrs"]
    L = params["LtoW"]
    cc = cc_from_params(T, t, L)
    f_raw = fibre_length(T, t, L)
    # Quadratic penalty outside experimental range
    if f_raw < fbr_obs_min:
        penalty = PENALTY_W * (fbr_obs_min - f_raw) ** 2
    elif f_raw > fbr_obs_max:
        penalty = PENALTY_W * (f_raw - fbr_obs_max) ** 2
    else:
        penalty = 0.0
    obj = cc - LAMBDA * f_raw + penalty
    return {"scalar_obj": (obj, 0.0)}
# to keep RNG same
random.seed(RNG)
np.random.seed(RNG)
try:
    import torch
    torch.manual_seed(RNG)
except ImportError:
    pass
# optimize funtion
best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "Temperature_C", "type": "range", "bounds": [110.0, 130.0]},
        {"name": "Time_hrs",      "type": "range", "bounds": [2.5, 4.5]},
        {"name": "LtoW",          "type": "range", "bounds": [5.0, 20.0]},
    ],
    evaluation_function=objective_evaluator,
    objective_name="scalar_obj",
    minimize=True,
    total_trials=trials,
    random_seed=RNG,
)
# Extract best params (dict)
T_best = best_parameters["Temperature_C"]
t_best = best_parameters["Time_hrs"]
L_best = best_parameters["LtoW"]
# Report results (show both raw and clipped fibre)
f_raw_best = fibre_length(T_best, t_best, L_best)
f_clip_best = float(np.clip(f_raw_best, fbr_obs_min, fbr_obs_max))
print("Best params:", {
    "Temperature_C": round(best_parameters["Temperature_C"], 0),
    "Time_hrs": round(best_parameters["Time_hrs"], 2),
    "LtoW": round(best_parameters["LtoW"], 2),
})
#print("Best scalarized objective:", values)
print("CC (tCO2/tPulp):", round(cc_from_params(T_best, t_best, L_best), 3))
print("Fibre length (mm) (raw):", round(f_raw_best, 3))
print("Fibre length (mm) (within exp range):", round(f_clip_best, 3))
#Pareto plot
# Minimal Pareto extraction and plot
# Experimental fibre bounds
fmin = float(df_exp["Fibre_mm"].min())
fmax = float(df_exp["Fibre_mm"].max())

points = []  # (cc, fibre, params)
for tr_id, tr in sorted(experiment.trials.items()):
    if tr.status.name != "COMPLETED":
        continue
    p = tr.arm.parameters
    T, t, L = p["Temperature_C"], p["Time_hrs"], p["LtoW"]
    cc = cc_from_params(T, t, L)
    f_raw = fibre_length(T, t, L)
    if f_raw > fmax or f_raw < fmin:
        continue
    points.append((cc, f_raw, p))
if not points:
    raise RuntimeError("No completed, in-range trials found.")
pts = np.array([(cc, f) for cc, f, _ in points])
n = len(pts)
is_pareto = np.ones(n, dtype=bool)
for i in range(n):
    if not is_pareto[i]:
        continue
    dominates = ((pts[:, 0] <= pts[i, 0]) & (pts[:, 1] >= pts[i, 1]) &
                 ((pts[:, 0] < pts[i, 0]) | (pts[:, 1] > pts[i, 1])))
    dominates[i] = False
    if np.any(dominates):
        is_pareto[i] = False
pareto_points = [points[i] for i in range(n) if is_pareto[i]]
#%% Pareto data frame
# Build rows in the same order as pareto_points
rows = []
for cc, f, p in pareto_points:
    rows.append({
        "Temperature (oC)": float(p.get("Temperature_C", np.nan)),
        "Time (hrs)": float(p.get("Time_hrs", np.nan)),
        "LtoW": float(p.get("LtoW", np.nan)),
        "CC (tCO2eq/tp)": float(cc),
        "Fibre (mm)": float(f),
    })
# Create DataFrame with explicit column order
df_pareto = pd.DataFrame(
    rows,
    columns=[
        "Temperature (oC)",
        "Time (hrs)",
        "LtoW",
        "CC (tCO2eq/tp)",
        "Fibre (mm)",
    ],
)
# Optional: preview and/or save
print(round(df_pareto,3))
# to adapt column names for latex
latex_cols = {
    "Temperature (oC)": r"Temperature ($^\circ$C)",
    "Time (hrs)": r"Time (hrs)",
    "LtoW": r"LtoW",
    "Climate Change (tCO2eq/tp)": r"Climate Change (t~CO$_2$eq / t$_p$)",
    "Fibre length (mm)": r"Fibre length (mm)",
}

df_pareto_ltx = df_pareto.rename(columns=latex_cols)
#%% Pareto Plot
plt.figure(figsize=(12/2.54, 8/2.54))
# Add experimental (human) data points from df_fibre_cc
# Ensure numeric and drop NaNs
exp_df = df_fibre_cc[["climate_change", "Fibre_mm"]].astype(float).dropna()
# Optional: keep experimental points within the same fibre range as AI trials
mask_in_range = (exp_df["Fibre_mm"] >= fmin) & (exp_df["Fibre_mm"] <= fmax)
exp_plot = exp_df[mask_in_range]
plt.scatter(
    exp_plot["climate_change"].values,
    exp_plot["Fibre_mm"].values,
    s=40, c="k", marker="x", linewidths=1.5, label="Experimental (human)"
)
# If you want to annotate points:
# for i, (x, y) in enumerate(zip(exp_plot["climate_change"], exp_plot["Fibre_mm"])):
#     plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
plt.scatter(pts[:, 0], pts[:, 1], s=20, alpha=0.5, label="BO trials (AI)")
pareto_pts = np.array([(cc, f) for cc, f, _ in pareto_points])
idx = np.argsort(pareto_pts[:, 0])
plt.plot(pareto_pts[idx, 0], pareto_pts[idx, 1], "r-o", label="BO Pareto front (AI)", linewidth=2)
plt.xlabel("Climate Change (t CO$_2$eq / tp)")
plt.ylabel("Fibre length (mm)")
#plt.title("CC vs Fibre Pareto (in-range only)")
plt.legend()
plt.tight_layout()
save_pareto_plot = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/figures_des/des_pareto.png')
plt.savefig(save_pareto_plot, dpi=300, bbox_inches='tight')
plt.show()
#%% To save data frames
# csv_path1 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/tables_des/df_given_data_des.csv')
# df_gd_des.to_csv(csv_path1, index=False)

# csv_path2 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/tables_des/df_material_balance_des.csv')
# df_mat_bal_DES.to_csv(csv_path2, index=True)

# csv_path3 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/tables_des/df_summary_des.csv')
# df_summary_des.to_csv(csv_path3, index=False)

# csv_path4 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/tables_des/df_des_lca.csv')
# df_des_lca.to_csv(csv_path4, index=False)

# csv_path5 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/tables_des/df_des_lca_tree.csv')
# df_des_lca_tree.to_csv(csv_path5, index=False)

# csv_path6 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/tables_des/df_weights_for_lre.csv')
# df_weights.to_csv(csv_path6, index=False)

# csv_path7 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/tables_des/df_des_ovat_ranges.csv')
# df_des_ovat_ranges.to_csv(csv_path7, index=False)

# csv_path8 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/tables_des/df_des_ovat_results.csv')
# df_des_ovat_results.to_csv(csv_path8, index=False)

# csv_path9 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/tables_des/df_mc.csv')
# df_mc.to_csv(csv_path9, index=False)

# csv_path10 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/tables_des/df_des_lca_sankey.csv')
# df_des_lca_sankey.to_csv(csv_path10, index=False)

# csv_path11 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/tables_des/df_fibre_cc.csv')
# df_fibre_cc.to_csv(csv_path11, index=False)

csv_path12 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/tables_des/df_pareto.csv')
df_pareto_ltx.to_csv(csv_path12, index=False)
#%% Contibution tree
# Load data: rows = impact categories, columns = contribution items
csv_path5 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/tables_des/df_des_lca_tree.csv')
df_tree = pd.read_csv(csv_path5)
#Summarize needed data for plotting the figure (LCA impacts)
# ensure df is available
MAX_EXPAND_NODES = 7
# category order. Manual change
categories = [
    "Acidification",
    "Climate change",
    "Climate change: biogenic",
    "Climate change: fossil",
    "Climate change: land use and land use change",
    "Ecotoxicity: freshwater",
    "Ecotoxicity: freshwater, inorganics",
    "Ecotoxicity: freshwater, organics",
    "Energy resources: non-renewable",
    "Eutrophication: freshwater",
    "Eutrophication: marine",
    "Eutrophication: terrestrial",
    "Human toxicity: carcinogenic",
    "Human toxicity: carcinogenic, inorganics",
    "Human toxicity: carcinogenic, organics",
    "Human toxicity: non-carcinogenic",
    "Human toxicity: non-carcinogenic, inorganics",
    "Human toxicity: non-carcinogenic, organics",
    "Ionising radiation: human health",
    "Land use",
    "Material resources: metals/minerals",
    "Ozone depletion",
    "Particulate matter formation",
    "Photochemical oxidant formation: human health",
    "Water use"
]
# columns. Manual change
contributions = [
    "Heat",
    "Wood",
    "Electricity",
    "Lactic Acid",
    "Choline Chloride",
    "Lignin (heat)",
    "Hemi (beet sugar)"
]
# build a matrix by category; skip the root (first) row in each block
rows = []
for cat in categories:
    arr = df_tree.loc[df_tree["impact_name"] == cat, "result"].to_numpy()
    children = arr[1:1 + MAX_EXPAND_NODES]  # exclude root
    # pad to length 7 if fewer children
    if len(children) < MAX_EXPAND_NODES:
        children = np.pad(children, (0, MAX_EXPAND_NODES - len(children)), constant_values=np.nan)
    rows.append(children)

values = np.vstack(rows)  
# make 10x7 DataFrame (rows = impact index 0..9, cols = nodes)
df_nodes = pd.DataFrame(values, index=categories, columns=contributions)
df_nodes.index.name = "impact_index"
# # save to CSV
# df_nodes.to_csv(csv_path2)
# print(round(df_nodes,4))
# Normalize so that positive values sum to 100% per category
data = df_nodes.T.to_numpy(dtype=float)                 # items x categories
pos_sums = np.where(data > 0, data, 0).sum(axis=0)
scale = np.divide(100.0, pos_sums, out=np.zeros_like(pos_sums, dtype=float), where=pos_sums != 0)
data_percents = data * scale
df_norm = pd.DataFrame(data_percents.T, index=df_nodes.index, columns=df_nodes.columns)
df_norm.index.name = "impact_name"
#If needed
# # # Exclude the "Global warming" row (robust matching: exact or contains 'global warming'/'gwp')
# idx = df_norm.index.astype(str).str.lower()
# gw_mask = (idx == 'Climate change') | idx.str.contains(r'\bgwp\b|Climate change', regex=True)
# df_plot = df_norm.loc[~gw_mask].copy()

# # Save normalized (GW-excluded) CSV
# csv_path3.parent.mkdir(parents=True, exist_ok=True)
# df_norm.to_csv(csv_path3)
# #df_plot.to_csv(csv_path3) # if gw was masked
# print(f"Saved normalized (GW-excluded) matrix to {csv_path3}")
# Plotting (horizontal stacked bars)
categories = df_norm.index.tolist()    # y-axis labels without GW
labels = df_norm.columns.tolist()      # stacked items
fig, ax = plt.subplots(figsize=(8 / 2.54, 12 / 2.54))
left_pos = np.zeros(len(categories))
left_neg = np.zeros(len(categories))
colors = [
    "#6AA9D6", "#FFB36A", "#86C67C", "#E07C7C",
    "#B39DD5", "#C49A88", "#F1A6D1", "#B9B9B9",
    "#D4D76A", "#6FD0DA", "#9AC3A4", "#F7C971"
]
for i, label in enumerate(labels):
    dp = df_norm[label].to_numpy()
    # Positive part
    dp_pos = np.clip(dp, 0, None)
    ax.barh(categories, dp_pos, label=label, left=left_pos, color=colors[i % len(colors)])
    left_pos += dp_pos
    # Negative part
    dp_neg = np.clip(dp, None, 0)
    ax.barh(categories, dp_neg, label='_nolegend_', left=left_neg, color=colors[i % len(colors)])
    left_neg += dp_neg
# Zero line and axis limits
ax.axvline(0, color='black', linewidth=1)
xmin = min(0, left_neg.min()) if len(left_neg) else 0
xmax = max(0, left_pos.max()) if len(left_pos) else 0
margin = 0.05 * max(abs(xmin), abs(xmax), 1)
ax.set_xlim(xmin - margin, xmax + margin)
# Optionally fix x-limits
ax.set_xlim(-70, 110)
# Labels and title
ax.set_xlabel('Contribution (%)', fontsize=10)
ax.set_ylabel('', fontsize=8)
# ax.set_title('EG-CO2 nodes (normalized, excluding Global warming)', fontsize=10)
ax.legend(loc='upper center', bbox_to_anchor=(0, -0.15), ncol=4, fontsize=8)
#fig.tight_layout()
# Save
# save_tree_plot = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/figures_des/des_tree.png')
# plt.savefig(save_tree_plot, dpi=300, bbox_inches='tight')
plt.show()
#%% One vertical stacked bar for Global warming
# CC value from the 10th row (0-based) in column 'result'
cc_des = df_tree['result'].iloc[8] # notice that the result should be located at index 8
# Get the values for the vertical colum
idx = df_nodes.index
# Try exact match first
cc_mask = idx.str.lower() == 'Climate change'
if not cc_mask.any():
    # Try rows that contain 'gwp' or 'global warming'
    cc_mask = idx.str.lower().str.contains('climate change|cc', regex=True)
if not cc_mask.any():
    raise ValueError("Could not find a 'Climate change' (or GWP) row in df_nodes index.")
cc_label = idx[cc_mask][0]
vals = df_nodes.loc[cc_label].to_numpy(dtype=float)  # 1 x n_items
items = df_nodes.columns.tolist()
# Colors (specified above)
# Plot: single vertical bar at x=0
fig, ax = plt.subplots(figsize=(4/2.54, 12/2.54)) # should be same height as figure above
x = np.array([0])
width = 0.6
bottom_pos = np.array([0.0])
bottom_neg = np.array([0.0])
for i, (item, v) in enumerate(zip(items, vals)):
    color = colors[i % len(colors)]
    pos = max(v, 0.0)
    neg = min(v, 0.0)
    # Positive stack
    if pos > 0:
        ax.bar(x, [pos], width=width, bottom=bottom_pos, color=color, label=item)
        bottom_pos += pos
    # Negative stack
    if neg < 0:
        ax.bar(x, [neg], width=width, bottom=bottom_neg, color=color, label='_nolegend_')
        bottom_neg += neg
# Axis formatting
# Remove x ticks and x label
ax.set_xlabel('')                           # no x label
ax.set_xticks([])                           # remove tick locations
ax.tick_params(axis='x', which='both',     # hide any remaining tick marks/labels
               bottom=False, top=False, labelbottom=False)
ax.set_ylabel('Climate change (t CO$_2$ eq per ton of pulp)')
# ax.set_title('Global warming (stacked contributions)')
ax.axhline(0, color='black', linewidth=1)
# Centered text annotation with the GW value (kgCO2eq)
y_center = 0.5 * (float(bottom_pos[0]) + float(bottom_neg[0]))
ax.text(
    x[0], y_center,
    f"CC:{cc_des:.3f}",
    ha='center', va='center',
    fontsize=7,
    color='black',
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2)
)
# Optional fixed limits (uncomment if desired)
# ax.set_ylim(-35, 110)
fig.tight_layout()
# Save
# save_cc_bar = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/figures_des/des_cc_bar.png')
# plt.savefig(save_cc_bar, dpi=300, bbox_inches='tight')
plt.show()
#%% To merge LCA figure
# Base directory for macOS (adjust if your share is mounted elsewhere)
base_dir = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/figures_des')

# Input files (left and right)
left_path = base_dir / "des_tree.png"
right_path = base_dir / "des_cc_bar.png"

# Output file (macOS path corresponding to requested location)
out_path = base_dir / "des_tree_cc_merged.png"

def main():
    # Open images with alpha preserved
    left_img = Image.open(left_path).convert("RGBA")
    right_img = Image.open(right_path).convert("RGBA")

    # Canvas dimensions: width = sum widths; height = max heights
    out_width = left_img.width + right_img.width
    out_height = max(left_img.height, right_img.height)

    # Create a transparent canvas
    merged = Image.new("RGBA", (out_width, out_height), (0, 0, 0, 0))

    # Vertically center each image on the canvas
    left_y = (out_height - left_img.height) // 2
    right_y = (out_height - right_img.height) // 2

    # Paste images side-by-side
    merged.paste(left_img, (0, left_y), left_img)
    merged.paste(right_img, (left_img.width, right_y), right_img)

    # Ensure parent exists and save as PNG
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(out_path)
    
    # Show the merged image with matplotlib
    plt.imshow(merged)
    plt.axis('off') # if on shows wierd axis
    plt.show() # ignore small boxes on bottoms. There are no on .png.
    # Alternatively, to open in the default image viewer:
    # merged.show()

    print(f"Merged image saved to: {out_path}")

if __name__ == "__main__":
    main()
#%% Radar plot: process parameters
# To load data
csv_path3 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/tables_des/df_summary_des.csv')
df_summary_des = pd.read_csv(csv_path3)
# Given data
# Derive data for EG-CO2 from df_summary...csv, for Kraft just type the values. See summary table from the manuscript.
data_sum = [
    ['Pulp yield', df_summary_des.iloc[0]['Value'], 47.0],
    ['Kappa number', df_summary_des.iloc[1]['Value'], 40.0],
    ['Fibre\nlength', df_summary_des.iloc[2]['Value'], 1.2],
    ['Wood',        df_summary_des.iloc[3]['Value'] * 0.5, 2.050],   # t/t
    ['Chemicals',   df_summary_des.iloc[4]['Value']+df_summary_des.iloc[2]['Value'], 0.009],  # t/t DES: LA+ChCl, Kraft: NaOH
    ['Water',       df_summary_des.iloc[5]['Value'], 1.64],    # t/t
    ['Heat',        df_summary_des.iloc[6]['Value'], 11.08],   # GJ/t
    ['Electricity', df_summary_des.iloc[7]['Value'], 115],# kWh/t
    ['Non fossil CO$_2$',df_summary_des.iloc[13]['Value'], 2.2]  # tCO2/t pulp   
]
# Name the columns
df_sum = pd.DataFrame(data_sum, columns=['Parameter', 'DES pulping', 'Kraft pulping'])
# If you want to fix the scenarios explicitly:
scenarios = ['DES pulping', 'Kraft pulping']
# Or derive from the data columns:
# scenarios = df.columns[1:].tolist()
# Normalization to 0–100%
USE_MINMAX = False  # set True to use min–max normalization instead
vals = df_sum.iloc[:, 1:].copy()
if USE_MINMAX:
    # Min–max per category: (x - min) / (max - min) -> [0, 1]
    minv = vals.min(axis=1)
    maxv = vals.max(axis=1)
    denom = (maxv - minv).replace(0, np.nan)
    norm01 = vals.sub(minv, axis=0).div(denom, axis=0).fillna(0.0)
else:
    # Max-abs per category: x/max(|x|) -> [-1, 1], then map to [0, 1]
    max_abs = vals.abs().max(axis=1).replace(0, np.nan)
    v = vals.div(max_abs, axis=0).fillna(0.0)  # [-1, 1]
    norm01 = (v + 1.0) / 2.0  # [0, 1]
percent = norm01 * 100.0  # [0, 100]
# Prepare radar coordinates
N = len(df_sum)  # number of parameters
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
angles_closed = np.concatenate([angles, [angles[0]]])
# Plot
cm_to_in = 1 / 2.54
fig_w_in = 12 * cm_to_in
fig_h_in = 12 * cm_to_in
fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(fig_w_in, fig_h_in))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
# Category labels
xticklabels = df_sum['Parameter'].astype(str).tolist()
ax.set_xticks(angles)
ax.set_xticklabels(xticklabels, fontsize=8)
ax.tick_params(axis="x", pad=10)
# Radial ticks 0–100%
yticks = [0, 25, 50, 75, 100]
ax.set_yticks(yticks)
ax.set_yticklabels([f"{y}%" for y in yticks], fontsize=8)
ax.set_ylim(0, 100)
ax.set_rlabel_position(90)
for lbl in ax.yaxis.get_ticklabels():
    lbl.set_rotation(90)
    lbl.set_va("center")
    lbl.set_ha("center")
ax.tick_params(axis="y", pad=6)
# Colors
colors = {"DES pulping": "#1f77b4", "Kraft pulping": "#ff7f0e"}
# Plot each scenario
for scen in scenarios:
    vals_pct = percent[scen].to_numpy()
    vals_closed = np.concatenate([vals_pct, [vals_pct[0]]])
    ax.plot(angles_closed, vals_closed, color=colors.get(scen, "C0"), linewidth=2, label=scen)
    ax.fill(angles_closed, vals_closed, color=colors.get(scen, "C0"), alpha=0.20)
# Legend
legend = ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.20), fontsize=8, frameon=True)
plt.setp(legend.get_title(), fontsize=9)
fig.tight_layout()
# save_radar_parameters = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/figures_des/des_radar_parameters.png')
# plt.savefig(save_radar_parameters, dpi=300, bbox_inches='tight')
plt.show()
#%% Radar plot: LCA impacts
csv_path4 = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/tables_des/df_des_lca.csv')
df_des_lca = pd.read_csv(csv_path4)
scenarios = ['DES pulping', 'Kraft pulping']
# Or derive from the data columns:
# scenarios = df.columns[1:].tolist()
# Normalization to 0–100%
USE_MINMAX = False  # set True to use min–max normalization instead
vals = df_des_lca[['DES pulping', 'Kraft pulping']].apply(pd.to_numeric, errors='coerce').copy()
if USE_MINMAX:
    # Min–max per category: (x - min) / (max - min) -> [0, 1]
    minv = vals.min(axis=1)
    maxv = vals.max(axis=1)
    denom = (maxv - minv).replace(0, np.nan)
    norm01 = vals.sub(minv, axis=0).div(denom, axis=0).fillna(0.0)
else:
    # Max-abs per category: x/max(|x|) -> [-1, 1], then map to [0, 1]
    max_abs = vals.abs().max(axis=1).replace(0, np.nan)
    v = vals.div(max_abs, axis=0).fillna(0.0)  # [-1, 1]
    norm01 = (v + 1.0) / 2.0  # [0, 1]
percent = norm01 * 100.0  # [0, 100]
# Prepare radar coordinates
N = len(df_des_lca)  # number of parameters
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
angles_closed = np.concatenate([angles, [angles[0]]])
# Plot
cm_to_in = 1 / 2.54
fig_w_in = 12 * cm_to_in
fig_h_in = 12 * cm_to_in
fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(fig_w_in, fig_h_in))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
categories = [
    "Acidification",
    "Climate change (CC)",
    "CC: biogenic",
    "CC: fossil",
    "CC: land use",
    "EcoTox: freshwater",
    "EcoTox: freshwater, inor.",
    "EcoTox: freshwater, org.",
    "EnerRes: non-renewable",
    "Eutroph.: freshwater",
    "Eutroph.: marine",
    "Eutroph.: terrestrial",
    "HumTox: carc.",
    "HumTox: carc., inor.",
    "HumTox: carc., org.",
    "HumTox: non-carc.",
    "HumTox: non-carc., inor.",
    "HumTox: non-carc., org.",
    "IonRad: human health",
    "Land use",
    "MatRes: metals/minerals",
    "O3 depletion",
    "PartMatForm",
    "PhotoOxForm: human health",
    "Water use"
]
# Category labels
xticklabels = [str(i+1) for i in range(len(df_des_lca))]
ax.set_xticklabels(xticklabels, fontsize=8)#df_des_lca.index.astype(str).tolist()
ax.set_xticks(angles)
ax.set_xticklabels(xticklabels, fontsize=8)
ax.tick_params(axis="x", pad=10)
# Radial ticks 0–100%
yticks = [0, 25, 50, 75, 100]
ax.set_yticks(yticks)
ax.set_yticklabels([f"{y}%" for y in yticks], fontsize=8)
ax.set_ylim(0, 100)
ax.set_rlabel_position(90)
for lbl in ax.yaxis.get_ticklabels():
    lbl.set_rotation(90)
    lbl.set_va("center")
    lbl.set_ha("center")
ax.tick_params(axis="y", pad=6)
# Colors
colors = {"DES pulping": "#1f77b4", "Kraft pulping": "#ff7f0e"}
# Plot each scenario
for scen in scenarios:
    vals_pct = percent[scen].to_numpy()
    vals_closed = np.concatenate([vals_pct, [vals_pct[0]]])
    ax.plot(angles_closed, vals_closed, color=colors.get(scen, "C0"), linewidth=2, label=scen)
    ax.fill(angles_closed, vals_closed, color=colors.get(scen, "C0"), alpha=0.20)
# Legend
legend = ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=8, frameon=False)
# disclosing numbers
mapping_text = "\n".join(f"{i+1:>2}. {name}" for i, name in enumerate(categories))
fig.text(
    1.15, 0.5, mapping_text,
    va='center', ha='left', fontsize=8, family='monospace',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='0.8'),
    transform=ax.transAxes
)

plt.setp(legend.get_title(), fontsize=9)
fig.tight_layout()
# save_radar_impacts = Path('/Volumes/ponoman1/data/Aalto/For Projects/EFP/IL_DES/figures_des/des_radar_impacts.png')
# plt.savefig(save_radar_impacts, dpi=300, bbox_inches='tight')
plt.show()
