import opendssdirect as dss
import os

dss.run_command("Redirect Run_IEEE14Bus.dss")
dss.run_command("Export Voltages IEEE_14_EXP_VOLTAGES.csv")
dss.run_command("Export Powers IEEE_14_EXP_POWERS.csv")

with open("IEEE_14_EXP_VOLTAGES.csv", "r") as f:
    print(f.read())

with open("IEEE_14_EXP_POWERS.csv", "r") as f:
    print(f.read())
