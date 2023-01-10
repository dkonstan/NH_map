# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle


NHmapFile = "NHmap_51points_cpcm.xlsx"
print("loading data...")
dfHB = pd.read_excel(NHmapFile, sheet_name="hbonded", engine="openpyxl")
dfFree = pd.read_excel(NHmapFile, sheet_name="free", engine="openpyxl")

NHScaling = (3498 / 3495.83738886008)  # gas phase NMA got 3495.83738886008 from FGH, experimental value 3498 (Mukamel)

# plt.hist(dfHB["freq01"] * NHScaling, alpha=0.5, bins=15, label="H-bonded", color="r")
# plt.hist(dfFree["freq01"] * NHScaling, alpha=0.5, bins=15, label="free", color="b")
# plt.xlabel("Frequency (cm$^{-1}$)")
# plt.ylabel("Count")
# plt.legend()
# print(np.mean(dfHB["freq01"] * NHScaling))
# print(np.mean(dfFree["freq01"] * NHScaling))
# exit()
# plt.show()
# exit()
# ------------------------------------
#       CARBONYL-BOUND NH MAP
# ------------------------------------

# nSamples x nFeatures
freqHB = dfHB["freq01"] * NHScaling

# starts in km^(1/2) / mol^(1/2)
val1 = 0.153835049   # sqrt(D^2/A^2.amu) / [km^(1/2) / mol^(1/2)] -> D/(A * sqrt(amu))
val2 = 0.393456  # e-bohr / D   -> e-bohr /(A * sqrt(amu))
val3 = 0.52917721092  # A / bohr -> e-bohr / (bohr * sqrt(amu))
val4 = 1 / np.sqrt(1822.8885)  # sqrt(amu) / sqrt(erm) -> e-bohr / (bohr * sqrt(erm)) = e / sqrt(erm)
reducedMassAU = 0.9403 * 1822.8885  # amu * (erm / amu) -> erm
val5 = np.sqrt(reducedMassAU)  # sqrt(erm)

muHB = dfHB["mu"] * val1 * val2 * val3 * val4 * val5  # "atomic units" (no mass weighting)

regressionX = LinearRegression()
regressionX.fit(freqHB.values.reshape(1, -1).T, dfHB["x01"])
rSquaredX = regressionX.score(freqHB.values.reshape(1, -1).T, dfHB["x01"])
interceptX = regressionX.intercept_
coefX = regressionX.coef_
print("x01 vs freq, hb")
print("\tintercept:", interceptX)
print("\tcoefficients:", list(coefX))
print("\tr^2:", rSquaredX)

dfHB = dfHB.drop(columns=["freq01", "freq02", "mu", "x01", "p01"])
# cov = np.cov(dfHB.T)
# cov /= np.diag(cov)
# covMat = plt.matshow(cov)
# plt.colorbar(covMat)
# plt.show()

electricFieldHB = np.array(dfHB.values)
regressionHB = RandomForestRegressor()
regressionHB.fit(electricFieldHB, freqHB)
rSquaredHB = regressionHB.score(electricFieldHB, freqHB)
# a = regressionHB.predict([1000000 * np.random.rand(153) - 0.5])
print("carbonyl-bound NH parameters:")
print("\tr^2:", rSquaredHB)
pickle.dump(regressionHB, open("regressionHbonded_frequency.pkl", "wb"))

regressionHB = RandomForestRegressor()
regressionHB.fit(electricFieldHB, muHB)
rSquaredHB = regressionHB.score(electricFieldHB, muHB)
print("carbonyl-bound NH parameters (mu):")
print("\tr^2:", rSquaredHB)
pickle.dump(regressionHB, open("regressionHbonded_dipolederivative.pkl", "wb"))

# ------------------------------------
#       'FREE' NH MAP
# ------------------------------------

freqFree = dfFree["freq01"] * NHScaling
muFree = dfFree["mu"] * val1 * val2 * val3 * val4 * val5  # "atomic units" (no mass though...)


regressionX = LinearRegression()
regressionX.fit(freqFree.values.reshape(1, -1).T, dfFree["x01"])
rSquaredX = regressionX.score(freqFree.values.reshape(1, -1).T, dfFree["x01"])
interceptX = regressionX.intercept_
coefX = regressionX.coef_
print("x01 vs freq, free")
print("\tintercept:", interceptX)
print("\tcoefficients:", list(coefX))
print("\tr^2:", rSquaredX)

dfFree = dfFree.drop(columns=["freq01", "freq02", "mu", "x01", "p01"])

electricFieldFree = np.array(dfFree.values)
regressionFree = RandomForestRegressor()
regressionFree.fit(electricFieldFree, freqFree)
rSquaredFree = regressionFree.score(electricFieldFree, freqFree)
print("'free' NH parameters:")
print("\tr^2:", rSquaredFree)
pickle.dump(regressionFree, open("regressionFree_frequency.pkl", "wb"))

regressionFree = RandomForestRegressor()
regressionFree.fit(electricFieldFree, muFree)
rSquaredFree = regressionFree.score(electricFieldFree, muFree)
print("'free' NH parameters (mu):")
print("\tr^2:", rSquaredFree)
pickle.dump(regressionFree, open("regressionFree_dipolederivative.pkl", "wb"))

inputData = [np.random.rand(153)]
print("testing pickling...")
print("\t original object prediction:")
print(regressionFree.predict(inputData))
regressionTest = pickle.load(open("regressionFree_dipolederivative.pkl", "rb"))
print("\t pickled object prediction")
print(regressionTest.predict(inputData))
