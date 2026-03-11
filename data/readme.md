
# Training data for ANNs

These are the conventions as per CalSim ANN training.

| Variable Name    | Description                                                                   |
|------------------|-------------------------------------------------------------------------------|
| datetime         | year-month-day  (all values are daily-averaged)|
| model            | Modeling platform used for this dataset (e.g. SCHISM, DSM2, RMA)   |
| scene            | Scenario being modeled - geographically (e.g. Baseline, Suisun Suite, Cache Suite)   |
| case             | Case number being referenced which affects time-series inputs   |
| northern_flow    | Sacramento (below Feather) + American + Yolo Bypass and Toe Drain + Mokelumne + Cosumnes + Calaveras - North Bay Aqueduct (cfs) |
| sac_flow         | Sacramento River inflow at I St - includes American River upstream (cfs) |
| sjr_flow         | San Joaquin River inflow at Vernalis (cfs) |
| exports          | Banks + Jones + CCC Rock Slough + CCC Middle/Old + CCC Victoria (cfs)|
| cu_total         | Net Delta Consumptive Use, for SCHISM includes evaporation (cfs) |
| cu_delta         | Net Delta Consumptive Use only in Delta, for SCHISM includes evaporation (cfs) |
| cu_suisun        | Net Delta Consumptive Use only in Suisun, for SCHISM includes evaporation(cfs) |
| ndo              | Net Delta Outflow as calculated by boundary inflows (cfs) |
| dcc              | Delta Cross Channel gate operations (0=Closed, 1=Open) |
| smscg            | Suisun Marsh Salinity Control Gate operations (0=Open, 1=Tidally Operated) |
| sf_tidal_energy  | Tidal energy of SFFPX station/boundary. Calculated over stage in feet. < (z- <z>)^2 >, where <> is a low-pass filter |
| sf_tidal_filter  | Subtidal variation around mean of SFFPX station/boundary. Calculated over stage in feet. <z>-z̄, where <> is a low-pass filter, and z̄ is the mean stage over the period of 1990-2020 (0.98 m or 3.22 ft). Note that for SCHISM, this is derived from observed data at SFFPX station, not model output.|
| x2               | X2 calculated from model EC results |
| ec locations     | trp,wci,vcu,rsl,old,rri2,bdt,lps,snc,dsj,bdl,nsl2,vol,tss,sss,oh4,god,bac,hol,mtz,tms,gzl,rsl |


Sign convention for flow is positive generally. For inflows it's positive when flowing **into** the domain, and for exports and consumptive use it's positive when flowing **out** of the domain.
