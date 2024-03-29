# Repository accompanying the manuscript Leveraging continuous glucose monitoring for personalized modeling of insulin-regulated glucose metabolism.

The repository contains Julia code to analyse postprandial glucose and insulin responses of individuals after a meal challenge using the EDES model.

## Files
- data.csv          Contains example experimental data used in calibrating the EDES model.
- fit_model.jl      Calibrates the model on individuals' meal challenge data. Plots the simulation and saves the estimated parameters, the simulation over time, and the residuals to file.
- func.jl           Contains auxiliary functions used in 'fit_model.jl'.

## Data Availability Statement
Data from the PERSON Study are unsuitable for public deposition due to ethical restriction and privacy of participant data. Data are available to researchers meeting the criteria for access to confidential data. Gabby Hul on behalf of the data management team of the department of Human Biology may be contacted at g.hul@maastrichtuniversity.nl to request the PERSON Study data.
