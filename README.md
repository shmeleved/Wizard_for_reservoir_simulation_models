# Wizard for reservoir simulation models

Main libraries:
* pandas
* numpy
* PyQt5
* plotly
* seaborn
* os


The application allows you to:
* read and decrypt the resulting reservoir simulation model calculation files (binary formats)
* automatically create development maps and upload them in exchange format (CPS-3)
* upload various model reports (Excel, PowerPoint)
* download the necessary data from the near-well zone 

**Which significantly reduces the routine load on the development engineer**

## [path_reader.py](https://github.com/shmeleved/Wizard_for_building_maps_and_reports_of_reservoir_simulation_models/blob/main/path_reader.py)
Contains launching the program and reading the path to the ".DATA" file

## [plot_export.py](https://github.com/shmeleved/Wizard_for_building_maps_and_reports_of_reservoir_simulation_models/blob/main/plot_export.py)
Contains code that allows you to graphically display all the available functionality of the application

## [data_generation.py](https://github.com/shmeleved/Wizard_for_building_maps_and_reports_of_reservoir_simulation_models/blob/main/data_generation.py)
Contains all the functions that allow you to process and create all the data
