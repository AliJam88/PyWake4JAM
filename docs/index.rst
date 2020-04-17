.. PyWake documentation master file, created by
   sphinx-quickstart on Mon Dec  3 13:24:21 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: logo.svg
    :align: center

Welcome to PyWake
===========================================

*- an open source wind farm simulation tool capable of calculating wind farm flow fields, power production and annual energy production (AEP) of wind farms.*


**Quick Start**::

    pip install py_wake

Source code repository (and issue tracker):
    https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake
    
License:
    MIT_

.. _MIT: https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake/blob/master/LICENSE


Contents:
    .. toctree::
        :maxdepth: 2
    
        installation
        notebooks/Introduction  
        notebooks/ChangeLog
        
        
    .. toctree::
        :caption: Tutorials
       
        notebooks/Quickstart
        notebooks/Site
        notebooks/WindTurbines
        notebooks/EngineeringWindFarmModels
        notebooks/RunWindFarmSimulation
        
   
    .. toctree::
        :maxdepth: 2
        :caption: Validation  
    
        validation
        
    .. toctree::
        :caption: API  
            
        api/WindTurbines
        api/Site
        api/WindFarmModel
        api/EngineeringWindFarmModels
        api/PredefinedEngineeringWindFarmModels
        api/SimulationResult
        api/FlowMap
        