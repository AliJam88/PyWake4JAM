WindTurbine classes
===================

.. inheritance-diagram:: py_wake.wind_turbines.OneTypeWindTurbines
    :parts: 1

- `WindTurbine`_ is the base class that allows multiple wind turbine types, 
- `OneTypeWindTurbines`_ subclass allowing multiple wind turbines but only type
    
    
WindTurbine
------------


.. autoclass:: py_wake.wind_turbines.WindTurbine
    :members:
    
       
    .. autosummary::
        __init__
        name
        hub_height
        diameter
        power
        ct
        plot
        from_WindTurbines
        from_WAsP_wtg
        
    .. automethod:: __init__
    
    
   
OneTypeWindTurbines
-------------------
.. autoclass:: py_wake.wind_turbines.OneTypeWindTurbines
    :members:
    

     
    .. automethod:: __init__