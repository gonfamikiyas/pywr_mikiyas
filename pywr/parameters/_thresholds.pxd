from ._parameters cimport IndexParameter, Parameter
from pywr.recorders._recorders cimport Recorder
from .._core cimport Timestep, Scenario, ScenarioIndex, AbstractNode, AbstractStorage
cimport numpy as np
ctypedef np.uint8_t uint8


cdef class AgregatedCostThresholdRecorder_test2(IndexParameter):
    cdef public double _threshold
    cdef public Parameter _threshold_parameter
    cdef double[:] values
    cdef int predicate
    cdef public bint ratchet
    cdef uint8[:] _triggered

    cdef public Recorder recorder1
    cdef public Recorder recorder2
    cdef public Parameter threshold1
    cdef public Parameter threshold2
    cdef public initial_value
    cdef public triggered_, triggered, data

    cdef public Recorder recorder
    cdef double gen_cost_rec_value, EDC_rec_value, gen_cost_threshold,EDC_threshold
  
    cpdef double _value_to_compare_cost(self, Timestep timestep, ScenarioIndex scenario_index) except? -1
    cpdef double _value_to_compare_EDC(self, Timestep timestep, ScenarioIndex scenario_index) except? -1





cdef class AgregatedCostThresholdRecorder_test3(IndexParameter):
    cdef public double _threshold
    cdef public Parameter _threshold_parameter
    cdef double[:] values
    cdef int predicate
    cdef public bint ratchet
    cdef uint8[:] _triggered

    cdef public Recorder recorder1
    cdef public Recorder recorder2
    cdef public Parameter threshold1
    cdef public Parameter threshold2
    cdef public initial_value
    cdef public triggered_, triggered, data

    cdef public Recorder recorder
    cdef double gen_cost_rec_value, EDC_rec_value, gen_cost_threshold,EDC_threshold
  
    cpdef double _value_to_compare_cost(self, Timestep timestep, ScenarioIndex scenario_index) except? -1
    cpdef double _value_to_compare_EDC(self, Timestep timestep, ScenarioIndex scenario_index) except? -1


cdef class AbstractThresholdParameter(IndexParameter):
    cdef public double _threshold
    cdef public Parameter _threshold_parameter
    cdef double[:] values
    cdef int predicate
    cdef public bint ratchet
    cdef uint8[:] _triggered
    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1

cdef class StorageThresholdParameter(AbstractThresholdParameter):
    cdef public AbstractStorage storage

cdef class NodeThresholdParameter(AbstractThresholdParameter):
    cdef public AbstractNode node

cdef class ParameterThresholdParameter(AbstractThresholdParameter):
    cdef public Parameter param

cdef class RecorderThresholdParameter(AbstractThresholdParameter):
    cdef public Recorder recorder
    cdef public initial_value


cdef class AgregatedCostThresholdRecorder_test(AbstractThresholdParameter):
    cdef public Recorder recorder
    cdef public initial_value



cdef class AgregatedThresholdRecorder(AbstractThresholdParameter):
    cdef public Recorder recorder
    cdef public initial_value

cdef class AgregatedCostThresholdRecorder(AbstractThresholdParameter):
    cdef public Recorder recorder1
    cdef public Recorder recorder2
    cdef public initial_value






cdef class AgregatedCostThresholdParameter(AbstractThresholdParameter):
    cdef public Parameter recorder1
    cdef public Parameter recorder2
    cdef public initial_value

cdef class CurrentYearThresholdParameter(AbstractThresholdParameter):
    pass

cdef class CurrentOrdinalDayThresholdParameter(AbstractThresholdParameter):
    pass
