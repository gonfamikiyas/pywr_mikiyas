from ._parameters import load_parameter
cimport numpy as np
import numpy as np

cdef enum Predicates:
    LT = 0
    GT = 1
    EQ = 2
    LE = 3
    GE = 4
_predicate_lookup = {
    "LT": Predicates.LT, "<": Predicates.LT,
    "GT": Predicates.GT, ">": Predicates.GT,
    "EQ": Predicates.EQ, "=": Predicates.EQ,
    "LE": Predicates.LE, "<=": Predicates.LE,
    "GE": Predicates.GE, ">=": Predicates.GE,
}







cdef class AbstractThresholdParameter(IndexParameter):
    """ Base class for parameters returning one of two values depending on other state.

    Parameters
    ----------
    threshold : double or Parameter
        Threshold to compare the value of the recorder to
    values : iterable of doubles
        If the predicate evaluates False the zeroth value is returned,
        otherwise the first value is returned.
    predicate : string
        One of {"LT", "GT", "EQ", "LE", "GE"}.
    ratchet : bool
        If true the parameter behaves like a ratchet. Once it is triggered first
        it stays in the triggered position (default=False).

    Methods
    -------
    value(timestep, scenario_index)
        Returns a value from the `values` attribute, using the index.
    index(timestep, scenario_index)
        Returns 1 if the predicate evaluates True, else 0.

    Notes
    -----
    On the first day of the model run the recorder will not have a value for
    the previous day. In this case the predicate evaluates to True.

    """
    def __init__(self, model, threshold, *args, values=None, predicate=None, ratchet=False, **kwargs):
        super(AbstractThresholdParameter, self).__init__(model, *args, **kwargs)
        self.threshold = threshold
        if values is None:
            self.values = None
        else:
            self.values = np.array(values, np.float64)
        if predicate is None:
            predicate = Predicates.LT
        elif isinstance(predicate, str):
            predicate = _predicate_lookup[predicate.upper()]
        self.predicate = predicate
        self.ratchet = ratchet

    cpdef setup(self):
        super(AbstractThresholdParameter, self).setup()
        cdef int ncomb = len(self.model.scenarios.combinations)
        self._triggered = np.empty(ncomb, dtype=np.uint8)

    cpdef reset(self):
        super(AbstractThresholdParameter, self).reset()
        self._triggered[...] = 0

    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        raise NotImplementedError()

    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns a value from the values attribute, using the index"""
        cdef int ind = self.get_index(scenario_index)
        cdef double v
        if self.values is not None:
            v = self.values[ind]
        else:
            return np.nan
        return v

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns 1 if the predicate evalutes True, else 0"""
        cdef double x
        cdef bint ind, triggered

        triggered = self._triggered[scenario_index.global_id]

        # Return triggered state if ratchet is enabled.
        if self.ratchet and triggered:
            return triggered

        x = self._value_to_compare(timestep, scenario_index)

        cdef double threshold
        if self._threshold_parameter is not None:
            threshold = self._threshold_parameter.value(timestep, scenario_index)
        else:
            threshold = self._threshold

        if self.predicate == Predicates.LT:
            ind = x < threshold
        elif self.predicate == Predicates.GT:
            ind = x > threshold
        elif self.predicate == Predicates.LE:
            ind = x <= threshold
        elif self.predicate == Predicates.GE:
            ind = x >= threshold
        else:
            ind = x == threshold

        self._triggered[scenario_index.global_id] = max(ind, triggered)
        return ind

    property threshold:
        def __get__(self):
            if self._threshold_parameter is not None:
                return self._threshold_parameter
            else:
                return self._threshold

        def __set__(self, value):
            if self._threshold_parameter is not None:
                self.children.remove(self._threshold_parameter)
                self._threshold_parameter = None
            if isinstance(value, Parameter):
                self._threshold_parameter = value
                self.children.add(self._threshold_parameter)
            else:
                self._threshold = value




cdef class AgregatedCostThresholdRecorder_test2(IndexParameter):
    """Returns one of two values depending on a Recorder value and a threshold

    Parameters
    ----------
    recorder : `pywr.recorder.Recorder`

    """

    def __init__(self,  model,  Recorder recorder1, Recorder recorder2, Parameter threshold1, Parameter threshold2, *args, values=None, predicate=None, ratchet=False, initial_value=1, **kwargs):
        super(AgregatedCostThresholdRecorder_test2, self).__init__(model, *args, **kwargs)

        if values is None:
            self.values = None
        else:
            self.values = np.array(values, np.float64)
        self.recorder1 = recorder1
        self.recorder2 = recorder2
        self._count = 0

        self.threshold1 = threshold1
        self.threshold2 = threshold2

        self.recorder1.parents.add(self)
        self.recorder2.parents.add(self)
        self.initial_value = initial_value


    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns a value from the values attribute, using the index"""
        cdef int ind = self.get_index(scenario_index)
        cdef double v
        if self.values is not None:
            v = self.values[ind]
        else:
            return np.nan
        return v
    cpdef setup(self):
        super(AgregatedCostThresholdRecorder_test2, self).setup()
        cdef int nts = len(self.model.timestepper)
        cdef int ncomb = len(self.model.scenarios.combinations)
        self.triggered = np.empty((ncomb), dtype=np.uint8)
        year = 1+self.model.timestepper.end.year-self.model.timestepper.start.year
        self.triggered_ = np.empty((ncomb,int(24*12*year)), dtype=np.uint8)
        self.data = np.zeros((int(24*12*year), ncomb), dtype=np.uint8)
        


    cpdef reset(self):
        super(AgregatedCostThresholdRecorder_test2, self).reset()
        self.triggered_[...] = 0
        self.triggered[...] = 0
        self.data[...] = 0

    cpdef double _value_to_compare_cost(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        # TODO Make this a more general API on Recorder
        gen_cost_val = int(np.asarray(self.recorder1.diff)[scenario_index.global_id])
        return gen_cost_val

    cpdef double _value_to_compare_EDC(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        # TODO Make this a more general API on Recorder
        EDC = int(np.asarray(self.recorder2.defc)[scenario_index.global_id])
        return EDC


    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
      
        """Returns 1 if the predicate evalutes True, else 0"""
        
        cdef double gen_cost_rec_value, EDC_rec_value, gen_cost_threshold,EDC_threshold
        cdef int index
        
        cdef bint ind, triggered
        if self._count >= len(self.data):
            self._count = 0
        index=self._count


        if index == 0:
            ind = self.initial_value

        trigger_years = [1977,1982,1987]

        if timestep.year in trigger_years and timestep.month==12:
            self.triggered[scenario_index.global_id]=(self.triggered_[scenario_index.global_id][index-1440:index]).max(axis=0)
        elif timestep.year==1974 and timestep.month==12:
            self.triggered[scenario_index.global_id]=(self.triggered_[scenario_index.global_id][index-864:index]).max(axis=0)

        gen_cost_rec_value = self._value_to_compare_cost(timestep, scenario_index)
        EDC_rec_value = self._value_to_compare_EDC(timestep, scenario_index)
        
        
        if gen_cost_rec_value<=0:
            gen_cost_rec_value=0


        gen_cost_threshold = self.threshold1.value(timestep, scenario_index)
        EDC_threshold = self.threshold2.value(timestep, scenario_index)

        if gen_cost_rec_value >= gen_cost_threshold or EDC_rec_value >= EDC_threshold:
            ind = 1
        elif gen_cost_rec_value <= gen_cost_threshold and EDC_rec_value <= EDC_threshold:
            ind = 0


        self.triggered_[scenario_index.global_id,index] = ind
        self.data[index,scenario_index.global_id] = ind

        self._count+=1
        
        
        return self.triggered[scenario_index.global_id]

    @classmethod
    def load(cls, model, data):
        from pywr.recorders._recorders import load_recorder  # delayed to prevent circular reference
        recorder1 = load_recorder(model, data["recorder"][0])
        recorder2 = load_recorder(model, data.pop("recorder")[1])

        
        threshold1 = load_parameter(model, data["threshold"][0])
        threshold2 = load_parameter(model, data.pop("threshold")[1])
        values = data.pop("values", None)
        predicate = data.pop("predicate", None)
        ratchet = data.pop("ratchet", None)

        return cls(model, recorder1,recorder2, threshold1, threshold2, predicate=predicate, **data)
AgregatedCostThresholdRecorder_test2.register()



cdef class AgregatedCostThresholdRecorder_test3(IndexParameter):
    """Returns one of two values depending on a Recorder value and a threshold

    Parameters
    ----------
    recorder : `pywr.recorder.Recorder`

    """

    def __init__(self,  model,  Recorder recorder1, Recorder recorder2, Parameter threshold1, Parameter threshold2, *args, values=None, predicate=None, ratchet=False, initial_value=1, **kwargs):
        super(AgregatedCostThresholdRecorder_test3, self).__init__(model, *args, **kwargs)

        if values is None:
            self.values = None
        else:
            self.values = np.array(values, np.float64)
        self.recorder1 = recorder1
        self.recorder2 = recorder2
        self._count = 0

        self.threshold1 = threshold1
        self.threshold2 = threshold2

        self.recorder1.parents.add(self)
        self.recorder2.parents.add(self)
        self.initial_value = initial_value


    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns a value from the values attribute, using the index"""
        cdef int ind = self.get_index(scenario_index)
        cdef double v
        if self.values is not None:
            v = self.values[ind]
        else:
            return np.nan
        return v
    cpdef setup(self):
        super(AgregatedCostThresholdRecorder_test3, self).setup()
        cdef int nts = len(self.model.timestepper)
        cdef int ncomb = len(self.model.scenarios.combinations)
        self.triggered = np.empty((ncomb), dtype=np.uint8)
        year = 1+self.model.timestepper.end.year-self.model.timestepper.start.year
        self.triggered_ = np.empty((ncomb,int(24*12*year)), dtype=np.uint8)
        self.data = np.zeros((int(24*12*year), ncomb), dtype=np.uint8)

    cpdef reset(self):
        super(AgregatedCostThresholdRecorder_test3, self).reset()
        self.triggered_[...] = 0
        self.triggered[...] = 0
        self.data[...] = 0

    cpdef double _value_to_compare_cost(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        # TODO Make this a more general API on Recorder
        gen_cost_val = int(np.asarray(self.recorder1.diff)[scenario_index.global_id])
        return gen_cost_val

    cpdef double _value_to_compare_EDC(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        # TODO Make this a more general API on Recorder
        EDC = int(np.asarray(self.recorder2.defc)[scenario_index.global_id])
        return EDC


    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
      
        """Returns 1 if the predicate evalutes True, else 0"""
        
        cdef double gen_cost_rec_value, EDC_rec_value, gen_cost_threshold,EDC_threshold
        cdef int index
        
        cdef bint ind, triggered
        if self._count >= len(self.data):
            self._count = 0
        index=self._count

        if index == 0:
            ind = self.initial_value

        trigger_years = [1977,1982,1987]

        if timestep.year in trigger_years and timestep.month==12:
            self.triggered[scenario_index.global_id]=(self.triggered_[scenario_index.global_id][index-1440:index]).max(axis=0)
        elif timestep.year==1974 and timestep.month==12:
            self.triggered[scenario_index.global_id]=(self.triggered_[scenario_index.global_id][index-864:index]).max(axis=0)

        gen_cost_rec_value = self._value_to_compare_cost(timestep, scenario_index)*-1
        EDC_rec_value = self._value_to_compare_EDC(timestep, scenario_index)
        
        if gen_cost_rec_value<=0:
            gen_cost_rec_value=0
        gen_cost_threshold = self.threshold1.value(timestep, scenario_index)
        EDC_threshold = self.threshold2.value(timestep, scenario_index)

        if gen_cost_rec_value >= gen_cost_threshold or EDC_rec_value >= EDC_threshold:
            ind = 1
        elif gen_cost_rec_value <= gen_cost_threshold and EDC_rec_value <= EDC_threshold:
            ind = 0


        self.triggered_[scenario_index.global_id,index] = ind
        self.data[index,scenario_index.global_id] = ind
        self._count+=1
        
        
        return self.triggered[scenario_index.global_id]

    @classmethod
    def load(cls, model, data):
        from pywr.recorders._recorders import load_recorder  # delayed to prevent circular reference
        recorder1 = load_recorder(model, data["recorder"][0])
        recorder2 = load_recorder(model, data.pop("recorder")[1])

        
        threshold1 = load_parameter(model, data["threshold"][0])
        threshold2 = load_parameter(model, data.pop("threshold")[1])
        values = data.pop("values", None)
        predicate = data.pop("predicate", None)
        ratchet = data.pop("ratchet", None)


        return cls(model, recorder1,recorder2, threshold1, threshold2, predicate=predicate, **data)
AgregatedCostThresholdRecorder_test3.register()



cdef class StorageThresholdParameter(AbstractThresholdParameter):
    """ Returns one of two values depending on current volume in a Storage node

    Parameters
    ----------
    recorder : `pywr.core.AbstractStorage`

    """
    def __init__(self, model, AbstractStorage storage, *args, **kwargs):
        super(StorageThresholdParameter, self).__init__(model, *args, **kwargs)
        self.storage = storage

    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        return self.storage._volume[scenario_index.global_id]

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("storage_node"))
        threshold = load_parameter(model, data.pop("threshold"))
        values = data.pop("values", None)
        predicate = data.pop("predicate", None)
        return cls(model, node, threshold, values=values, predicate=predicate, **data)
StorageThresholdParameter.register()


cdef class NodeThresholdParameter(AbstractThresholdParameter):
    """ Returns one of two values depending on previous flow in a node

    Parameters
    ----------
    recorder : `pywr.core.AbstractNode`

    """
    def __init__(self, model, AbstractNode node, *args, **kwargs):
        super(NodeThresholdParameter, self).__init__(model, *args, **kwargs)
        self.node = node

    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        return self.node._prev_flow[scenario_index.global_id]

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        if timestep.index == 0:
            # previous flow on initial timestep is undefined
            return 0
        return AbstractThresholdParameter.index(self, timestep, scenario_index)

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        threshold = load_parameter(model, data.pop("threshold"))
        values = data.pop("values", None)
        predicate = data.pop("predicate", None)
        return cls(model, node, threshold, values=values, predicate=predicate, **data)
NodeThresholdParameter.register()


cdef class ParameterThresholdParameter(AbstractThresholdParameter):
    """ Returns one of two values depending on the value of a Parameter

    Parameters
    ----------
    recorder : `pywr.core.AbstractNode`

    """
    def __init__(self, model, Parameter param, *args, **kwargs):
        super(ParameterThresholdParameter, self).__init__(model, *args, **kwargs)
        self.param = param
        self.children.add(param)

    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        return self.param.get_value(scenario_index)

    @classmethod
    def load(cls, model, data):
        param = load_parameter(model, data.pop('parameter'))
        threshold = load_parameter(model, data.pop("threshold"))
        values = data.pop("values", None)
        predicate = data.pop("predicate", None)
        return cls(model, param, threshold, values=values, predicate=predicate, **data)
ParameterThresholdParameter.register()






cdef class RecorderThresholdParameter(AbstractThresholdParameter):
    """Returns one of two values depending on a Recorder value and a threshold

    Parameters
    ----------
    recorder : `pywr.recorder.Recorder`

    """

    def __init__(self,  model, Recorder recorder, *args, initial_value=1, **kwargs):
        super(RecorderThresholdParameter, self).__init__(model, *args, **kwargs)
        self.recorder = recorder
        self.recorder.parents.add(self)
        self.initial_value = initial_value

    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        # TODO Make this a more general API on Recorder
        return self.recorder.data[timestep.index - 1, scenario_index.global_id]

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns 1 if the predicate evalutes True, else 0"""
        cdef int index = timestep.index
        cdef int ind
        if index == 0:
            # on the first day the recorder doesn't have a value so we have no
            # threshold to compare to
            ind = self.initial_value
        else:
            ind = super(RecorderThresholdParameter, self).index(timestep, scenario_index)
        return ind

    @classmethod
    def load(cls, model, data):
        from pywr.recorders._recorders import load_recorder  # delayed to prevent circular reference
        recorder = load_recorder(model, data.pop("recorder"))
        threshold = load_parameter(model, data.pop("threshold"))
        values = data.pop("values", None)
        predicate = data.pop("predicate", None)
        return cls(model, recorder, threshold, values=values, predicate=predicate, **data)
RecorderThresholdParameter.register()






cdef class AgregatedCostThresholdRecorder_test(AbstractThresholdParameter):
    """Returns one of two values depending on a Recorder value and a threshold

    Parameters
    ----------
    recorder : `pywr.recorder.Recorder`

    """

    def __init__(self,  model, Recorder recorder, *args, initial_value=1, **kwargs):
        super(AgregatedCostThresholdRecorder_test, self).__init__(model, *args, **kwargs)
        self.recorder = recorder
        self.recorder.parents.add(self)
        self.initial_value = initial_value

    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        # TODO Make this a more general API on Recorder
        return self.recorder.defc[scenario_index.global_id]

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns 1 if the predicate evalutes True, else 0"""
        cdef int index = timestep.index
        cdef int ind
        if index == 0:
            # on the first day the recorder doesn't have a value so we have no
            # threshold to compare to
            ind = self.initial_value
        else:
            ind = super(AgregatedCostThresholdRecorder_test, self).index(timestep, scenario_index)
        return ind

    @classmethod
    def load(cls, model, data):
        from pywr.recorders._recorders import load_recorder  # delayed to prevent circular reference
        recorder = load_recorder(model, data.pop("recorder"))
        threshold = load_parameter(model, data.pop("threshold"))
        values = data.pop("values", None)
        predicate = data.pop("predicate", None)
        return cls(model, recorder, threshold, values=values, predicate=predicate, **data)
AgregatedCostThresholdRecorder_test.register()





cdef class AgregatedThresholdRecorder(AbstractThresholdParameter):
    """Returns one of two values depending on a Recorder value and a threshold

    Parameters
    ----------
    recorder : `pywr.recorder.Recorder`

    """

    def __init__(self,  model, Recorder recorder, *args, initial_value=1, **kwargs):
        super(AgregatedThresholdRecorder, self).__init__(model, *args, **kwargs)
        self.recorder = recorder
        self.recorder.parents.add(self)
        self.initial_value = initial_value

    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        # TODO Make this a more general API on Recorder
        return np.array(self.recorder.values())[scenario_index.global_id]

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns 1 if the predicate evalutes True, else 0"""
        cdef int index = timestep.index
        cdef int ind
        if index == 0:
            # on the first day the recorder doesn't have a value so we have no
            # threshold to compare to
            ind = self.initial_value
        else:
            ind = super(AgregatedThresholdRecorder, self).index(timestep, scenario_index)
        return ind

    @classmethod
    def load(cls, model, data):
        from pywr.recorders._recorders import load_recorder  # delayed to prevent circular reference
        recorder = load_recorder(model, data.pop("recorder"))
        threshold = load_parameter(model, data.pop("threshold"))
        values = data.pop("values", None)
        predicate = data.pop("predicate", None)
        return cls(model, recorder, threshold, values=values, predicate=predicate, **data)
AgregatedThresholdRecorder.register()








cdef class AgregatedCostThresholdParameter(AbstractThresholdParameter):
    """Returns one of two values depending on a Recorder value and a threshold

    Parameters
    ----------
    recorder : `pywr.recorder.Recorder`

    """

    def __init__(self,  model, Parameter recorder1, Parameter recorder2, *args, initial_value=1, **kwargs):
        super(AgregatedCostThresholdParameter, self).__init__(model, *args, **kwargs)
        self.recorder1 = recorder1
        self.recorder2 = recorder2
        self.recorder1.parents.add(self)
        self.recorder2.parents.add(self)
        self.initial_value = initial_value

    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        # TODO Make this a more general API on Recorder
        return (self.recorder1.get_value(scenario_index)-self.recorder2.get_value(scenario_index))

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns 1 if the predicate evalutes True, else 0"""
        cdef int index = timestep.index
        cdef int ind
        if index == 0:
            # on the first day the recorder doesn't have a value so we have no
            # threshold to compare to
            ind = self.initial_value
        else:
            ind = super(AgregatedCostThresholdParameter, self).index(timestep, scenario_index)
        return ind

    @classmethod
    def load(cls, model, data):
        recorder = data.pop("recorder")
        recorder1 = load_parameter(model, recorder[0])
        recorder2 = load_parameter(model, recorder[1])
        threshold = load_parameter(model, data.pop("threshold"))
        values = data.pop("values", None)
        predicate = data.pop("predicate", None)
        return cls(model, recorder1, recorder2, threshold, values=values, predicate=predicate, **data)
AgregatedCostThresholdParameter.register()
































cdef class CurrentYearThresholdParameter(AbstractThresholdParameter):
    """ Returns one of two values depending on the year of the current timestep..
    """
    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        return float(timestep.year)

    @classmethod
    def load(cls, model, data):
        threshold = load_parameter(model, data.pop("threshold"))
        values = data.pop("values", None)
        predicate = data.pop("predicate", None)
        return cls(model, threshold, values=values, predicate=predicate, **data)
CurrentYearThresholdParameter.register()


cdef class CurrentOrdinalDayThresholdParameter(AbstractThresholdParameter):
    """ Returns one of two values depending on the ordinal of the current timestep.
    """
    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        return float(timestep.datetime.toordinal())

    @classmethod
    def load(cls, model, data):
        threshold = load_parameter(model, data.pop("threshold"))
        values = data.pop("values", None)
        predicate = data.pop("predicate", None)
        return cls(model, threshold, values=values, predicate=predicate, **data)
CurrentOrdinalDayThresholdParameter.register()
