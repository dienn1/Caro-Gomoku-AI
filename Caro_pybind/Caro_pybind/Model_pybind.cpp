#include <pybind11/pybind11.h>
#include "model.h"
#include "model_test.h"

namespace py = pybind11;

PYBIND11_MODULE(Model_pybind, m)
{
	m.def("SmallNetTest", &py_SmallNetTest);

	py::class_<SmallNet>(m, "SmallNet")
		.def(py::init(&SmallNet::py_Create))
		.def("forward", &SmallNet::py_forward)
		.def("__call__", &SmallNet::py_forward);
}