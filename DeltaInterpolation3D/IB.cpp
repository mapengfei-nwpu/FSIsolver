#include "IBInterpolation.h"
#include "IBMesh.h"

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
using namespace dolfin;

namespace py = pybind11;
PYBIND11_MODULE(IB, m)
{
    py::class_<DeltaInterpolation>(m, "DeltaInterpolation")
        .def(py::init<std::shared_ptr<IBMesh>, std::shared_ptr<Mesh>, std::shared_ptr<Function>>())
		.def("solid_to_fluid", &DeltaInterpolation::solid_to_fluid)
		.def("fluid_to_solid", &DeltaInterpolation::fluid_to_solid)
		.def("evaluate_current_points", &DeltaInterpolation::evaluate_current_points)
		.def("set_bandwidth", &DeltaInterpolation::set_bandwidth)
		;
    
    py::class_<IBMesh, std::shared_ptr<IBMesh>>(m, "IBMesh")
        .def(py::init<std::array<dolfin::Point, 2>, std::vector<size_t>>())
		.def("mesh",&IBMesh::mesh)
		.def("hash",&IBMesh::hash)
		.def("map",&IBMesh::map)
		.def("get_adjacents",&IBMesh::get_adjacents)
		.def("cell_length",&IBMesh::cell_length)
		;
}