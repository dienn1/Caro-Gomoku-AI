#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "caro.h"

namespace py = pybind11;


PYBIND11_MODULE(Caro_pybind, m)
{
	py::class_<Point>(m, "Point")
		.def(py::init<int, int>())
		.def(py::init<>())
		.def("__str__", &Point::to_string)
		.def(py::self + py::self)
		.def(py::self - py::self)
		.def(py::self == py::self)
		.def(py::self < py::self)
		.def("__call__", &Point::operator())
		.def("__hash__", &Point::hash);

	py::class_<Caro>(m, "Caro")
		.def(py::init<int, int, int>(), py::arg("_dim") = 19, py::arg("count") = 5, py::arg("_ai_moves_range") = 1)
		.def("__copy__", [](const Caro& self)
			{
				return Caro(self);
			})
		.def("__str__", &Caro::to_string)
		.def("simulate", &Caro::simulate, py::arg("n_turns") = -1)
		.def("disable_print", &Caro::disable_print)
		.def("enable_print", &Caro::enable_print)
		.def("get_moves", &Caro::get_moves)
		.def("get_prev_move", &Caro::get_prev_move)
		.def("get_state", &Caro::get_state)
		.def("has_ended", &Caro::has_ended)
		.def("current_player", &Caro::current_player)
		.def("get_turn_count", &Caro::get_turn_count)
		.def("get_dim", &Caro::get_dim)
		.def("get_board", &Caro::get_board)
		.def("get_AI_moves_range", &Caro::get_AI_moves_range)
		.def("set_AI_moves_range", &Caro::set_AI_moves_range)
		.def("play", &Caro::play)
		.def("undo", &Caro::undo)
		.def("get_random_move", &Caro::get_random_move)
		.def("get_AI_moves", &Caro::get_AI_moves)
		.def("get_moves_added_history", &Caro::get_moves_added_history)
		.def("get_move_history", &Caro::get_move_history);
}
