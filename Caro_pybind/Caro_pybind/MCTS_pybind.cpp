#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "mcts.h"
#include "tree.h"
#include "caro.h"
namespace py = pybind11;


PYBIND11_MODULE(MCTS_pybind, m)
{
	py::class_<TreeNode>(m, "TreeNode")
		.def(py::init<Point, int, TreeNode*, int>(), py::arg("_move"), py::arg("_player"), py::arg("_parent") = nullptr, py::arg("_turn_count") = 0)
		.def_readwrite("visit_count", &TreeNode::visit_count)
		.def_readwrite("total_reward", &TreeNode::total_reward)
		.def_readwrite("move", &TreeNode::move)
		.def_readwrite("parent", &TreeNode::parent)
		.def_readwrite("children", &TreeNode::children)
		.def_readonly("player", &TreeNode::player)
		.def_readwrite("turn_count", &TreeNode::turn_count)
		.def("average_reward", &TreeNode::average_reward)
		.def("uct", &TreeNode::uct)
		.def("get_player", &TreeNode::get_player)
		.def("__str__", &TreeNode::to_string);

	py::class_<MCTS_AI>(m, "MCTS_AI")
		.def(py::init<int, int, int, Caro const&, int, std::function<double(std::array<std::array<int, 30>, 30>)>>(),
			py::arg("_player"), py::arg("_min_visits"), py::arg("_n_sim"), py::arg("_board"), py::arg("_ai_moves_range") = 1, py::arg("_eval") = nullptr)
		.def("get_move", &MCTS_AI::get_move)
		.def("get_tree_depth", &MCTS_AI::get_tree_depth)
		.def("average_child_count", &MCTS_AI::average_child_count)
		.def("get_player", &MCTS_AI::get_player)
		.def("predicted_reward", &MCTS_AI::predicted_reward)
		.def("get_current_node", &MCTS_AI::get_current_node);
}