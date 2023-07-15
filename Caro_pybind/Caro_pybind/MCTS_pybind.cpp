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
		.def(py::init<Point, int, TreeNode*, unsigned int>(), py::arg("_move"), py::arg("_player"), py::arg("_parent") = nullptr, py::arg("_turn_count") = 0)
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
		.def(py::init<int, unsigned int, unsigned int, Caro const&, std::string, unsigned int, 
			std::function<float(std::array<std::array<int, 30>, 30>, int)>, unsigned int, unsigned int>(),
			py::arg("_player"), py::arg("_min_visits"), py::arg("_n_sim"), py::arg("_board"), 
			py::arg("_mode") = "greedy_visit", py::arg("_ai_moves_range") = 1, py::arg("_eval") = nullptr, py::arg("_prior_strength") = 1, py::arg("_random_threshold") = 6)
		.def("initialize_model", &MCTS_AI::initialize_model)
		.def("get_move", &MCTS_AI::get_move)
		.def("play_move", &MCTS_AI::play_move)
		.def("switch_player", &MCTS_AI::switch_player)
		.def("get_uct_temperature", &MCTS_AI::get_uct_temperature)
		.def("set_uct_temperature", &MCTS_AI::set_uct_temperature)
		.def("get_play_temperature", &MCTS_AI::get_play_temperature)
		.def("set_play_temperature", &MCTS_AI::set_play_temperature)
		.def("get_rollout_weight", &MCTS_AI::get_rollout_weight)
		.def("set_rollout_weight", &MCTS_AI::set_rollout_weight)
		.def("get_prior_strength", &MCTS_AI::get_prior_strength)
		.def("set_prior_strength", &MCTS_AI::set_prior_strength)
		.def("enable_random_transform", &MCTS_AI::enable_random_transform)
		.def("disable_random_transform", &MCTS_AI::disable_random_transform)
		.def("get_search_distribution", &MCTS_AI::get_search_distribution)
		.def("get_tree_depth", &MCTS_AI::get_tree_depth)
		.def("get_current_node_children_count", &MCTS_AI::get_current_node_children_count)
		.def("get_current_node_child_move", &MCTS_AI::get_current_node_child_move)
		.def("get_current_node_child_average_reward", &MCTS_AI::get_current_node_child_average_reward)
		.def("average_child_count", &MCTS_AI::average_child_count)
		.def("get_player", &MCTS_AI::get_player)
		.def("predicted_reward", &MCTS_AI::predicted_reward)
		.def("get_current_node", &MCTS_AI::get_current_node);
}