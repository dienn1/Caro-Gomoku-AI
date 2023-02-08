#ifndef CARO_CPP_MCTS_H
#define CARO_CPP_MCTS_H

#include "caro.h"
#include "tree.h"
#include <cmath>
#include <functional>
#include <cstdlib>
#include <time.h>
#include "constants.h"
#include <string>
#include <random>
using namespace constants;


class MCTS_AI
{
private:
    int min_visits;
    int n_sim;
    bool use_prior;
    unsigned int prior_strength;
    std::function<float(std::array<std::array<int, 30>, 30>, int)> evaluate_prior;
    std::string mode;
    int random_threshold;

    static double uct(TreeNode* node)
    {
        return node->uct();
    }

    int player;  // 1 for X, -1 for O
    TreeNode* current_node;
    Caro board;
    int AI_moves_range;
    int current_depth;
    int current_max_depth;
    std::vector<TreeNode*> nodes_vector;

    size_t child_count;
    size_t expanded_nodes_count;


    float mcts(TreeNode* node);
    TreeNode* mcts_selection(TreeNode* node);
    static TreeNode* reward_selection(TreeNode* node);
    static TreeNode* random_selection(TreeNode* node);
    static TreeNode* visit_selection(TreeNode* node);
    static TreeNode* weighted_visit_selection(TreeNode* node);
    TreeNode* posterior_selection(TreeNode* node);
    int simulate();
    void expand_node(TreeNode* node);
    float posterior_eval(TreeNode* node) const;
    float evaluate_node(TreeNode* node) const;

public:
    MCTS_AI(int _player, int _min_visits, int _n_sim, Caro const& _board, std::string _mode="greedy", int _ai_moves_range = 1, 
        std::function<float(std::array<std::array<int, 30>, 30>, int)> _eval = nullptr, int _prior_strength = 1, int _random_threshold = 8) :
    player(_player), min_visits(_min_visits), n_sim(_n_sim), current_node(nullptr), AI_moves_range(_ai_moves_range), evaluate_prior(_eval), use_prior(false), mode(_mode),
    child_count(0), expanded_nodes_count(0), current_depth(0), current_max_depth(0), random_threshold(_random_threshold)
    {
        board = Caro(_board);
        board.set_AI_moves_range(AI_moves_range);
        board.disable_print();
        if (evaluate_prior != nullptr)
        {
            use_prior = true;
            if (_prior_strength >= 0)
            {
                prior_strength = _prior_strength;
            }
        }
        std::srand(std::time(0));
    }
    ~MCTS_AI();

    [[nodiscard]] Point get_move(Point prev_move);

    [[nodiscard]] int get_tree_depth() const { return current_max_depth - current_depth;}

    [[nodiscard]] float average_child_count() const
    {
        if (expanded_nodes_count == 0){ return 0;}
        return float(child_count)/expanded_nodes_count;
    }

    [[nodiscard]] int get_player() const { return player; }

    [[nodiscard]] float predicted_reward() const
    {
        if (mode == "greedy_post")
        {
            return player * posterior_eval(current_node);
        }
        return player * current_node->average_reward();
    }

    [[nodiscard]] const TreeNode* get_current_node() { return current_node; }
};

#endif //CARO_CPP_MCTS_H
