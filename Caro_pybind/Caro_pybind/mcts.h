#ifndef CARO_CPP_MCTS_H
#define CARO_CPP_MCTS_H

#include "caro.h"
#include "tree.h"
#include <cmath>
#include <functional>
#include "constants.h"
using namespace constants;


class MCTS_AI
{
private:
    int min_visits;
    int n_sim;
    bool use_prior;
    std::function<double(std::array<std::array<int, 30>, 30>, int)> evaluate_prior;

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


    int mcts(TreeNode* node);
    TreeNode* mcts_selection(TreeNode* node);
    static TreeNode* winrate_selection(TreeNode* node);
    int simulate();
    void expand_node(TreeNode* node);
    double evaluate_node(TreeNode* node);

public:
    MCTS_AI(int _player, int _min_visits, int _n_sim, Caro const& _board, int _ai_moves_range = 1, std::function<double(std::array<std::array<int, 30>, 30>, int)> _eval = nullptr) :
    player(_player), min_visits(_min_visits), n_sim(_n_sim), current_node(nullptr), AI_moves_range(_ai_moves_range), evaluate_prior(_eval), use_prior(false),
    child_count(0), expanded_nodes_count(0), current_depth(0), current_max_depth(0)
    {
        board = Caro(_board);
        board.set_AI_moves_range(AI_moves_range);
        board.disable_print();
        if (evaluate_prior != nullptr)
        {
            use_prior = true;
        }
    }
    ~MCTS_AI();

    [[nodiscard]] Point get_move(Point prev_move);

    [[nodiscard]] int get_tree_depth() const { return current_max_depth - current_depth;}

    [[nodiscard]] double average_child_count() const
    {
        if (expanded_nodes_count == 0){ return 0;}
        return double(child_count)/double(expanded_nodes_count);
    }

    [[nodiscard]] int get_player() const { return player; }

    [[nodiscard]] double predicted_winrate() const
    {
        return current_node->winrate();
    }

    [[nodiscard]] const TreeNode* get_current_node() { return current_node; }
};

#endif //CARO_CPP_MCTS_H
