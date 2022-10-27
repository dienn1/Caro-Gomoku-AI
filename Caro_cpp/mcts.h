#ifndef CARO_CPP_MCTS_H
#define CARO_CPP_MCTS_H

#include "caro.h"
#include <cmath>
#include "constants.h"
using namespace constants;


class TreeNode
{
public:
    Point move;     // The move that leads to this node (the edge)
    TreeNode* parent;   // The parent state the playing move leads to
    std::vector<TreeNode*> children;   // expanded from possible moves from AI_moves
    int visit_count;
    int win;
    int player;     // the player that make the move

    TreeNode(Point _move, int _player, TreeNode* _parent=nullptr):
            move(_move), player(_player), parent(_parent), visit_count(0), win(0) {}

    [[nodiscard]] double winrate() const
    {
        if (visit_count == 0) { return 0;}
        return double(win) / visit_count;
    }

    [[nodiscard]] double uct() const;

    [[nodiscard]] std::string to_string() const;
};


class MCTS_AI
{
private:
    int min_visits;
    int n_sim;
    int player;  // 1 for X, -1 for O
    TreeNode* current_node;
    Caro board;
    int AI_moves_range;
    std::vector<TreeNode*> nodes_vector;

    size_t child_count;
    size_t expanded_nodes_count;

    int mcts(TreeNode* node);
    static TreeNode* mcts_selection(TreeNode* node);
    int simulate();
    void expand_node(TreeNode* node);

public:
    MCTS_AI(int _player, int _min_visits, int _n_sim, Caro const& _board, int _ai_moves_range=1):
    player(_player), min_visits(_min_visits), n_sim(_n_sim), current_node(nullptr), AI_moves_range(_ai_moves_range),
    child_count(0), expanded_nodes_count(0)
    {
        board = Caro(_board);
        board.set_AI_moves_range(AI_moves_range);
        board.disable_print();
    }
    ~MCTS_AI();

    [[nodiscard]] Point get_move(Point prev_move);

    [[nodiscard]] double average_child_count() const
    {
        if (expanded_nodes_count == 0){ return 0;}
        return double(child_count)/double(expanded_nodes_count);
    }

};

#endif //CARO_CPP_MCTS_H
