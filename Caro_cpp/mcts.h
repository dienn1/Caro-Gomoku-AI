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
    Caro* board;
    std::vector<TreeNode*> nodes_vector;

    int mcts(TreeNode* node);
    static TreeNode* mcts_selection(TreeNode* node);
    int simulate();

public:
    MCTS_AI(int _player, int _min_visits, int _n_sim, Caro* _board):
            player(_player), min_visits(_min_visits), n_sim(_n_sim), board(_board), current_node(nullptr){}
    ~MCTS_AI();

    [[nodiscard]] Point get_move();

};

#endif //CARO_CPP_MCTS_H
