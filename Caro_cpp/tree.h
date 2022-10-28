#ifndef CARO_CPP_TREE_H
#define CARO_CPP_TREE_H

#include "caro.h"
#include <cmath>
#include "constants.h"
using namespace constants;


class TreeNode
{
public:
    int visit_count;
    int win;
    Point move;     // The move that leads to this node (the edge)
    TreeNode* parent;   // The parent state the playing move leads to
    std::vector<TreeNode*> children;   // expanded from possible moves from AI_moves
    const int player;     // the player that make the move
    int turn_count;

    TreeNode(Point _move, int _player, TreeNode* _parent=nullptr, int _turn_count=0):
            move(_move), player(_player), parent(_parent), visit_count(0), win(0)
    {
        if (_turn_count > 0)
        {
            turn_count = _turn_count;
        }
        if (parent == nullptr)
        {
            turn_count = 0;
        }
        else
        {
            turn_count = parent->turn_count + 1;
        }
    }

    [[nodiscard]] double winrate() const
    {
        if (visit_count == 0) { return 0;}
        return double(win) / visit_count;
    }

    [[nodiscard]] double uct() const;

    [[nodiscard]] std::string to_string() const;
};
#endif //CARO_CPP_TREE_H
