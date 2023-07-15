#ifndef CARO_CPP_TREE_H
#define CARO_CPP_TREE_H

#include "caro.h"
#include <cmath>
#include "constants.h"
using namespace constants;


class TreeNode
{
public:
    unsigned int visit_count;
    float total_reward;
    Point move;     // The move that leads to this node (the edge)
    TreeNode* parent;   // The parent state the playing move leads to
    std::vector<TreeNode*> children;   // expanded from possible moves from AI_moves
    const int player;     // the player that make the move
    unsigned int turn_count;

    float prior_eval;
    float noise;
    float weight;

    TreeNode(Point _move, int _player, TreeNode* _parent=nullptr, unsigned int _turn_count=0):
            move(_move), player(_player), parent(_parent), visit_count(0), total_reward(0), prior_eval(0), noise(0), weight(1)
    {
        if (_turn_count > 0)
        {
            turn_count = _turn_count;
        }
        else if (parent == nullptr)
        {
            turn_count = 0;
        }
        else
        {
            turn_count = parent->turn_count + 1;
        }
    }

    [[nodiscard]] float average_reward() const;

    [[nodiscard]] float exploration_value(bool use_prior=false) const;

    [[nodiscard]] float uct() const;

    [[nodiscard]] int get_player() const;

    [[nodiscard]] std::string to_string() const;

    void generate_noise();
};
#endif //CARO_CPP_TREE_H
