#include "tree.h"


std::string TreeNode::to_string() const {
    return "TreeNode: " + move.to_string() + " player: " + CHAR_P[player];
}

double TreeNode::average_reward() const
{
    if (visit_count == 0) { return 0.0;}
    return player * (double) total_reward / (visit_count);
}

double TreeNode::exploration_value() const
{
    return C * sqrt(log(parent->visit_count) / visit_count);
}

double TreeNode::uct() const {
    return average_reward() + exploration_value();
}

int TreeNode::get_player() const
{
    return player;
}
