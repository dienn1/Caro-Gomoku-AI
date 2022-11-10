#include "tree.h"


std::string TreeNode::to_string() const {
    return "TreeNode: " + move.to_string() + " player: " + CHAR_P[player];
}

double TreeNode::winrate() const
{
    if (visit_count == 0) { return 0.0;}
    return (double) win / (visit_count);
}

double TreeNode::exploration_value() const
{
    return C * sqrt(log(parent->visit_count) / visit_count);
}

double TreeNode::uct() const {
    return winrate() + exploration_value();
}

int TreeNode::get_player() const
{
    return player;
}
