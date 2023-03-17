#include "tree.h"


std::string TreeNode::to_string() const 
{
    return "TreeNode: " + move.to_string() + " player: " + CHAR_P[player];
}

void TreeNode::generate_noise()
{
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> normal(0, 1);

    noise = tanh(normal(rng));
}

float TreeNode::average_reward() const
{
    if (visit_count == 0) { return 0.0;}
    return player * total_reward / visit_count;
}

float TreeNode::exploration_value(bool use_prior) const
{
    return C * sqrt(log(parent->visit_count) / (visit_count + int(use_prior)));
}

float TreeNode::uct() const {
    return average_reward() + exploration_value();
}

int TreeNode::get_player() const
{
    return player;
}
