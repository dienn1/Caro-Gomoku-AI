#include "tree.h"


std::string TreeNode::to_string() const {
    return "TreeNode: " + move.to_string() + " player: " + CHAR_P[player];
}

double TreeNode::uct() const {
    return winrate() + C * sqrt(log(parent->visit_count) / visit_count);
}
