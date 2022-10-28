#include "mcts.h"


std::string TreeNode::to_string() const {
    return "TreeNode: " + move.to_string() + " player: " + CHAR_P[player];
}

double TreeNode::uct() const {
    return winrate() + C * sqrt(log(parent->visit_count) / visit_count);
}


void MCTS_AI::expand_node(TreeNode *node)
{
    expanded_nodes_count++;
    int depth = node->turn_count + 1;
    if (depth > current_max_depth)
    {
        current_max_depth = depth;
    }
    std::set<Point> moves = board.get_AI_moves();
    for (Point const& p : moves)
    {
        auto* child = new TreeNode(p, -node->player, node);
        nodes_vector.push_back(child);
        node->children.push_back(child);
    }
    child_count += node->children.size();
}

int MCTS_AI::mcts(TreeNode *node) {
    if (node->visit_count >= min_visits)    // matured node
    {
        node->visit_count++;
        if (node->children.empty())     // Initialize child nodes if empty
        {
            expand_node(node);
        }
        TreeNode* next = mcts_selection(node);
        board.play(next->move);
        if (board.has_ended())     // If the game end after this move is made
        {
            next->visit_count++;
            int result = board.get_state();
            if (result)     // if result is not 0 (not tie) then move leads to a winning state for player next
            {
                next->win++;
            }
            board.undo();
            return result;
        }
        int result = mcts(next); // recursion call mcts on node next if game doesn't end
        board.undo();
        // propagate result back up
        if (result == node->player)
        {
            node->win++;
        }
        return result;
    }
    else        // not enough maturity
    {
        node->visit_count++;
        int result = simulate();
        // propagate result back up
        if (result == node->player)
        {
            node->win++;
        }
        return result;
    }
}

TreeNode* MCTS_AI::mcts_selection(TreeNode *node)
{
    TreeNode* current = node->children[0];
    for (TreeNode* child : node->children)
    {
        if (current->uct() < child->uct())
        {
            current = child;
        }
    }
    return current;
}

int MCTS_AI::simulate()
{
    Caro temp_board = Caro(board);
    temp_board.simulate();
    return temp_board.get_state();
}

Point MCTS_AI::get_move(Point prev_move)
{
    board.play(prev_move);
    if (current_node == nullptr)    // AI first move, current_node will be nullptr
    {
        current_node = new TreeNode(prev_move, -player, nullptr, board.get_turn_count());
        nodes_vector.push_back(current_node);
        expand_node(current_node);
    }
    else    // update current_node to be its child with prev_move
    {
        for (TreeNode* child : current_node->children)
        {
            if (child->move == prev_move)
            {
                current_node = child;
                break;
            }
        }
    }
    current_depth = current_node->turn_count;

    // MCTS for n_sim iterations
    for (int n = 0; n < n_sim; n++)
    {
        mcts(current_node);
    }
    current_node =  winrate_selection(current_node);
    board.play(current_node->move);
    return current_node->move;
}

// Pick move based on winrate
TreeNode *MCTS_AI::winrate_selection(TreeNode *node) {
    TreeNode* current = node->children[0];
    for (TreeNode* child : node->children)
    {
        if (current->winrate() < child->winrate())
        {
            current = child;
        }
    }
    return current;
}


MCTS_AI::~MCTS_AI()
{
    for (TreeNode* node : nodes_vector)
    {
        delete node;
    }
}
