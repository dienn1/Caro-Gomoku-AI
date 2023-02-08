#include "mcts.h"


void MCTS_AI::expand_node(TreeNode *node)
{
    expanded_nodes_count++;
    int depth = node->turn_count + 1;
    if (depth > current_max_depth)
    {
        current_max_depth = depth;
    }
    std::set<Point> moves = board.get_AI_moves();
    auto board_temp = board.get_board();
    for (Point const& p : moves)
    {
        auto* child = new TreeNode(p, -node->player, node, depth);
        nodes_vector.push_back(child);
        node->children.push_back(child);

        if (use_prior)  // evaluate prior for each node if a prior is given
        {
            board_temp[p(0)][p(1)] = child->player;
            child->prior_eval = evaluate_prior(board_temp, board.get_dim());
            board_temp[p(0)][p(1)] = 0;
        }
    }
    child_count += node->children.size();
}

float MCTS_AI::posterior_eval(TreeNode* node) const
{
    if (prior_strength == 0 && node->visit_count == 0)
    {
        return 0;
    }
    return node->player * (node->prior_eval * prior_strength + node->total_reward) / (prior_strength + node->visit_count);
}

float MCTS_AI::evaluate_node(TreeNode* node) const
{
    if (!use_prior)
    {
        return node->uct();
    }
    return posterior_eval(node) + node->exploration_value(use_prior);
}

float MCTS_AI::mcts(TreeNode *node)
{
    if (node->visit_count >= min_visits)    // matured node
    {
        node->visit_count++;
        if (node->children.empty())     // Initialize child nodes if empty
        {
            expand_node(node);
        }
        TreeNode* next = mcts_selection(node);
        board.play(next->move);
        if (board.has_ended())     // If the game ends, accumulate reward for next and current node, then propagate result back up
        {
            next->visit_count++;
            int result = board.get_state();
            next->total_reward += result;
            node->total_reward += result;
            board.undo();
            return result;
        }
        float result = mcts(next); // recursion call mcts on node next if game doesn't end
        board.undo();
        // propagate result back up
        node->total_reward += result;
        return result;
    }
    else        // not enough maturity
    {
        node->visit_count++;
        float result;
        if (use_prior)
        {
            result = node->prior_eval;
        }
        else
        {
            result = simulate();
        }
        if (node->visit_count >= min_visits)    // if matured
        {
            expand_node(node);
        }
        // propagate result back up
        node->total_reward += result;
        return result;
    }
}

TreeNode* MCTS_AI::mcts_selection(TreeNode *node)
{
    TreeNode* current = node->children[0];
    double current_eval = evaluate_node(current);
    double child_eval = 0;
    for (TreeNode* child : node->children)
    {
        child_eval = evaluate_node(child);
        if (current_eval < child_eval)
        {
            current = child;
            current_eval = child_eval;
        }
    }
    return current;
}

int MCTS_AI::simulate()
{
    int current_turn = board.get_turn_count();
    board.simulate();
    int end_turn = board.get_turn_count();
    int final_state = board.get_state();
    for (int i = 0; i < end_turn - current_turn; i++)
    {
        board.undo();
    }
    return final_state;
}

Point MCTS_AI::get_move(Point prev_move)
{
    if (prev_move != Point(-1, -1))
    {
        board.play(prev_move);
    }
    // AI first move, current_node will be nullptr
    if (current_node == nullptr)
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

    if (current_node->children.empty())     // Initialize child nodes if empty
    {
        expand_node(current_node);
    }

    current_depth = current_node->turn_count;

    // MCTS for n_sim iterations
    for (int n = 0; n < n_sim; n++)
    {
        mcts(current_node);
    }

    if (mode == "random")
    {
        current_node = random_selection(current_node);
    }
    else if (mode == "mcts")
    {
        current_node = mcts_selection(current_node);
    }
    else if (mode == "greedy_post")
    {
        current_node = posterior_selection(current_node);
        std::cout << "PLAY " << current_node->move.to_string() << std::endl;
    }
    else if (mode == "greedy")
    {
        current_node = reward_selection(current_node);
    }
    else if (mode == "greedy_visit")
    {
        current_node = visit_selection(current_node);
    }
    else if (mode == "weighted_visit")
    {
        current_node = weighted_visit_selection(current_node);
    }
    else if (mode == "alpha_zero")
    {
        if (board.get_turn_count() <= random_threshold)
        {
            current_node = weighted_visit_selection(current_node);
        }
        else
        {
            current_node = visit_selection(current_node);
        }
    }
    else
    {
        std::cout << "INVALID MODE" << std::endl;
        exit(69);
    }
    board.play(current_node->move);
    if (current_node->children.empty())     // Initialize child nodes if empty
    {
        expand_node(current_node);
    }
    return current_node->move;
}

// Pick move based on average_reward
TreeNode *MCTS_AI::reward_selection(TreeNode *node) {
    TreeNode* current = node->children[0];      // THIS LINE IS BUGGED FOR SOME REASON AFRER USING GET CURRENT_NODE IN PYTHON -> BECAUSE GARBAGE COLLECTOR OMEGALUL
    for (TreeNode* child : node->children)
    {
        if (current->average_reward() < child->average_reward())
        {
            current = child;
        }
    }
    return current;
}

TreeNode* MCTS_AI::random_selection(TreeNode* node)
{
    int rand_index = std::rand() % node->children.size();
    return node->children[rand_index];
}

TreeNode* MCTS_AI::visit_selection(TreeNode* node)
{
    TreeNode* current = node->children[0];
    for (TreeNode* child : node->children)
    {
        if (current->visit_count < child->visit_count)
        {
            current = child;
        }
    }
    return current;
}

// Select node proportional to its visit count
TreeNode* MCTS_AI::weighted_visit_selection(TreeNode* node)
{
    int total_weights = 0;

    for (int i = 0; i < node->children.size(); i++)
    {
        total_weights += node->children[i]->visit_count;
    }

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> udis(0, total_weights);
    float rnd = udis(rng);

    for (int i = 0; i < node->children.size(); i++)
    {
        if (rnd <= node->children[i]->visit_count)
        {
            return node->children[i];
        }
        rnd -= node->children[i]->visit_count;
    }

    return nullptr;
}

TreeNode* MCTS_AI::posterior_selection(TreeNode* node)
{
    TreeNode* current = node->children[0];
    float current_eval = posterior_eval(current);
    float child_eval = 0;
    for (TreeNode* child : node->children)
    {
        child_eval = posterior_eval(child);
        if (current_eval < child_eval)
        {
            current = child;
            current_eval = child_eval;
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
