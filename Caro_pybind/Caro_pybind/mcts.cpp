#include "mcts.h"


void MCTS_AI::expand_node(TreeNode *node)
{
    expanded_nodes_count++;
    unsigned int depth = node->turn_count + 1;
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

        if (use_prior)  // evaluate prior for each node if a prior evaluator is given
        {
            board_temp[p(0)][p(1)] = child->player;
            float prior;
            if (model != nullptr)
            {
                prior = eval_prior_model(board_temp, board.get_dim());
            }
            else if (evaluate_prior != nullptr)
            {
                prior = evaluate_prior(board_temp, board.get_dim());
            }
            else
            {
                std::cout << "NO MODEL TO EVALUATE PRIOR\n";
            }
            child->prior_eval = prior;
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

float MCTS_AI::evaluate_uct(TreeNode* node) const
{
    if (!use_prior)
    {
        return node->uct();
    }
    return ((posterior_eval(node) + 1) * 0.5 + 1) + node->exploration_value(use_prior);     // shift reward to [1, 2] to make sure UCT > 0
}

float MCTS_AI::mcts(TreeNode* node, bool weighted_select)
{
    if (node->visit_count >= min_visits)    // matured node
    {
        node->visit_count++;
        if (node->children.empty())     // Initialize child nodes if empty
        {
            expand_node(node);
        }
        TreeNode* next;
        if (weighted_select)
        {
            next = mcts_selection_temperature(node);
        }
        else
        {
            next = mcts_selection(node);
        }
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
        float result = mcts(next); // recursion call mcts on node next if game doesn't end (no weighted selection)
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
            if (rollout_weight < 0.001f)
            {
                result = node->prior_eval;
            }
            else
            {
                result = rollout_weight * simulate() + (1 - rollout_weight) * node->prior_eval;
            }
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
    double current_eval = evaluate_uct(current);
    double child_eval = 0;
    for (TreeNode* child : node->children)
    {
        child_eval = evaluate_uct(child);
        if (current_eval < child_eval)
        {
            current = child;
            current_eval = child_eval;
        }
    }
    return current;
}

void MCTS_AI::evaluate_children_weights(TreeNode* node)
{
    if (uct_temperature < 0.001)
    {
        return;
    }

    float exponent = 1 / uct_temperature;

    for (TreeNode* child : node->children)
    {
        child->weight = pow(evaluate_uct(child), exponent);
    }
}

// MCTS select according to weighted UCT score
TreeNode* MCTS_AI::mcts_selection_temperature(TreeNode* node)
{
    if (uct_temperature < 0.01)
    {
        return mcts_selection(node);
    }

    evaluate_children_weights(node);
    float total_weights = 0;
    for (int i = 0; i < node->children.size(); i++)
    {
        total_weights += node->children[i]->weight;
    }

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> udis(0, total_weights);
    float rnd = udis(rng);

    int temp = 0;
    for (int i = 0; i < node->children.size(); i++)
    {
        temp = i;
        if (rnd <= node->children[i]->weight)
        {
            return node->children[i];
        }
        rnd -= node->children[i]->weight;
    }

    std::cout << "RANDOM SELECTION DID NOT PICK ANY\n";
    return  node->children[temp];
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

// Return the move the AI think is best according to mode
Point MCTS_AI::get_move(Point prev_move)
{
    if (prev_move != Point(-1, -1))
    {
        if (current_node == nullptr || !(prev_move == current_node->move && player != current_node->player))    // in the case of not playing against itself
        {
            bool valid = board.play(prev_move);
            if (!valid)
            {
                std::cout << "INVALID MOVE " << prev_move.to_string() << "\n";
                return Point(-1, -1);
            }
        }
    }
    // AI first move, current_node will be nullptr, current_node is the previous move of the other player
    if (current_node == nullptr)
    {
        current_node = new TreeNode(prev_move, -player, nullptr, board.get_turn_count());
        nodes_vector.push_back(current_node);
        expand_node(current_node);
    }
    else    // update current_node to be its child with prev_move
    {
        bool found_move = false;
        for (TreeNode* child : current_node->children)
        {
            if (child->move == prev_move)
            {
                current_node = child;
                found_move = true;
                break;
            }
        }
        if (!found_move)    // There is no child node with prev_move
        {
            current_node = new TreeNode(prev_move, -player, current_node, board.get_turn_count());
            nodes_vector.push_back(current_node);
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
        mcts(current_node, true);   // weighted selection at root_node only
    }

    TreeNode* next_move;

    if (mode == "random")
    {
        next_move = random_selection(current_node);
    }
    else if (mode == "mcts")
    {
        next_move = mcts_selection(current_node);
    }
    else if (mode == "greedy_post")
    {
        next_move = posterior_selection(current_node);
    }
    else if (mode == "greedy")
    {
        next_move = reward_selection(current_node);
    }
    else if (mode == "greedy_visit")
    {
        next_move = visit_selection(current_node);
    }
    else if (mode == "weighted_visit")
    {
        next_move = weighted_visit_selection(current_node);
    }
    else if (mode == "weighted_reward")
    {
        next_move = weighted_reward_selection(current_node);
    }
    else if (mode == "alpha_zero")
    {
        if (board.get_turn_count() <= random_threshold)
        {
            next_move = weighted_visit_selection(current_node);
        }
        else
        {
            next_move = visit_selection(current_node);
        }
    }
    else if (mode == "alpha_zero_reward")
    {
        if (board.get_turn_count() <= random_threshold)
        {
            next_move = weighted_reward_selection(current_node);
        }
        else
        {
            next_move = reward_selection(current_node);
        }
    }
    else
    {
        std::cout << "INVALID MODE" << std::endl;
        exit(69);
    }

    return next_move->move;
}

// Tell the AI to play a move on its board
void MCTS_AI::play_move(Point move)
{
    bool valid = board.play(move);
    if (!valid)
    {
        std::cout << "INVALID MOVE " << move.to_string() << "\n";
        return;
    }

    bool found_move = false;
    for (TreeNode* child : current_node->children)
    {
        if (child->move == move)
        {
            current_node = child;
            found_move = true;
            break;
        }
    }

    if (!found_move)    // There is no child node with move
    {
        current_node = new TreeNode(move, player, current_node, board.get_turn_count());
        nodes_vector.push_back(current_node);
    }

    if (current_node->children.empty())     // Initialize child nodes if empty
    {
        expand_node(current_node);
    }

}

// FOR DIM=7 only
std::array<int, 49> MCTS_AI::get_search_distribution()
{
    std::array<int, 49> search_vector{};
    for (TreeNode* child : current_node->children)
    {
        search_vector[child->move(0) * 7 + child->move(1)] = child->visit_count;
    }
    return search_vector;
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
    if (play_temperature < 0.001)
    {
        return visit_selection(node);
    }

    float total_weights = 0;
    std::vector<float> weight_vector;
    float exponent = 1 / play_temperature;
    for (int i = 0; i < node->children.size(); i++)
    {
        float w = pow(node->children[i]->visit_count, exponent);
        total_weights += w;
        weight_vector.push_back(w);
    }

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> udis(0, total_weights);
    float rnd = udis(rng);
    int temp = 0;
    for (int i = 0; i < node->children.size(); i++)
    {
        temp = i;
        if (rnd <= weight_vector[i])
        {
            return node->children[i];
        }
        rnd -= weight_vector[i];
    }

    std::cout << "RANDOM SELECTION DID NOT PICK ANY\n";
    return  node->children[temp];
}

TreeNode* MCTS_AI::weighted_reward_selection(TreeNode* node)
{
    if (play_temperature < 0.001)
    {
        return reward_selection(node);
    }

    float total_weights = 0;
    std::vector<float> weight_vector;
    float exponent = 1 / play_temperature;
    for (int i = 0; i < node->children.size(); i++)
    {
        float shifted_reward = (node->children[i]->average_reward() + 1) * 0.5 + 1;     // shift reward to be in [1, 2]
        float w = pow(shifted_reward, exponent);
        total_weights += w;
        weight_vector.push_back(w);
    }

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> udis(0, total_weights);
    float rnd = udis(rng);
    int temp = 0;
    for (int i = 0; i < node->children.size(); i++)
    {
        temp = i;
        if (rnd <= weight_vector[i])
        {
            return node->children[i];
        }
        rnd -= weight_vector[i];
    }

    std::cout << "RANDOM SELECTION DID NOT PICK ANY\n";
    return  node->children[temp];
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
