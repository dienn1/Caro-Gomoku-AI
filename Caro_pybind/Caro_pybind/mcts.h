#pragma once

#ifndef CARO_CPP_MCTS_H
#define CARO_CPP_MCTS_H

#include "caro.h"
#include "tree.h"
#include <cmath>
#include <functional>
#include <cstdlib>
#include <ctime>
#include "constants.h"
#include <string>
#include <random>
#include "model.h"
#include <Eigen>
#include <algorithm>
using namespace constants;


class MCTS_AI
{
private:
    unsigned int min_visits;
    unsigned int n_sim;
    bool use_prior;
    unsigned int prior_strength;
    float uct_temperature = 0;
    float play_temperature = 1;
    float rollout_weight = 1;
    bool random_transform = false;
    std::function<float(std::array<std::array<int, 30>, 30>, int)> evaluate_prior;
    std::string mode;
    int random_threshold;
    SmallNet* model;

    static double uct(TreeNode* node)
    {
        return node->uct();
    }

    const int BOARD_TRANSFORMS[32][3] = { {3, 0, 0}, {3, 0, 1}, {3, 1, 0}, {3, 1, 1}, {3, 2, 0}, {3, 2, 1}, {3, 3, 0}, {3, 3, 1}, {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {0, 2, 0}, {0, 2, 1}, {0, 3, 0}, {0, 3, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}, {1, 2, 0}, {1, 2, 1}, {1, 3, 0}, {1, 3, 1}, {2, 0, 0}, {2, 0, 1}, {2, 1, 0}, {2, 1, 1}, {2, 2, 0}, {2, 2, 1}, {2, 3, 0}, {2, 3, 1} };
    int player;  // 1 for X, -1 for O
    TreeNode* current_node;
    Caro board;
    unsigned int AI_moves_range;
    unsigned int current_depth;
    unsigned int current_max_depth;
    std::vector<TreeNode*> nodes_vector;

    size_t child_count;
    size_t expanded_nodes_count;


    float mcts(TreeNode* node, bool weighted_select=false);
    TreeNode* mcts_selection(TreeNode* node);
    TreeNode* mcts_selection_temperature(TreeNode* node);
    void evaluate_children_weights(TreeNode* node);
    static TreeNode* reward_selection(TreeNode* node);
    static TreeNode* random_selection(TreeNode* node);
    static TreeNode* visit_selection(TreeNode* node);
    TreeNode* weighted_visit_selection(TreeNode* node);
    TreeNode* weighted_reward_selection(TreeNode* node);
    TreeNode* posterior_selection(TreeNode* node);
    int simulate();
    void expand_node(TreeNode* node);
    float posterior_eval(TreeNode* node) const;
    float evaluate_uct(TreeNode* node) const;

public:
    MCTS_AI(int _player, unsigned int _min_visits, unsigned int _n_sim, Caro const& _board, std::string _mode = "greedy_visit", unsigned int _ai_moves_range = 1,
        std::function<float(std::array<std::array<int, 30>, 30>, int)> _eval = nullptr, unsigned int _prior_strength = 1, unsigned int _random_threshold = 6) :
    player(_player), min_visits(_min_visits), n_sim(_n_sim), current_node(nullptr), AI_moves_range(_ai_moves_range), evaluate_prior(_eval), use_prior(false), mode(_mode),
    child_count(0), expanded_nodes_count(0), current_depth(0), current_max_depth(0), random_threshold(_random_threshold)
    {
        board = Caro(_board);
        board.set_AI_moves_range(AI_moves_range);
        board.disable_print();
        if (evaluate_prior != nullptr)
        {
            use_prior = true;
            if (_prior_strength >= 0)
            {
                prior_strength = _prior_strength;
            }
            rollout_weight = 0;
        }
        model = nullptr;
        std::srand(std::time(0));
    }
    ~MCTS_AI();

    void initialize_model(SmallNet& m_model)
    {
        model = &m_model;
        use_prior = true;
        rollout_weight = 0;
    }

    // TODO: MAKE THIS GENERAL FOR ALL DIM
    float eval_prior_model(std::array<std::array<int, 30>, 30> board_array, int _dim)
    {
        Array77f board_eigen[2];
        // convert board_array
        for (int i = 0; i < 7; i++)
        {
            for (int j = 0; j < 7; j++)
            {
                int pixel = board_array[i][j];
                if (pixel == 1)     // X
                {
                    board_eigen[0](i, j) = 1;
                    board_eigen[1](i, j) = 0;
                }
                else if (pixel == -1)   // O
                {
                    board_eigen[0](i, j) = 0;
                    board_eigen[1](i, j) = 1;
                }
                else    // empty
                {
                    board_eigen[0](i, j) = 0;
                    board_eigen[1](i, j) = 0;
                }
            }
        }

        if (random_transform)
        {
            Array77f temp_board[2];
            srand(time(NULL)); //initialize the random seed
            int trans_i = rand() % 32;
            // Rotation
            for (int i = 0; i < BOARD_TRANSFORMS[trans_i][0]; i++)
            {
                temp_board[0] = board_eigen[0].transpose().colwise().reverse();
                temp_board[1] = board_eigen[1].transpose().colwise().reverse();
                board_eigen[0] = temp_board[0];
                board_eigen[1] = temp_board[1];
            }
            // Mirror
            if (BOARD_TRANSFORMS[trans_i][1] == 1)  // miror col
            {
                temp_board[0] = board_eigen[0].colwise().reverse();
                temp_board[1] = board_eigen[1].colwise().reverse();
                board_eigen[0] = temp_board[0];
                board_eigen[1] = temp_board[1];
            }
            else if (BOARD_TRANSFORMS[trans_i][1] == 2)  // miror row
            {
                temp_board[0] = board_eigen[0].rowwise().reverse();
                temp_board[1] = board_eigen[1].rowwise().reverse();
                board_eigen[0] = temp_board[0];
                board_eigen[1] = temp_board[1];
            }
            else if (BOARD_TRANSFORMS[trans_i][1] == 3)  // miror row and col
            {
                temp_board[0] = board_eigen[0].reverse();
                temp_board[1] = board_eigen[1].reverse();
                board_eigen[0] = temp_board[0];
                board_eigen[1] = temp_board[1];
            }
            // Transpose
            if (BOARD_TRANSFORMS[trans_i][2] == 1)
            {
                temp_board[0] = board_eigen[0].transpose();
                temp_board[1] = board_eigen[1].transpose();
                board_eigen[0] = temp_board[0];
                board_eigen[1] = temp_board[1];
            }
        }
        return model->forward(board_eigen);
    }


    [[nodiscard]] Point get_move(Point prev_move);

    void switch_player() { player = -player; }

    // TODO: Make this generic for all dim
    [[nodiscard]] std::array<int, 49> get_search_distribution();

    [[nodiscard]] float get_uct_temperature() { return uct_temperature; }
    void set_uct_temperature(float t) 
    {
        if (t < 0.1f)
        {
            uct_temperature = 0;
        }
        else
        {
            uct_temperature = t;
        }
    }

    [[nodiscard]] float get_play_temperature() { return play_temperature; }
    void set_play_temperature(float t)
    {
        if (t < 0.1f)
        {
            play_temperature = 0;
        }
        else
        {
            play_temperature = t;
        }
    }

    [[nodiscard]] float get_rollout_weight() { return rollout_weight; }
    void set_rollout_weight(float w)
    {
        if (w < 0.001f)
        {
            rollout_weight = 0;
        }
        else
        {
            rollout_weight = std::min(w, 1.0f);
        }
    }

    [[nodiscard]] unsigned int get_prior_strength() { return prior_strength; }
    void set_prior_strength(unsigned int p)
    {
        prior_strength = std::max(1u, p);
    }

    void enable_random_transform()
    {
        random_transform = true;
    }

    void disable_random_transform()
    {
        random_transform = false;
    }

    void play_move(Point move);

    [[nodiscard]] int get_tree_depth() const { return current_max_depth - current_depth; }

    [[nodiscard]] int get_current_node_children_count() const { return current_node->children.size(); }

    [[nodiscard]] Point get_current_node_child_move(int index) const { return current_node->children[index]->move; }
    [[nodiscard]] float get_current_node_child_average_reward(int index) const 
    { 
        return current_node->children[index]->player * current_node->children[index]->average_reward(); 
    }

    [[nodiscard]] float average_child_count() const
    {
        if (expanded_nodes_count == 0){ return 0;}
        return float(child_count)/expanded_nodes_count;
    }

    [[nodiscard]] int get_player() const { return player; }

    [[nodiscard]] float predicted_reward() const
    {
        if (mode == "greedy_post")
        {
            return player * posterior_eval(current_node);
        }
        return player * current_node->average_reward();
    }

    [[nodiscard]] const TreeNode* get_current_node() { return current_node; }
};

#endif //CARO_CPP_MCTS_H
