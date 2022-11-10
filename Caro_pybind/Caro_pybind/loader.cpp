#include "loader.h"


void save_data_point(std::ofstream& outfile, const TreeNode* node, const Caro& board)
{
    double winrate = node->winrate();
    int player = node->player;
    if (player < 0) { player = 2;}
    outfile << player << "\n";
    auto board_array = board.get_board();
    for (int i = 0; i < board.get_dim(); i++)
    {
        for (int j = 0; j < board.get_dim(); j++)
        {
            int tmp = board_array[i][j];
            if (tmp < 0) { tmp = 2; }
            outfile << tmp << " ";
        }
        outfile << "\n";
    }
    outfile << winrate << "\n\n";
}