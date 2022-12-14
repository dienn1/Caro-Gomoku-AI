#include "caro.h"
#include "mcts.h"
#include "loader.h"
#include <chrono>

using namespace std::chrono;
namespace py = pybind11;


Point get_random_move(const std::set<Point>& moves)
{
    std::mt19937 rng( std::random_device{}() );
    Point p;
    std::sample( moves.begin(), moves.end(), &p, 1, rng );

    return p;
}


int main() {
    std::string dir = "training_data/pass1.txt";
    std::ofstream outfile;
    outfile.open(dir);
    outfile << "#PASS1\n\n\n";
    int n = 20;
    int data_count = 0;
    int win = 0;
    int tie = 0;
    int dim = 19;
    int moves_range = 1;
    auto start_all = high_resolution_clock::now();
    for (int i = 0; i < n; i++)
    {
        Caro board = Caro(dim);
        board.disable_print();

        int n_sim1 = 20000;
        int min_visit1 = 10;
        int n_sim2 = 20000;
        int min_visit2 = 10;

        MCTS_AI mcts_ai = MCTS_AI(1, min_visit1, n_sim1, board, moves_range);
        MCTS_AI mcts_ai2 = MCTS_AI(-1, min_visit2, n_sim2, board, moves_range);

        std::cout << "GAME " << i << std::endl;
        while (!board.has_ended())
        {
            auto start = high_resolution_clock::now();
            board.play(mcts_ai.get_move(board.get_prev_move()));
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);
            std::cout << duration.count() << " SECONDS" << std::endl;
            std::cout << "DEPTH: " << mcts_ai.get_tree_depth() << std::endl;
            std::cout << "X PLAYED " << board.get_prev_move().to_string() << " with predicted average_reward " << mcts_ai.predicted_reward() << std::endl;
            std::cout << board.to_string() << std::endl;
            save_data_point(outfile, mcts_ai.get_current_node(), board);
            data_count++;

            if (board.has_ended()) { break; }

            start = high_resolution_clock::now();
            board.play(mcts_ai2.get_move(board.get_prev_move()));
            stop = high_resolution_clock::now();
            duration = duration_cast<seconds>(stop - start);
            std::cout << duration.count() << " SECONDS" << std::endl;
            std::cout << "DEPTH: " << mcts_ai2.get_tree_depth() << std::endl;
            std::cout << "O PLAYED " << board.get_prev_move().to_string() << " with predicted average_reward " << mcts_ai2.predicted_reward() << std::endl;
            std::cout << board.to_string() << std::endl;
            save_data_point(outfile, mcts_ai2.get_current_node(), board);
            data_count++;
        }

        if (board.get_state() == 1)
        {
            win++;
            std::cout << "X WON" << std::endl;
        }
        else if (board.get_state() == -1)
        {
            std::cout << "O WON" << std::endl;
        }
        else
        {
            tie++;
            std::cout << "TIE" << std::endl;
        }
        std::cout << "AI1 has average " << mcts_ai.average_child_count() << " children per expanded node" << std::endl;
        std::cout << "AI2 has average " << mcts_ai2.average_child_count() << " children per expanded node" << std::endl;
        std::cout << board.to_string() << std::endl << std::endl;
        std::cout << "------------------------------------------------------------------------" << std::endl << std::endl;
    }
    auto stop_all = high_resolution_clock::now();
    auto duration_all = duration_cast<seconds>(stop_all - start_all);
    std::cout << duration_all.count() << " SECONDS FOR " << n << " GAMES" << std::endl;
    std::cout << duration_all.count()/n << " SECONDS PER GAME" << std::endl;
    std::cout << "X WON: " << win << std::endl;
    std::cout << "TIE: " << tie << std::endl;

    outfile << data_count << std::endl;
    outfile.close();


    return 0;
}
