#include "caro.h"
#include "mcts.h"


Point get_random_move(const std::set<Point>& moves)
{
    std::mt19937 rng( std::random_device{}() );
    Point p;
    std::sample( moves.begin(), moves.end(), &p, 1, rng );

    return p;
}


int main() {
//    int t[30][30] = {};
//    std::cout << CHAR_P[1] << std::endl;
//

//    Point a = Point(2,4);
//    Point b = Point(69, 96);
//    Point c = Point(2, 6);
//    Point d = Point(42, 2);
//    std::set<Point> s;
//    s.insert(a); s.insert(b); s.insert(c); s.insert(d);
//    std::cout << s.size() << std::endl;
//    s.erase(Point(2, 4));
//    std::cout << s.size() << std::endl;
//    Point p = get_random_move(s);

//    Caro c = Caro(19);
//    Caro c1(c);
//    std::cout << c.to_string() << std::endl;
//    c.simulate();
//    std::cout << c.to_string() << std::endl;
//    c1.simulate();
//    std::cout << c1.to_string() << std::endl;
//    std::stack s = c.get_moves_added_history();
//    size_t n = 0;
//    while (!s.empty())
//    {
//        n += s.top().size();
//        s.pop();
//    }
//    std::cout << n << " MOVES ADDED" << std::endl;
//
//    std::cout << "MOVES HISTORY" << std::endl;
//    std::stack h = c.get_move_history();
//    while (!h.empty())
//    {
//        std::cout << h.top().to_string() << " ";
//        h.pop();
//    }
//    std::cout << std::endl;

//    int n_undo = 100;
//    std::cout << "UNDOING " << n_undo << " TIMES" << std::endl;
//    for (int i = 0; i < n_undo; i++)
//    {
//        c.undo();
//    }
//    std::cout << c.to_string() << std::endl;
//    std::stack s1 = c.get_moves_added_history();
//    size_t n1 = 0;
//    while (!s1.empty())
//    {
//        n1 += s1.top().size();
//        s1.pop();
//    }
//    std::cout << n1 << " MOVES ADDED" << std::endl;
//
//    std::cout << "MOVES HISTORY" << std::endl;
//    std::stack h1 = c.get_move_history();
//    while (!h1.empty())
//    {
//        std::cout << h1.top().to_string() << " ";
//        h1.pop();
//    }
//    std::cout << std::endl;

//    c.simulate();
//    std::cout << c.to_string() << std::endl;


//    Point p = Point(1,1) + Point(0, -1);
//    std::cout << (p == Point(0,0)) << std::endl;

    int n = 20;
    int win = 0;
    for (int i = 0; i < n; i++)
    {
        Caro board = Caro(10);
        board.disable_print();
        int n_sim = 10000;
        int min_visit = 20;
        MCTS_AI mcts_ai = MCTS_AI(1, min_visit, n_sim, &board);
        MCTS_AI mcts_ai2 = MCTS_AI(-1, min_visit, n_sim, &board);
        while (!board.has_ended())
        {
            //std::cout << "MCTS AI THINKING ..." << std::endl;
            board.play(mcts_ai.get_move());
            if (board.has_ended()) { break; }
            board.play(mcts_ai2.get_move());
            //std::cout << board.to_string() << std::endl;
            //std::cin.ignore();
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
            std::cout << "TIE" << std::endl;
        }
        std::cout << board.to_string() << std::endl << std::endl;
    }
    std::cout << float(win)/n << std::endl;

    return 0;
}
