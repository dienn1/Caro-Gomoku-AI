#include "caro.h"


const Point DOWN(1,0);
const Point UP(-1,0);
const Point RIGHT(0,1);
const Point LEFT(0,-1);
const Point DOWN_RIGHT(1,1);
const Point DOWN_LEFT(1,-1);
const Point UP_RIGHT(-1,1);
const Point UP_LEFT(-1,-1);

const int MAX_DIM = 30;
const std::string CHAR[3] = {"O", ".", "X"};
const std::string* CHAR_P = &CHAR[1]; // this is to be able to use negative index e.g. CHAR_P[-1] = "O" and CHAR_P[1] = "X"


Caro::Caro(int _dim): prev_move(Point(-1, -1))
{
    COUNT = 5;
    turn_count = 0;
    dim = std::min(_dim, MAX_DIM);
    size = dim * dim;

    game_state = 0;     // 0:Undecided, 1: X wins, -1: O wins
    game_ended = false;
    turn = 1;           // 1:X  -1:O

    AI_moves.insert(Point(dim/2, dim/2));
}

std::string Caro::to_string() {
    std::string res;
    std::string row_str;

    for (int i = 0; i < dim; i++)
    {
        row_str = "";
        for (int j = 0; j < dim; j++)
        {
            row_str += CHAR_P[board[i][j]] + " ";
        }
        res += row_str + "\n";
    }
    return res;
}

bool Caro::in_bound(Point pos) const
{
    return dim > pos(0) && pos(0) >= 0  && dim > pos(1) && pos(1) >= 0;
}

bool Caro::is_unoccupied(Point pos) const
{
    return in_bound(pos) && board[pos(0)][pos(1)] == 0;
}

bool Caro::play(Point pos)
{
    if (game_ended)
    {
        std::cout << "GAME ALREADY ENDED" << std::endl;
        return false;
    }
    if (!in_bound(pos))
    {
        std::cout << pos.to_string() + " OUT OF BOUND" << std::endl;
        return false;
    }
    if (board[pos(0)][pos(1)] == 0)
    {
        board[pos(0)][pos(1)] = turn;
        prev_move = pos;
        turn_count++;
        check_win();
        generate_AI_moves();
        turn = -turn;
        return true;
    }
    else
    {
        std::cout << pos.to_string() + " IS ALREADY OCCUPIED" << std::endl;
        return false;
    }
}

void Caro::generate_AI_moves(int n)
{
    Point pos = prev_move;
    AI_moves.erase(pos);

    for (int i = pos(0)-n; i <= pos(0)+n; i++)
    {
        for (int j = pos(1)-n; j <= pos(1)+n; j++)
        {
            Point tmp = Point(i,j);
            if (is_unoccupied(tmp))
            {
                AI_moves.insert(tmp);
            }
        }
    }
}

void Caro::check_win()
{
    if (turn_count < COUNT*2-1) { return; }     // Impossible to win at this point
    Point pos = prev_move;
    game_ended = check_diagonal(pos) || check_vertical(pos) || check_horizontal(pos);
    if (game_ended)
    {
        game_state = turn;
        std::cout << CHAR_P[game_state] + " WON IN " + std::to_string(turn_count) + " turns" << std::endl;
    }
    else if (turn_count == size)
    {
        game_ended = true;
        std::cout << "TIE IN " + std::to_string(turn_count) + " turns" << std::endl;
    }
}

std::pair<int, bool> Caro::count_line(Point pos, Point inc)
{
    std::pair res = {0, false};
    while (true)
    {
        Point new_pos = pos + inc;
        if (in_bound(new_pos))
        {
            if (board[pos(0)][pos(1)] == board[new_pos(0)][new_pos(1)])     // If the line is still going
            {
                res.first += 1;
                pos = new_pos;
            }
            else if (board[new_pos(0)][new_pos(1)] == 0)    // the line stops
            {
                return res;
            }
            else    // the line is blocked
            {
                res.second = true;
                return res;
            }
        }
        else    // new_pos out of bound
        {
            return res;
        }
    }
    return res;
}

bool Caro::check_line(Point pos, Point dir1, Point dir2)
{
    std::pair<int, bool> line1 = count_line(pos, dir1);
    std::pair<int, bool> line2 = count_line(pos, dir2);
    int length = 1 + line1.first + line2.first;
    bool blocked = line1.second && line2.second; // True if the line is blocked both directions
    if (length > COUNT || (length == COUNT && !blocked))
    {
        return true;
    }
    return false;
}

bool Caro::check_vertical(Point pos)
{
    return check_line(pos, UP, DOWN);
}

bool Caro::check_horizontal(Point pos)
{
    return check_line(pos, RIGHT, LEFT);
}

bool Caro::check_diagonal(Point pos)
{
    return check_line(pos, DOWN_RIGHT, UP_LEFT) or check_line(pos, UP_RIGHT, DOWN_LEFT);
}


// reference: https://www.reddit.com/r/cpp_questions/comments/r6fqsb/question_using_rand_to_pick_a_random_in_an/hmss00o/
Point Caro::get_random_move()
{
    std::mt19937 rng( std::random_device{}() );
    Point p;
    std::sample( AI_moves.begin(), AI_moves.end(), &p, 1, rng );
    return p;
}

void Caro::simulate(int n_turns)
{
    if (n_turns == -1)  // simulate until the end of the game
    {
        while (!game_ended)
        {
            play(get_random_move());
        }
    }
    else
    {
        for (int i = 0; i < n_turns; i++)
        {
            play(get_random_move());
        }
    }
}



