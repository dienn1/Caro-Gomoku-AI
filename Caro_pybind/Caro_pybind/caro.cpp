#include "caro.h"

const Point DOWN(1,0);
const Point UP(-1,0);
const Point RIGHT(0,1);
const Point LEFT(0,-1);
const Point DOWN_RIGHT(1,1);
const Point DOWN_LEFT(1,-1);
const Point UP_RIGHT(-1,1);
const Point UP_LEFT(-1,-1);


Caro::Caro(int _dim, int _ai_moves_range): prev_move(Point(-1, -1)), AI_moves_range(_ai_moves_range)
{
    print = true;

    COUNT = 5;
    turn_count = 0;
    dim = std::min(_dim, MAX_DIM);
    size = dim * dim;

    game_state = 0;     // 0:Undecided, 1: X wins, -1: O wins
    game_ended = false;
    player = 1;           // 1:X  -1:O

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
    if (board[pos(0)][pos(1)] == 0)     // The move is valid
    {
        // std::cout << pos.to_string() << " PLAYED" << std::endl;
        board[pos(0)][pos(1)] = player;
        prev_move = pos;
        turn_count++;
        check_win();
        generate_AI_moves(AI_moves_range);
        player = -player;
        move_history.push(pos);
        return true;
    }
    else
    {
        std::cout << pos.to_string() + " IS ALREADY OCCUPIED" << std::endl;
        return false;
    }
}

void Caro::undo()
{
    if (move_history.empty())
    {
        return;
    }

    Point pos = move_history.top();
    // std::cout << "UNDOING MOVE " << pos.to_string() << std::endl;

    prev_move = pos;
    board[pos(0)][pos(1)] = 0;
    turn_count--;

    game_ended = false;     // undoing any check_win() action
    game_state = 0;

    std::vector<Point>& moves_added = moves_added_history.top();    // undoing moves added to AI_moves from playing pos
    for (Point const& p : moves_added)
    {
        AI_moves.erase(p);
    }
    moves_added_history.pop();
    AI_moves.insert(pos);   // re-insert pos back to AI_moves

    player = -player;
    move_history.pop();
}

void Caro::generate_AI_moves(int n)
{
    Point pos = prev_move;
    AI_moves.erase(pos);

    std::vector<Point> moves_added;   // vector of moves added this player to be pushed into moves_added_history

    for (int i = pos(0)-n; i <= pos(0)+n; i++)
    {
        for (int j = pos(1)-n; j <= pos(1)+n; j++)
        {
            Point tmp = Point(i,j);
            if (is_unoccupied(tmp))
            {
                std::pair res = AI_moves.insert(tmp);
                if (res.second)     // if tmp is a new move being added
                {
                    moves_added.emplace_back(tmp);    // insert a copy of tmp with emplace
                }
            }
        }
    }
    moves_added_history.push(moves_added);
}

void Caro::check_win()
{
    if (turn_count < COUNT*2-1) { return; }     // Impossible to win at this point
    Point pos = prev_move;
    game_ended = check_diagonal(pos) || check_vertical(pos) || check_horizontal(pos);
    if (game_ended)
    {
        game_state = player;
        if (print)
        {
            std::cout << CHAR_P[game_state] + " WON IN " + std::to_string(turn_count) + " turns" << std::endl;
        }
    }
    else if (turn_count == size)
    {
        game_ended = true;
        if (print)
        {
            std::cout << "TIE IN " + std::to_string(turn_count) + " turns" << std::endl;
        }
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
    return check_line(pos, DOWN_RIGHT, UP_LEFT) || check_line(pos, UP_RIGHT, DOWN_LEFT);
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



