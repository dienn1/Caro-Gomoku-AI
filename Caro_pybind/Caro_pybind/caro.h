#ifndef CARO_CPP_CARO_H
#define CARO_CPP_CARO_H


#include <set>
#include <unordered_set>
#include <string>
#include <utility>
#include <iostream>
#include <algorithm>
#include <random>
#include <stack>
#include <iterator>
#include <array>
#include "constants.h"
using namespace constants;

struct Point
{
private:
    int X;
    int Y;
public:
    Point(int x, int y): X(x), Y(y) {}
    Point(): X(0), Y(0) {}

    [[nodiscard]] std::string to_string() const
    {
        return "(" + std::to_string(X) + ", " + std::to_string(Y) + ")";
    }

    bool operator==(const Point& p) const
    {
        return X == p.X && Y == p.Y;
    }

    bool operator<(const Point& p) const
    {
        return (X < p.X) || (X <= p.X && (Y < p.Y));
    }

    Point operator+(const Point& other) const
    {
        return {X + other(0), Y + other(1)};
    }

    Point operator-(const Point& other) const
    {
        return {X - other(0), Y - other(1)};
    }

    int operator()(int i) const
    {
        if (i == 0){ return X;}
        else { return Y; }
    }
};

struct PointHashFunction
{
    size_t operator()(const Point& p) const
    {
        return ((size_t)p(0))<<2 | p(1);
    }
};

class Caro
{
public:
    std::array<std::array<int, 30>, 30> board = {}; // Empty board (all zeros), 30x30 is the max size

    explicit Caro(int _dim=19, int count=5, int _ai_moves_range=1);

    std::string to_string();

    [[nodiscard]] bool in_bound(Point pos) const;
    [[nodiscard]] bool is_unoccupied(Point pos) const;

    [[nodiscard]] std::set<Point> get_moves() const { return AI_moves; }

    [[nodiscard]] Point get_prev_move() const {return prev_move; }

    [[nodiscard]] int get_state() const { return game_state; }

    [[nodiscard]] bool has_ended() const { return game_ended; }

    [[nodiscard]] int current_player() const { return player; }

    [[nodiscard]] int get_turn_count() const { return turn_count; }

    [[nodiscard]] int get_dim() const { return dim; }

    [[nodiscard]] auto get_board() const { return board; }

    [[nodiscard]] int get_AI_moves_range() const { return AI_moves_range; }
    void set_AI_moves_range(int moves_range) { AI_moves_range = moves_range; }

    void disable_print() {print = false;}
    void enable_print() {print = true;}

    bool play(Point pos);

    void undo();

    Point get_random_move();

    void simulate(int n_turns=-1);

    [[nodiscard]] std::set<Point> get_AI_moves() const { return AI_moves;}
    [[nodiscard]] std::stack<std::vector<Point>> get_moves_added_history() const { return moves_added_history; }
    [[nodiscard]] std::stack<Point> get_move_history() const { return move_history; }

private:
    int COUNT, turn_count, dim, size, game_state, player, AI_moves_range;
    bool print;
    Point prev_move;
    bool game_ended;
    std::set<Point> AI_moves;
    std::stack<Point> move_history;
    std::stack<std::vector<Point>> moves_added_history;     // history of moves added to AI_moves after each player

    void generate_AI_moves(int n=2);

    void check_win();

    std::pair<int, bool> count_line(Point pos, Point inc);
    bool check_line(Point pos, Point dir1, Point dir2);
    bool check_vertical(Point pos);
    bool check_horizontal(Point pos);
    bool check_diagonal(Point pos);


};

#endif //CARO_CPP_CARO_H
