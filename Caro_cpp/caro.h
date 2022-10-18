#ifndef CARO_CPP_CARO_H
#define CARO_CPP_CARO_H

#include <set>
#include <unordered_set>

using namespace std;

class Caro
{
private:
    int COUNT, turn_count, dim, size, game_state, turn;
    int board[30][30];
    int prev_move[2];
    bool game_ended;
    unordered_set<int*> AI_moves;

    void generate_AI_moves(int n=2);

    void check_win();

    int* count_line(int* pos, int* inc);
    bool check_line(int* pos, int* dir1, int* dir2);
    bool check_vertical(int* pos);
    bool check_horizontal(int* pos);
    bool check_diagonal(int* pos);


public:
    bool in_bound(int* pos);
    bool is_unoccupied(int* pos);

    unordered_set<int*> get_moves() const { return AI_moves; }

    int get_state() const { return game_state; }

    bool has_ended() const { return game_ended; }

    bool play(int* pos);

    void simulate(int n=-1);

};

#endif //CARO_CPP_CARO_H
