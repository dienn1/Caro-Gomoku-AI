#include <iostream>
#include <unordered_set>
#include <set>
#include <string>
#include <utility>
#include "caro.h"


//struct Point
//{
//    const int X;
//    const int Y;
//    Point(int x, int y): X(x), Y(y) {}
//
//    bool operator==(const Point& p) const
//    {
//        return X == p.X && Y == p.Y;
//    }
//
//    bool operator <(const Point& p) const
//    {
//        return (X < p.X) || (X <= p.X && (Y < p.Y));
//    }
//};
//
//struct PointHashFunction
//{
//    size_t operator()(const Point& p) const
//    {
//        return ((size_t)p.X)<<3 | p.Y;;
//    }
//};

Point get_random_move(const std::set<Point>& moves)
{
    std::mt19937 rng( std::random_device{}() );
    Point p;
    std::sample( moves.begin(), moves.end(), &p, 1, rng );

    return p;
}


int main() {
//    const std::string CHAR[3] = {"O", ".", "X"};
//    const std::string* CHAR_P = &CHAR[1]; // this is to be able to use negative index e.g. CHAR_P[-1] = "O" and CHAR_P[1] = "X"
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

    Caro c = Caro(10);
    Caro c1(c);
    std::cout << c.to_string() << std::endl;
    c.simulate();
    std::cout << c.to_string() << std::endl;
    c1.simulate();
    std::cout << c1.to_string() << std::endl;

//    Point p = Point(1,1) + Point(0, -1);
//    std::cout << (p == Point(0,0)) << std::endl;

    return 0;
}
