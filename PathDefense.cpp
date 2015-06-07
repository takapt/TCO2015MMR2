#ifndef LOCAL
#define NDEBUG
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <climits>
#include <cfloat>
#include <ctime>
#include <cassert>
#include <map>
#include <utility>
#include <set>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <sstream>
#include <complex>
#include <stack>
#include <queue>
#include <numeric>
#include <list>
#include <iomanip>
#include <fstream>
#include <bitset>

using namespace std;

#define foreach(it, c) for (__typeof__((c).begin()) it=(c).begin(); it != (c).end(); ++it)
template <typename T> void print_container(ostream& os, const T& c) { const char* _s = " "; if (!c.empty()) { __typeof__(c.begin()) last = --c.end(); foreach (it, c) { os << *it; if (it != last) os << _s; } } }
template <typename T> ostream& operator<<(ostream& os, const vector<T>& c) { print_container(os, c); return os; }
template <typename T> ostream& operator<<(ostream& os, const set<T>& c) { print_container(os, c); return os; }
template <typename T> ostream& operator<<(ostream& os, const multiset<T>& c) { print_container(os, c); return os; }
template <typename T> ostream& operator<<(ostream& os, const deque<T>& c) { print_container(os, c); return os; }
template <typename T, typename U> ostream& operator<<(ostream& os, const map<T, U>& c) { print_container(os, c); return os; }
template <typename T, typename U> ostream& operator<<(ostream& os, const pair<T, U>& p) { os << "(" << p.first << ", " << p.second << ")"; return os; }

template <typename T> void print(T a, int n, const string& split = " ") { for (int i = 0; i < n; i++) { cerr << a[i]; if (i + 1 != n) cerr << split; } cerr << endl; }
template <typename T> void print2d(T a, int w, int h, int width = -1, int br = 0) { for (int i = 0; i < h; ++i) { for (int j = 0; j < w; ++j) { if (width != -1) cerr.width(width); cerr << a[i][j] << ' '; } cerr << endl; } while (br--) cerr << endl; }
template <typename T> void input(T& a, int n) { for (int i = 0; i < n; ++i) cin >> a[i]; }
#define dump(v) (cerr << #v << ": " << v << endl)
// #define dump(v)

#define rep(i, n) for (int i = 0; i < (int)(n); ++i)
#define erep(i, n) for (int i = 0; i <= (int)(n); ++i)
#define all(a) (a).begin(), (a).end()
#define rall(a) (a).rbegin(), (a).rend()
#define clr(a, x) memset(a, x, sizeof(a))
#define sz(a) ((int)(a).size())
#define mp(a, b) make_pair(a, b)
#define ten(n) ((long long)(1e##n))

template <typename T, typename U> void upmin(T& a, const U& b) { a = min<T>(a, b); }
template <typename T, typename U> void upmax(T& a, const U& b) { a = max<T>(a, b); }
template <typename T> void uniq(T& a) { sort(a.begin(), a.end()); a.erase(unique(a.begin(), a.end()), a.end()); }
template <class T> string to_s(const T& a) { ostringstream os; os << a; return os.str(); }
template <class T> T to_T(const string& s) { istringstream is(s); T res; is >> res; return res; }
bool in_rect(int x, int y, int w, int h) { return 0 <= x && x < w && 0 <= y && y < h; }

typedef long long ll;
typedef pair<int, int> pint;
typedef unsigned long long ull;

const int DX[] = { 0, 1, 0, -1 };
const int DY[] = { 1, 0, -1, 0 };


int getms_calls = 0;
#ifdef _MSC_VER
#include <Windows.h>
#else
#include <sys/time.h>
#endif
class Timer
{
    typedef double time_type;
    typedef unsigned int skip_type;

private:
    time_type start_time;
    time_type elapsed;

#ifdef _MSC_VER
    time_type get_ms() { return (time_type)GetTickCount64() / 1000; }
#else
    time_type get_ms() { ++getms_calls; struct timeval t; gettimeofday(&t, NULL); return (time_type)t.tv_sec * 1000 + (time_type)t.tv_usec / 1000; }
//     time_type get_ms() { ++getms_calls; return 0; }
#endif

public:
    Timer() {}

    void start() { start_time = get_ms(); }
    time_type get_elapsed() { return elapsed = get_ms() - start_time; }
};

class Random
{
private:
    unsigned int  x, y, z, w;
public:
    Random(unsigned int x
             , unsigned int y
             , unsigned int z
             , unsigned int w)
        : x(x), y(y), z(z), w(w) { }
    Random() 
        : x(123456789), y(362436069), z(521288629), w(88675123) { }
    Random(unsigned int seed)
        : x(123456789), y(362436069), z(521288629), w(seed) { }

    unsigned int next()
    {
        unsigned int t = x ^ (x << 11);
        x = y;
        y = z;
        z = w;
        return w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
    }

    int next_int() { return next(); }

    // [0, upper)
    int next_int(int upper) { assert(upper > 0); return next() % upper; }

    // [low, high]
    int next_int(int low, int high) { assert(low <= high); return next_int(high - low + 1) + low; }

    double next_double(double upper) { return upper * next() / UINT_MAX; }
    double next_double(double low, double high) { return next_double(high - low) + low; }

    template <typename T>
    int select(const vector<T>& ratio)
    {
        T sum = accumulate(ratio.begin(), ratio.end(), (T)0);
        T v = next_double(sum) + (T)1e-6;
        for (int i = 0; i < (int)ratio.size(); ++i)
        {
            v -= ratio[i];
            if (v <= 0)
                return i;
        }
        return 0;
    }
};
Random g_rand;


#ifdef LOCAL
const double G_TL = 20.0 * 1000.0;
#else
const double G_TL = 2 * 9.6 * 1000.0;
#endif
Timer g_timer;



struct Pos
{
    int x, y;
    Pos(int x, int y)
        : x(x), y(y)
    {
    }
    Pos()
        : x(0), y(0)
    {
    }

    bool operator==(const Pos& other) const
    {
        return x == other.x && y == other.y;
    }
    bool operator !=(const Pos& other) const
    {
        return x != other.x || y != other.y;
    }

    void operator+=(const Pos& other)
    {
        x += other.x;
        y += other.y;
    }
    void operator-=(const Pos& other)
    {
        x -= other.x;
        y -= other.y;
    }

    Pos operator+(const Pos& other) const
    {
        Pos res = *this;
        res += other;
        return res;
    }
    Pos operator-(const Pos& other) const
    {
        Pos res = *this;
        res -= other;
        return res;
    }
    Pos operator*(int a) const
    {
        return Pos(x * a, y * a);
    }

    bool operator<(const Pos& other) const
    {
        if (x != other.x)
            return x < other.x;
        else
            return y < other.y;
    }

    int sq_dist(const Pos& p) const
    {
        int dx = x - p.x;
        int dy = y - p.y;
        return dx * dx + dy * dy;
    }
    double dist(const Pos& p) const
    {
        return sqrt(sq_dist(p));
    }

    bool in_range(const Pos& p, int range) const
    {
        return sq_dist(p) <= range * range;
    }

    Pos next(int dir) const
    {
        return Pos(x + DX[dir], y + DY[dir]);
    }

    void move(int dir)
    {
        x += DX[dir];
        y += DY[dir];
    }
};
Pos operator*(int a, const Pos& pos)
{
    return pos * a;
}
ostream& operator<<(ostream& os, const Pos& pos)
{
    os << "(" << pos.x << ", " << pos.y << ")";
    return os;
}

template <typename T>
class Array2D
{
public:
    Array2D(int w, int h)
        : w_(w), h_(h)
    {
    }

    Array2D(int w, int h, const T& init_val)
        : w_(w), h_(h)
    {
        clear(init_val);
    }

    Array2D()
        : w_(-114514), h_(-1919810)
    {
    }

    int width() const { return w_; }
    int height() const { return h_; }

    T& at(int x, int y)
    {
        assert(in_rect(x, y, width(), height()));
        return a[y][x];
    }
    T& at(const Pos& pos)
    {
        return at(pos.x, pos.y);
    }

    void clear(const T& val)
    {
        rep(y, height()) rep(x, width())
            at(x, y) = val;
    }

private:
    int w_, h_;
    T a[64][64];
};


const int MIN_TOWER_RANGE = 1;
const int MAX_TOWER_RANGE = 5;
const int MIN_TOWER_DAMAGE = 1;
const int MAX_TOWER_DAMAGE = 5;
const int MIN_TOWER_COST = 5;
const int MAX_TOWER_COST = 40;

struct TowerType
{
    int range, damage, cost, id;
};
struct Tower
{
    Pos pos;
    TowerType* type;
    Tower(const Pos& pos, TowerType* type)
        : pos(pos), type(type)
    {
    }

    bool in_range(const Pos& p) const
    {
        return pos.in_range(p, type->range);
    }
};
struct Creep
{
    Pos pos;
    int hp, id;
};
struct Command
{
    Pos pos;
    TowerType type;
};

class Board
{
public:
    Board(){}
    Board(const vector<string>& board)
        : size_(board.size())
    {
        rep(y, size()) rep(x, size())
        {
            if (board[y][x] == '.')
                a[y][x] = 0;
            else if (board[y][x] == '#')
                a[y][x] = 100;
            else if (isdigit(board[y][x]))
                a[y][x] = 200 + board[y][x] - '0';
            else
                abort();
        }
    }

    bool is_path(int x, int y) const
    {
        assert(in(x, y));
        return a[y][x] == 0;
    }
    bool is_path(const Pos& pos) const
    {
        return is_path(pos.x, pos.y);
    }
    bool is_wall(int x, int y) const
    {
        assert(in(x, y));
        return 100 <= a[y][x] && a[y][x] < 200;
    }
    bool is_wall(const Pos& pos) const
    {
        return is_wall(pos.x, pos.y);
    }
    bool is_tower(int x, int y) const
    {
        assert(in(x, y));
        return 101 <= a[y][x] && a[y][x] < 200;
    }
    bool is_tower(const Pos& pos) const
    {
        return is_tower(pos.x, pos.y);
    }
    bool is_base(int x, int y) const
    {
        assert(in(x, y));
        return 200 <= a[y][x] && a[y][x] < 300;
    }
    bool is_base(const Pos& pos) const
    {
        return is_base(pos.x, pos.y);
    }
    bool can_build(int x, int y) const
    {
        assert(in(x, y));
        return a[y][x] == 100;
    }
    bool can_build(const Pos& pos) const
    {
        return can_build(pos.x, pos.y);
    }
    int tower_id(int x, int y) const
    {
        assert(in(x, y));
        assert(is_tower(x, y));
        return a[y][x] - 101;
    }
    bool tower_id(const Pos& pos) const
    {
        return tower_id(pos.x, pos.y);
    }
    int base_id(int x, int y) const
    {
        assert(in(x, y));
        assert(is_base(x, y));
        return a[y][x] - 200;
    }
    int base_id(const Pos& pos) const
    {
        return base_id(pos.x, pos.y);
    }

    void build(int x, int y, int id)
    {
        assert(in(x, y));
        assert(can_build(x, y));
        a[y][x] = 101 + id;
    }

    int size() const { return size_; }

    bool in(int x, int y) const { return 0 <= x && x < size() && 0 <= y && y < size(); }
    bool in(const Pos& pos) const { return in(pos.x, pos.y); }

    vector<Pos> path_in_range(int sx, int sy, int range) const
    {
        assert(in(sx, sy));

        vector<Pos> p;
        for (int y = max(0, sy - range); y <= min(size() - 1, sy + range); ++y)
        {
            for (int x = max(0, sx - range); x <= min(size() - 1, sx + range); ++x)
            {
                if (Pos(sx, sy).in_range(Pos(x, y), range) && is_path(x, y))
                    p.push_back(Pos(x, y));
            }
        }
        return p;
    }

private:
    int size_;
    int a[64][64];
};


int lower_range[2 * 64 * 64]; // (dx * dx + dy * dy) -> lower attackable range
void init_lower_range()
{
    rep(i, 2 * 64 * 64)
        lower_range[i] = 1919810;
    for (int r = MAX_TOWER_RANGE; r >= 0; --r)
        lower_range[r * r] = r;
    for (int r = MAX_TOWER_RANGE * MAX_TOWER_RANGE; r > 0; --r)
        upmin(lower_range[r - 1], lower_range[r]);
}

class PathBuilder
{
public:
    PathBuilder(){}
    PathBuilder(const Board& board)
        : board(board)
    {
        clr(tries, 0);
        clr(_can_move, 0);

        rep(y, board.size()) rep(x, board.size())
        {
            rep(dir, 4)
            {
                int nx = x + DX[dir], ny = y + DY[dir];
                if (board.in(nx, ny) && board.is_base(nx, ny))
                    _can_move[y][x][dir] = true;
            }
        }
    }

    void add_move(const Pos& from, const Pos& to)
    {
        Pos diff = to - from;
        assert(abs(diff.x) + abs(diff.y) == 1);

        int dir = -1;
        rep(d, 4)
            if (Pos(DX[d], DY[d]) == diff)
                dir = d;
        assert(dir != -1);

        _can_move[from.y][from.x][dir] = true;
        ++tries[from.y][from.x];
    }

    bool can_move(const Pos& p, int dir) const
    {
        return _can_move[p.y][p.x][dir];
    }
    bool possible_move(const Pos& p, int dir, int possible_turn) const
    {
        Pos np = p.next(dir);
        return board.in(np) && (board.is_path(np) || board.is_base(np)) && (_can_move[p.y][p.x][dir] || tries[p.y][p.x] < possible_turn);
    }

    vector<Pos> min_cost_path(const Pos& start, const Pos& prev = Pos(-1, -1)) const
    {
        assert(prev.x == -1 || start.sq_dist(prev) == 1);

        int dp[64][64];
        int prev_dir[64][64];
        clr(dp, -1);
        queue<Pos> q;
        q.push(start);
        dp[start.y][start.x] = 0;
        while (!q.empty())
        {
            const Pos cur = q.front();
            q.pop();

            rep(dir, 4)
            {
                const int nx = cur.x + DX[dir], ny = cur.y + DY[dir];
                if (cur == start && Pos(nx, ny) == prev)
                    continue;

                if (board.in(nx, ny) && possible_move(cur, dir, 20) && (board.is_path(nx, ny) || board.is_base(nx, ny)) && dp[ny][nx] == -1)
                {
                    dp[ny][nx] = dp[cur.y][cur.x] + 1;
                    prev_dir[ny][nx] = dir;

                    if (board.is_base(nx, ny))
                    {
                        vector<Pos> path;
                        for (int x = nx, y = ny; x != start.x || y != start.y; )
                        {
                            assert(dp[y][x] != -1);

                            path.push_back(Pos(x, y));

                            int pdir = (prev_dir[y][x] + 2) % 4;
                            x += DX[pdir];
                            y += DY[pdir];
                        }
                        reverse(all(path));
                        return path;
                    }

                    q.push(Pos(nx, ny));
                }
            }
        }

        return min_cost_path_fail_safe(start, prev);
    }

    // this dont use possible_move. try all dirs
    vector<Pos> min_cost_path_fail_safe(const Pos& start, const Pos& prev = Pos(-1, -1)) const
    {
        int dp[64][64];
        int prev_dir[64][64];
        clr(dp, -1);
        queue<Pos> q;
        q.push(start);
        dp[start.y][start.x] = 0;
        while (!q.empty())
        {
            const Pos cur = q.front();
            q.pop();

            rep(dir, 4)
            {
                const int nx = cur.x + DX[dir], ny = cur.y + DY[dir];
                if (cur == start && Pos(nx, ny) == prev)
                    continue;

                if (board.in(nx, ny) && (board.is_path(nx, ny) || board.is_base(nx, ny)) && dp[ny][nx] == -1)
                {
                    dp[ny][nx] = dp[cur.y][cur.x] + 1;
                    prev_dir[ny][nx] = dir;

                    if (board.is_base(nx, ny))
                    {
                        vector<Pos> path;
                        for (int x = nx, y = ny; x != start.x || y != start.y; )
                        {
                            assert(dp[y][x] != -1);

                            path.push_back(Pos(x, y));

                            int pdir = (prev_dir[y][x] + 2) % 4;
                            x += DX[pdir];
                            y += DY[pdir];
                        }
                        reverse(all(path));
                        return path;
                    }

                    q.push(Pos(nx, ny));
                }
            }
        }

        assert(false);
    }

    vector<Pos> random_path(const Pos& start, const Pos& start_prev = Pos(-1, -1)) const
    {
//         return min_cost_path(start, start_prev);
        rep(try_i, 50)
        {
            vector<Pos> path;

            bool retry = false;
            Pos pos = start, prev = start_prev;
            while (!board.is_base(pos))
            {
                vector<int> dirs;
                rep(dir, 4)
                    if (possible_move(pos, dir, 5) && pos.next(dir) != prev)
                        dirs.push_back(dir);
                if (dirs.empty() || path.size() > 5 * board.size())
                {
                    retry = true;
                    break;
                }

                int dir = dirs[g_rand.next_int(dirs.size())];
                prev = pos;
                pos.x += DX[dir];
                pos.y += DY[dir];
                path.push_back(pos);
            }
            if (!retry)
                return path;
        }
//         cerr << "ng" << endl;
        return min_cost_path(start, start_prev);
    }

private:
    Board board;
    int tries[64][64];
    bool _can_move[64][64][4];
};

vector<Pos> spawn_pos;
vector<Pos> list_spawn_pos(const Board& board)
{
    set<Pos> spawn_pos;
    rep(x, board.size())
    {
        if (board.is_path(x, 0))
            spawn_pos.insert(Pos(x, 0));
        if (board.is_path(x, board.size() - 1))
            spawn_pos.insert(Pos(x, board.size() - 1));
    }
    rep(y, board.size())
    {
        if (board.is_path(0, y))
            spawn_pos.insert(Pos(0, y));
        if (board.is_path(board.size() - 1, y))
            spawn_pos.insert(Pos(board.size() - 1, y));
    }

    rep(a, 2) rep(b, 2)
    {
        int sx = a ? 0 : board.size() - 1;
        int sy = b ? 0 : board.size() - 1;
        if (board.is_path(sx, sy))
        {
            rep(dir, 4)
            {
                Pos p = Pos(sx, sy).next(dir);
                while (board.in(p) && board.is_path(p))
                {
                    spawn_pos.erase(p);
                    p.move(dir);
                }
            }
        }
    }

    return vector<Pos>(all(spawn_pos));
}

class World
{
public:
    World(const Board& board, int max_creep_hp, int creep_money, const int current_turn, const int current_money, const vector<Creep>& current_creeps, const vector<Pos>& current_creep_prev, const vector<Tower>& towers, const vector<int>& current_base_hps, const PathBuilder& path_builder, const Array2D<vector<int>>& attack_tower, int appear_creeps)
        : board(board), max_creep_hp(max_creep_hp), creep_money(creep_money), money(current_money), towers(towers), base_hps(current_base_hps), turn(current_turn), attack_tower(attack_tower)
    {
        set<int> remain_ids;
        rep(i, 2040)
            remain_ids.insert(i);
        for (auto& c : current_creeps)
            remain_ids.erase(c.id);

        int num_add_creep;
        if (current_turn < 100)
            num_add_creep = g_rand.next_int(1000 * (2000 - current_turn) / 2000, 2000 * (2000 - current_turn) / 2000);
        else
            num_add_creep = max(0, min(2000 - (int)current_creeps.size(), (int)((double)appear_creeps / current_turn * 2000 - appear_creeps)));

        rep(_, num_add_creep)
        {
            int appear_turn = g_rand.next_int(current_turn, 1999);

            int id = *remain_ids.begin();
            remain_ids.erase(id);
            appear_ids[appear_turn].push_back(id);

            creep_hp[id] = max_creep_hp * (1 << (appear_turn / 500));
            assert(!remain_ids.empty());

            Pos pos = spawn_pos[g_rand.next_int(spawn_pos.size())];
            auto path = current_turn < 50 ? path_builder.min_cost_path(pos) : path_builder.random_path(pos);
            assert(path.size() > 1);
            assert(board.is_base(path.back()));
            creep_paths[id].push_back(pos);
            creep_paths[id].insert(creep_paths[id].end(), all(path));
        }

        assert(current_creeps.size() == current_creep_prev.size());
        rep(i, current_creeps.size())
        {
            auto& c = current_creeps[i];
            alive_ids.insert(c.id);

            Pos prev = current_creep_prev[i];
            auto path = current_turn < 50 ? path_builder.min_cost_path(c.pos, prev) : path_builder.random_path(c.pos, prev);
            creep_paths[c.id] = deque<Pos>(all(path));
            creep_hp[c.id] = c.hp;
            assert(c.hp > 0);
        }
    }

    void step_turn()
    {
        vector<int> dead_ids;

        for (int id : alive_ids)
        {
            assert(0 <= id && id < 2010);
            auto& path = creep_paths[id];
            if (path.size() == 1)
            {
                Pos p = creep_paths[id].front();

                dead_ids.push_back(id);

                assert(board.is_base(p));
                int base_i = board.base_id(p);
                base_hps[base_i] = max(0, base_hps[base_i] - creep_hp[id]);
            }
            assert(!path.empty());
        }
        for (int id : dead_ids)
            alive_ids.erase(id);

        for (int id : appear_ids[turn])
        {
            assert(creep_hp[id] > 0);
            alive_ids.insert(id);
        }

        update_attack();

        for (int id : alive_ids)
        {
            assert(!creep_paths[id].empty());
            creep_paths[id].pop_front();
        }


        ++turn;
    }

    void update_attack()
    {
//         vector<vector<int>> cand(towers.size());
//         for (auto& v : cand)
//             v.clear();
        static vector<int> cand[64 * 64];
        rep(i, towers.size())
            cand[i].clear();

        for (int id : alive_ids)
        {
            assert(creep_hp[id] > 0);
            assert(creep_paths[id].size() > 1);
            for (auto& tower_i : attack_tower.at(creep_paths[id].front()))
                cand[tower_i].push_back(id);
        }

        rep(tower_i, towers.size())
        {
            auto& tower = towers[tower_i];

//             tuple<int, int> target(1919810, -1);
            int target = INT_MAX;
            for (int id : cand[tower_i])
            {
                assert(tower.in_range(creep_paths[id].front()));
                if (creep_hp[id] > 0)
//                     upmin(target, make_tuple(tower.pos.sq_dist(creep_paths[id].front()), id));
                    upmin(target, (tower.pos.sq_dist(creep_paths[id].front()) << 16) | id);
            }

//             int id = get<1>(target);
//             if (id != -1)
            if (target != INT_MAX)
            {
                int id = target & ((1 << 16) - 1);
                creep_hp[id] = max(0, creep_hp[id] - tower.type->damage);
                if (creep_hp[id] == 0)
                {
                    money += creep_money;
                    alive_ids.erase(id);
                }
            }
        }
    }

    void add_tower(const Tower& t)
    {
        money -= t.type->cost;
        for (auto& p : board.path_in_range(t.pos.x, t.pos.y, t.type->range))
            attack_tower.at(p).push_back(towers.size());
        towers.push_back(t);
    }

    void go(int until)
    {
        while (turn < until)
            step_turn();
    }

    int score() const
    {
        return money + accumulate(all(base_hps), 0);
    }

    Board board;
    int max_creep_hp, creep_money;
    vector<Tower> towers;
    Array2D<vector<int>> attack_tower;
    vector<int> base_hps;

    vector<int> appear_ids[2048];
    int creep_hp[2048];
    deque<Pos> creep_paths[2048];

    int turn;
    int money;
    set<int> alive_ids;
};


vector<Pos> predict_path(const Pos& start,  const Board& board, const Pos& prev = Pos(-1, -1))
{
    int dp[64][64];
    int prev_dir[64][64];
    clr(dp, -1);
    queue<Pos> q;
    q.push(start);
    dp[start.y][start.x] = 0;
    if (prev.x != -1)
        dp[prev.y][prev.x] = 0;
    while (!q.empty())
    {
        const Pos cur = q.front();
        q.pop();

        rep(dir, 4)
        {
            const int nx = cur.x + DX[dir], ny = cur.y + DY[dir];
            if (board.in(nx, ny) && (board.is_path(nx, ny) || board.is_base(nx, ny)) && dp[ny][nx] == -1)
            {
                dp[ny][nx] = dp[cur.y][cur.x] + 1;
                prev_dir[ny][nx] = dir;

                if (board.is_base(nx, ny))
                {
                    vector<Pos> path;
                    for (int x = nx, y = ny; x != start.x || y != start.y; )
                    {
                        assert(dp[y][x] != -1);

                        path.push_back(Pos(x, y));

                        int pdir = (prev_dir[y][x] + 2) % 4;
                        x += DX[pdir];
                        y += DY[pdir];
                    }
                    reverse(all(path));
                    return path;
                }

                q.push(Pos(nx, ny));
            }
        }
    }

    assert(false);
}


pair<vector<Creep>, vector<int>> simulate(vector<Creep> creeps, const vector<vector<Pos>>& paths, vector<int> base_hps, const Board& board, const vector<Tower>& towers, Array2D<vector<int>>& attack_tower, const int turns)
{
    assert(creeps.size() == paths.size());

//     vector<vector<int>> cand(towers.size());
    static vector<int> cand[64 * 64];

    int id_to_index[2048];
    rep(i, creeps.size())
        id_to_index[creeps[i].id] = i;

    int dead = 0;
    rep(turn, turns)
    {
        if (dead == creeps.size())
            break;

//         for (auto& v : cand)
//             v.clear();
        rep(i, towers.size())
            cand[i].clear();

        rep(i, creeps.size())
        {
            if (creeps[i].hp > 0)
            {
                if (turn == (int)paths[i].size() - 1)
                {
                    assert(board.is_base(paths[i].back()));
                    int bi = board.base_id(paths[i].back());
                    base_hps[bi] = max(0, base_hps[bi] - creeps[i].hp);

                    ++dead;
                }
                else if (turn < (int)paths[i].size() - 1)
                {
                    for (auto& tower_i : attack_tower.at(paths[i][turn]))
                        cand[tower_i].push_back(i);
                }
            }
        }

        rep(tower_i, towers.size())
        {
            auto& tower = towers[tower_i];

//             tuple<int, int, int> target(1919810, 1919810, -1);
            int target = INT_MAX;
            for (int i : cand[tower_i])
            {
                assert(tower.in_range(paths[i][turn]));
                if (creeps[i].hp > 0)
//                     upmin(target, make_tuple(tower.pos.sq_dist(paths[i][turn]), creeps[i].id, i));
                    upmin(target, (tower.pos.sq_dist(paths[i][turn]) << 16) | creeps[i].id);
            }

//             int i = get<2>(target);
//             if (i != -1)
            if (target != INT_MAX)
            {
                int i = id_to_index[target & ((1 << 16) - 1)];
                assert(0 <= i && i < creeps.size());
                creeps[i].hp = max(0, creeps[i].hp - tower.type->damage);

                if (creeps[i].hp == 0)
                    ++dead;
            }
        }
    }

    return make_pair(creeps, base_hps);
}

class Solver
{
public:
    Solver(){}
    Solver(const vector<string>& board_, int max_creep_hp, int creep_money, const vector<TowerType>& tower_types_)
        : board(Board(board_)), max_creep_hp(max_creep_hp), creep_money(creep_money), tower_types(tower_types_),
        current_turn(-1),
        path_builder(board_)
    {
        vector<double> cost(tower_types.size());
        rep(i, tower_types.size())
            cost[i] = (double)tower_types[i].cost / (tower_types[i].damage * tower_types[i].range);
        const double min_cost = *min_element(all(cost));
        rep(i, tower_types.size())
            if (cost[i] < min_cost * 1.3)
                use_tower_types.push_back(tower_types[i]);

        Pos pos[10];
        int num_base = 0;
        rep(y, board.size()) rep(x, board.size())
        {
            if (board.is_base(x, y))
            {
                ++num_base;
                pos[board.base_id(x, y)] = Pos(x, y);
            }
        }
        base_pos = vector<Pos>(pos, pos + num_base);
    }

    vector<Command> place_towers(const vector<Creep>& creeps, int money, const vector<int>& base_hps)
    {
        ++current_turn;

//         if (g_timer.get_elapsed() > G_TL * 0.93)
//         {
// //             static bool f;
// //             if (!f)
// //             {
// //                 dump(current_turn);
// //                 dump(g_timer.get_elapsed());
// //             }
// //             f = true;
//             return {};
//         }

        vector<Pos> creep_prev_pos(creeps.size(), Pos(-1, -1));
        rep(i, creeps.size())
        {
            auto& c = creeps[i];
            auto& path = actual_paths[c.id];
            if (!path.empty())
            {
                assert(c.pos.sq_dist(path.back()) == 1);
                creep_prev_pos[i] = path.back();
                path_builder.add_move(path.back(), c.pos);
            }
            path.push_back(c.pos);
        }


        const int full_hp = base_hps.size() * 1000;
        const int lost_hp = full_hp - accumulate(all(base_hps), 0);
//         if ((double)lost_hp / full_hp > 0.2)
//         {
//             return {};
//         }

        vector<vector<Pos>> paths(creeps.size());
        rep(i, creeps.size())
        {
            const auto& path = actual_paths[creeps[i].id];
            paths[i] = predict_path(creeps[i].pos, board);
//             if (path.size() <= 1)
//                 paths[i] = path_builder.min_cost_path(creeps[i].pos);
//             else
//                 paths[i] = path_builder.min_cost_path(creeps[i].pos, path[(int)path.size() - 2]);
        }


        const int simulate_turns = min(2 * board.size(), 2000 - current_turn);
        vector<Command> commands;
        for (;;)
        {
            Array2D<vector<int>> attack_tower(board.size(), board.size());
            rep(i, towers.size())
            {
                auto& t = towers[i];
                for (auto& p : board.path_in_range(t.pos.x, t.pos.y, t.type->range))
                    attack_tower.at(p).push_back(i);
            }


            vector<Creep> predict_creeps;
            vector<int> predict_base_hps;
            tie(predict_creeps, predict_base_hps) = simulate(creeps, paths, base_hps, board, towers, attack_tower, simulate_turns);
            int predict_score = accumulate(all(predict_base_hps), 0);
            bool all_kill = true;
            for (auto& c : predict_creeps)
            {
                if (c.hp == 0)
                    predict_score += creep_money * 5;
                else
                    all_kill = false;
            }
            if (all_kill)
                break;

            const int inf = 1919810;
            int attackable_range[64][64];
            rep(y, board.size()) rep(x, board.size()) 
                attackable_range[y][x] = inf;
            rep(ci, creeps.size())
            {
                if (predict_creeps[ci].hp > 0)
                {
                    for (int i = (int)paths[ci].size() - 2; i >= 0; --i)
                    {
                        for (int dy = -MAX_TOWER_RANGE; dy <= MAX_TOWER_RANGE; ++dy)
                        {
                            for (int dx = -MAX_TOWER_RANGE; dx <= MAX_TOWER_RANGE; ++dx)
                            {
                                int x = paths[ci][i].x + dx;
                                int y = paths[ci][i].y + dy;
                                if (board.in(x, y))
                                {
                                    upmin(attackable_range[y][x], lower_range[dx * dx + dy * dy]);
                                }
                            }
                        }
                    }
                }
            }


            double best = 0;
            Command best_command;
            rep(y, board.size()) rep(x, board.size())
            {
                if (board.can_build(x, y))
                {
                    for (auto& tower_type : use_tower_types)
                    {
                        if (money >= tower_type.cost && tower_type.range >= attackable_range[y][x])
                        {
                            vector<Tower> ntowers = towers;
                            ntowers.push_back(Tower(Pos(x, y), &tower_type));

                            for (auto& p : board.path_in_range(x, y, tower_type.range))
                                attack_tower.at(p).push_back((int)ntowers.size() - 1);

                            vector<Creep> npredict_creeps;
                            vector<int> npredict_base_hps;
                            tie(npredict_creeps, npredict_base_hps) = simulate(creeps, paths, base_hps, board, ntowers, attack_tower, simulate_turns);

                            for (auto& p : board.path_in_range(x, y, tower_type.range))
                                attack_tower.at(p).pop_back();


                            int npredict_score = accumulate(all(npredict_base_hps), 0);
                            for (auto& c : npredict_creeps)
                                if (c.hp == 0)
                                    npredict_score += creep_money * 5;

                            double score = double(npredict_score - predict_score) - tower_type.cost / 3 * 3;

                            score *= 10000;
                            score += tower_type.damage * board.path_in_range(x, y, tower_type.range).size() / (double)tower_type.cost;

                            score *= 10000;
                            double nearest_base_dist = 1e9;
                            for (auto& p : base_pos)
                                upmin(nearest_base_dist, p.dist(Pos(x, y)));
                            score += 1000 - nearest_base_dist;

                            if (score > best)
                            {
                                best = score;
                                best_command.pos = Pos(x, y);
                                best_command.type = tower_type;
                            }
                        }
                    }
                }
            }
            if (best < 1e-3)
                break;

            const Tower tower(best_command.pos, &tower_types[best_command.type.id]);
//             bool in_range = false;
//             rep(i, creeps.size())
//             {
//                 rep(j, min((int)paths.size() - 1, 20))
//                 {
//                     if (tower.in_range(paths[i][j]))
//                     {
//                         in_range = true;
//                         break;
//                     }
//                 }
//                 if (in_range)
//                     break;
//             }
//             if (!in_range)
//                 break;

//             if (towers.empty() && current_turn < 100)
//             if (best / 1e8 < tower.type->cost)
            {
                int appear_creeps = 0;
                rep(i, 2010)
                    if (!actual_paths[i].empty())
                        ++appear_creeps;

                int bad = 0;
                rep(_, 2)
                {
                    World ori_world(board, max_creep_hp, creep_money, current_turn, money, creeps, creep_prev_pos, towers, base_hps, path_builder, attack_tower, appear_creeps);
                    World next_world = ori_world;
                    ori_world.go(2000);
//                     if (accumulate(all(ori_world.base_hps), 0) == 0)
                    {
                        next_world.add_tower(tower);
                        next_world.go(2000);

                        if (ori_world.score() >= next_world.score())
                        {
                            //                         dump(current_turn);
                            //                         dump(ori_world.money);
                            //                         dump(ori_world.score());
                            //                         dump(next_world.money);
                            //                         dump(next_world.score());
                            //                         dump(base_hps);
                            //                         dump(ori_world.base_hps);
                            //                         dump(next_world.base_hps);
                            //                         cerr << endl;
                            ++bad;
                        }
                    }
                }
                if (bad >= 2)
                    break;
            }

            towers.push_back(tower);
            board.build(best_command.pos.x, best_command.pos.y, best_command.type.id);
            money -= tower_types[best_command.type.id].cost;
            commands.push_back(best_command);
        }

        return commands;
    }



    int max_creep_hp, creep_money;
    vector<TowerType> tower_types;
    vector<TowerType> use_tower_types;
    vector<Pos> base_pos;

    Board board;
    vector<Tower> towers;

    vector<Pos> actual_paths[2048];

    PathBuilder path_builder;

    int current_turn;
};



class PathDefense
{
public:
    int init(vector <string> board, int money, int creepHealth, int creepMoney, vector <int> towerTypes)
    {
        g_timer.start();

        init_lower_range();
        ::spawn_pos = list_spawn_pos(board);

        vector<TowerType> tower_types(towerTypes.size() / 3);
        rep(i, tower_types.size())
        {
            tower_types[i].range = towerTypes[3 * i];
            tower_types[i].damage = towerTypes[3 * i + 1];
            tower_types[i].cost = towerTypes[3 * i + 2];
            tower_types[i].id = i;
        }

        solver = Solver(board, creepHealth, creepMoney, tower_types);

        return 114514;
    }
    vector <int> placeTowers(vector <int> creep_, int money, vector <int> baseHealth)
    {
        vector<Creep> creeps(creep_.size() / 4);
        rep(i, creeps.size())
        {
            creeps[i].id = creep_[4 * i];
            creeps[i].hp = creep_[4 * i + 1];
            creeps[i].pos.x = creep_[4 * i + 2];
            creeps[i].pos.y = creep_[4 * i + 3];
        }
        sort(all(creeps), [](const Creep& a, const Creep& b){ return a.id < b.id; });

        vector<Command> commands = solver.place_towers(creeps, money, baseHealth);
        vector<int> res;
        for (auto& c : commands)
        {
            res.push_back(c.pos.x);
            res.push_back(c.pos.y);
            res.push_back(c.type.id);
        }

        if (solver.current_turn == 1999)
            dump(g_timer.get_elapsed());

        return res;
    }

private:
    Solver solver;
};


#ifdef LOCAL
int main()
{
    int n, money;
    cin >> n >> money;
    vector<string> board(n);
    input(board, n);
    int creepHealth, creepMoney;
    cin >> creepHealth >> creepMoney;
    int nt;
    cin >> nt;
    vector<int> towerType(nt);
    input(towerType, nt);

#ifdef GEN_INPUT
    ofstream fs("input");
    fs << n << endl;
    fs << money << endl;
    fs << board << endl;
    fs << creepHealth << endl;
    fs << creepMoney << endl;
    fs << nt << endl;
    fs << towerType << endl;
    fs.flush();
#endif

    PathDefense pd;
    pd.init(board, money, creepHealth, creepMoney, towerType);

    rep(t, 2000)
    {
        cin >> money;
        int nc;
        cin >> nc;
        vector<int> creep(nc);
        input(creep, nc);
        int b;
        cin >> b;
        vector<int> baseHealth(b);
        input(baseHealth, b);
        vector<int> ret = pd.placeTowers(creep, money, baseHealth);
        cout << ret.size() << endl;
        for (auto& r : ret)
            cout << r << endl;
        cout.flush();

#ifdef GEN_INPUT
        fs << money << endl;
        fs << nc << endl;
        fs << creep << endl;
        fs << b << endl;
        fs << baseHealth << endl;
        fs.flush();
#endif
    }
}
#endif
