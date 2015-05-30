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
    int next_int(int upper) { return next() % upper; }

    // [low, high]
    int next_int(int low, int high) { return next_int(high - low + 1) + low; }

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
const double G_TL = 10.0 * 1000.0;
#else
const double G_TL = 9.6 * 1000.0;
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


vector<Pos> predict_path(const Pos& start,  const Board& board)
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

    vector<vector<int>> cand(towers.size());

    int dead = 0;
    rep(turn, turns)
    {
        if (dead == creeps.size())
            break;

        for (auto& v : cand)
            v.clear();

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

            tuple<int, int, int> target(1919810, 1919810, -1);
            for (int i : cand[tower_i])
            {
                assert(tower.in_range(paths[i][turn]));
                if (creeps[i].hp > 0)
                    upmin(target, make_tuple(tower.pos.sq_dist(paths[i][turn]), creeps[i].id, i));
            }

            int i = get<2>(target);
            if (i != -1)
            {
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
        current_turn(-1)
    {
        vector<double> cost(tower_types.size());
        rep(i, tower_types.size())
            cost[i] = (double)tower_types[i].cost / (tower_types[i].damage * tower_types[i].range);
        const double min_cost = *min_element(all(cost));
        rep(i, tower_types.size())
            if (cost[i] < min_cost * 3)
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

        const int full_hp = base_hps.size() * 1000;
        const int lost_hp = full_hp - accumulate(all(base_hps), 0);
        if ((double)lost_hp / full_hp > 0.3)
        {
            return {};
        }

        vector<vector<Pos>> paths(creeps.size());
        rep(i, creeps.size())
            paths[i] = predict_path(creeps[i].pos, board);


        const int simulate_turns = 2 * board.size();
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
//                     rep(i, (int)paths[ci].size() - 1)
                    for (int i = (int)paths[ci].size() - 2; i >= max(0, (int)paths[ci].size() - 6); --i)
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

                            double score = double(npredict_score - predict_score) - tower_type.cost;

                            score *= 10000;
                            score += board.path_in_range(x, y, tower_type.range).size();

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

            Tower tower(best_command.pos, &tower_types[best_command.type.id]);
            bool in_range = false;
            rep(i, creeps.size())
            {
                if (tower.in_range(paths[i][0]))
                {
                    in_range = true;
                    break;
                }
            }
            if (!in_range)
                break;

            towers.push_back(tower);
            board.build(best_command.pos.x, best_command.pos.y, best_command.type.id);
            money -= tower_types[best_command.type.id].cost;
            commands.push_back(best_command);
        }

        return commands;
    }




private:
    int max_creep_hp, creep_money;
    vector<TowerType> tower_types;
    vector<TowerType> use_tower_types;
    vector<Pos> base_pos;

    Board board;
    vector<Tower> towers;

    int current_turn;
};



class PathDefense
{
public:
    int init(vector <string> board, int money, int creepHealth, int creepMoney, vector <int> towerTypes)
    {
        init_lower_range();

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
    }
}
#endif
