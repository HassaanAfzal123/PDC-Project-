#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <string>
#include <sstream>
#include <unordered_map>
#include <chrono>
using namespace std;
using ll = long long;
const ll INF = (ll)1e18;

static unordered_map<int, unordered_map<int, int>> adj;

bool read_metis(const string& file) {
    ifstream in(file);
    if (!in.is_open()) {
        cerr << "Error: Could not open " << file << "\n";
        return false;
    }
    int n, m, fmt;
    if (!(in >> n >> m >> fmt)) {
        cerr << "Error: Invalid METIS header\n";
        return false;
    }
    string line;
    getline(in, line); 

    for (int u = 1; u <= n; u++) {
        if (!getline(in, line)) {
            cerr << "Error: Incomplete METIS graph data\n";
            return false;
        }
        istringstream iss(line);
        int v, w;
        while (iss >> v) {
            w = 1; 
            if (fmt == 1) {
                if (!(iss >> w)) {
                    cerr << "Error: Missing weight for edge\n";
                    return false;
                }
            }
            adj[u][v] = w;
            adj[v][u] = w; 
        }
    }
    return true;
}

unordered_map<int, ll> dijkstra(int src) {
    unordered_map<int, ll> dist;
    dist[src] = 0;
    priority_queue<pair<ll, int>, vector<pair<ll, int>>, greater<>> pq;
    pq.push({0, src});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto& [v, w] : adj[u]) {
            ll nd = d + w;
            if (!dist.count(v) || nd < dist[v]) {
                dist[v] = nd;
                pq.push({nd, v});
            }
        }
    }
    return dist;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " graph.txt updates.txt\n";
        return 1;
    }
    string graphFile = argv[1], updatesFile = argv[2];

    if (!read_metis(graphFile)) {
        return 1;
    }

    struct Up { char op; int u, v, w; };
    vector<Up> updates;
    {
        ifstream in(updatesFile);
        if (!in.is_open()) {
            cerr << "Error: Could not open " << updatesFile << "\n";
            return 1;
        }
        char op;
        int u, v, w;
        string line;
        while (getline(in, line)) {
            istringstream iss(line);
            if (!(iss >> op >> u >> v)) {
                cerr << "Error: Invalid update format\n";
                return 1;
            }
            w = 0; 
            if (op == 'I' && !(iss >> w)) {
                cerr << "Error: Missing weight for insertion\n";
                return 1;
            }
            updates.push_back({op, u, v, w});
        }
    }

    auto t0 = chrono::high_resolution_clock::now();
    for (const auto& u : updates) {
        if (u.op == 'I') {
        
            adj[u.u][u.v] = u.w;
            adj[u.v][u.u] = u.w; 
        } else if (u.op == 'D') {
            adj[u.u].erase(u.v);
            adj[u.v].erase(u.u);
        } else {
            cerr << "Error: Unknown operation " << u.op << "\n";
            continue;
        }
        auto dist = dijkstra(1);
    }
    auto t1 = chrono::high_resolution_clock::now();
    auto ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

    cout << "Total time (ms): " << ms << "\n";
    return 0;
}