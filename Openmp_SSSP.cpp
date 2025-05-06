#include <bits/stdc++.h>
#include <omp.h>
#include <chrono>
using namespace std;
using pii = pair<int,int>;
const long long INF = LLONG_MAX/4;

//----------------------------------------------------------------
// GLOBALS
//----------------------------------------------------------------
int n;                                  // number of vertices (1-based)
vector<vector<pii>> adj;                // adjacency list: (neighbor, weight)
vector<long long> distv;                // current distance from source
vector<int> parentv;                    // parent in SSSP tree
vector<char> affectedDel, affected;     // flags for deletion & any change

//----------------------------------------------------------------
// UTILITIES
//----------------------------------------------------------------
bool edge_exists(int u, int v){
    for(auto &pr : adj[u])
        if(pr.first==v) return true;
    return false;
}

void log_time(const char* phase,
              chrono::high_resolution_clock::time_point start){
    auto end = chrono::high_resolution_clock::now();
    auto ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    #pragma omp critical
    cout<<"[TIME] "<<phase<<" took "<<ms<<" ms\n";
}

// Initialize SSSP from scratch (Dijkstra)
void init_sssp(int src){
    auto t0 = chrono::high_resolution_clock::now();
    distv.assign(n+1, INF);
    parentv.assign(n+1, 0);
    distv[src] = 0;
    using T = pair<long long,int>;
    priority_queue<T,vector<T>,greater<T>> pq;
    pq.push({0,src});
    while(!pq.empty()){
        auto [d,u] = pq.top(); pq.pop();
        if(d!=distv[u]) continue;
        for(auto [v,w]: adj[u]){
            if(distv[v] > d + w){
                distv[v] = d + w;
                parentv[v] = u;
                pq.push({distv[v],v});
            }
        }
    }
    log_time("Dijkstra init_sssp", t0);
}

//----------------------------------------------------------------
// PHASE I: Process deletions & insertions
//----------------------------------------------------------------
void process_changes(const vector<pair<int,int>>& dels,
                     const vector<tuple<int,int,int>>& ins){
    fill(affectedDel.begin(), affectedDel.end(), 0);
    fill(affected.begin(),    affected.end(),    0);

    auto t0 = chrono::high_resolution_clock::now();
    // 1) Deletions
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<(int)dels.size(); i++){
        auto [u,v] = dels[i];
        bool inTree = (parentv[v]==u)||(parentv[u]==v);
        if(!inTree) continue;
        int child = (parentv[v]==u ? v : u);
        affectedDel[child] = 1;
        affected   [child] = 1;
        distv[child] = INF;
        parentv[child] = 0;
        #pragma omp critical
        cout<<"[DEL] Edge "<<u<<"-"<<v<<" cut; vertex "<<child<<" disconnected\n";
    }

    // 2) Insertions
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<(int)ins.size(); i++){
        auto [u,v,w] = ins[i];
        // try u->v
        if(distv[u] + w < distv[v]){
            #pragma omp critical
            cout<<"[INS] "<<u<<"->"<<v<<" improves "<<distv[v]<<"->"<<(distv[u]+w)<<"\n";
            distv[v] = distv[u] + w;
            parentv[v] = u;
            affected[v] = 1;
        }
        // try v->u
        if(distv[v] + w < distv[u]){
            #pragma omp critical
            cout<<"[INS] "<<v<<"->"<<u<<" improves "<<distv[u]<<"->"<<(distv[v]+w)<<"\n";
            distv[u] = distv[v] + w;
            parentv[u] = v;
            affected[u] = 1;
        }
        // add edge if not present
        #pragma omp critical
        {
            if(!edge_exists(u,v)){
                adj[u].emplace_back(v,w);
                adj[v].emplace_back(u,w);
            }
        }
    }
    log_time("Phase I process_changes", t0);
}

//----------------------------------------------------------------
// PHASE II: Update affected subgraph
//----------------------------------------------------------------
void update_affected(){
    auto t0 = chrono::high_resolution_clock::now();
    // A: propagate deletions down tree
    queue<int> Q;
    for(int i=1;i<=n;i++)
        if(affectedDel[i]) Q.push(i);

    while(!Q.empty()){
        int u=Q.front(); Q.pop();
        for(auto [v,w]: adj[u]){
            if(parentv[v]==u && !affectedDel[v]){
                affectedDel[v]=1;
                affected[v]=1;
                distv[v]=INF;
                parentv[v]=0;
                Q.push(v);
                #pragma omp critical
                cout<<"[PROP-DEL] vertex "<<v<<" disconnected\n";
            }
        }
    }

    // B: iterative relax until no change
    bool any=true;
    int iter=0;
    while(any){
        iter++;
        any=false;
        #pragma omp parallel for schedule(dynamic) reduction(|:any)
        for(int u=1;u<=n;u++){
            if(!affected[u]) continue;
            affected[u]=0;
            for(auto [v,w]: adj[u]){
                // relax u via v
                if(distv[v] + w < distv[u]){
                    long long old=distv[u];
                    distv[u]=distv[v]+w;
                    parentv[u]=v;
                    any=true;
                    #pragma omp critical
                    cout<<"[ITER "<<iter<<"] u="<<u<<" via v="<<v<<" "<<old<<"->"<<distv[u]<<"\n";
                }
                // relax v via u
                if(distv[u] + w < distv[v]){
                    long long old=distv[v];
                    distv[v]=distv[u]+w;
                    parentv[v]=u;
                    any=true;
                    affected[v]=1;
                    #pragma omp critical
                    cout<<"[ITER "<<iter<<"] v="<<v<<" via u="<<u<<" "<<old<<"->"<<distv[v]<<"\n";
                }
            }
        }
    }
    log_time("Phase II update_affected", t0);
    cout<<"[UPDATE] Converged in "<<iter<<" iterations\n";
}

//----------------------------------------------------------------
// MAIN
//----------------------------------------------------------------
int main(int argc, char** argv){
    if(argc<4){
        cerr<<"Usage: "<<argv[0]
            <<" <graph_adj.txt> <updates.txt> <source> [out_dist.txt]\n";
        return 1;
    }
    string graphFile  = argv[1];
    string updateFile = argv[2];
    int source        = stoi(argv[3]);
    bool saveOut = (argc==5);
    string outFile = saveOut ? argv[4] : "";

    // 1) Load graph (METIS format)
    ifstream gin(graphFile);
    long long m; int fmt;
    gin>>n>>m>>fmt;
    gin.ignore(numeric_limits<streamsize>::max(), '\n');
    adj.assign(n+1,{});
    for(int u=1;u<=n;u++){
        string line; getline(gin,line);
        istringstream iss(line);
        int v,w;
        while(iss>>v>>w) adj[u].emplace_back(v,w);
    }
    cout<<"[LOAD] n="<<n<<", edges(meta)="<<m<<"\n";

    // prepare arrays
    distv.resize(n+1);
    parentv.resize(n+1);
    affectedDel.resize(n+1);
    affected.resize(n+1);

    // 2) Initial SSSP
    init_sssp(source);

    // 3) Read updates
    vector<pair<int,int>> dels;
    vector<tuple<int,int,int>> ins;
    ifstream uin(updateFile);
    string op;
    while(uin>>op){
        if(op=="D"){
            int u,v; uin>>u>>v; dels.emplace_back(u,v);
        } else if(op=="I"){
            int u,v,w; uin>>u>>v>>w; ins.emplace_back(u,v,w);
        }
    }
    cout<<"[UPDATES] dels="<<dels.size()<<", ins="<<ins.size()<<"\n";

    // 4) Phase I
    process_changes(dels, ins);

    // 5) Phase II
    update_affected();

    // 6) Optionally save distances
    if(saveOut){
        ofstream fout(outFile);
        for(int u=1;u<=n;u++){
            fout<<u<<" "<<(distv[u]>=INF?-1:distv[u])<<" "<<parentv[u]<<"\n";
        }
        cout<<"[OUTPUT] Written distances to "<<outFile<<"\n";
    }

    return 0;
}
