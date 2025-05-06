// dynamic_sssp_mpi.cpp
#include <iostream>
#include <vector>
#include <queue>
#include <tuple>
#include <limits>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <climits>
#include <set>
using namespace std;
using pii = pair<int,int>;
const long long INF = LLONG_MAX/4;

int nranks;
int n;  // local number of vertices
vector<vector<pii>> adj;
vector<long long> distv;
vector<int> parentv;
vector<char> affectedDel, affected;
set<pair<int,int>> updatedEdges;
vector<string> updatedEdgesLog;

vector<int> localToGlobal;
unordered_map<int,int> globalToLocal;


struct CrossEdge { int local, remoteGlob, w, peerPart; };
vector<CrossEdge> crossEdges;


bool edge_exists(int u, int v){
    for(auto &pr : adj[u]) if(pr.first==v) return true;
    return false;
}

template<class T>
void log_time(const char* phase, T t0){
    auto t1 = chrono::high_resolution_clock::now();
    auto ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    //#pragma omp critical
   // cout<<"[TIME] "<<phase<<" took "<<ms<<" ms\n";
}

void init_sssp(int src){
    auto t0 = chrono::high_resolution_clock::now();
    distv.assign(n+1, INF);
    parentv.assign(n+1, -1);
    distv[src] = 0;
    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
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
}

void process_changes(const vector<pair<int,int>>& dels,
                     const vector<tuple<int,int,int>>& ins){
    fill(affectedDel.begin(), affectedDel.end(), 0);
    fill(affected.begin(),    affected.end(),    0);
    auto t0 = chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for(int i=0;i<(int)dels.size();i++){
        auto [u,v] = dels[i];
        if(u<1||u>n||v<1||v>n) continue;
        bool inTree = (parentv[v]==u)||(parentv[u]==v);
        if(!inTree) continue;
        int child = (parentv[v]==u? v : u);
        affectedDel[child]=1; affected[child]=1;
        distv[child]=INF; parentv[child]=-1;
    }

    vector<tuple<int,int,int>> crossBuf;
    #pragma omp parallel
    {
        vector<tuple<int,int,int>> threadBuf;
        #pragma omp for nowait
        for(int i=0;i<(int)ins.size();i++){
            auto [u,v,w] = ins[i];
            bool localU = (u>=1 && u<=n);
            bool localV = (v>=1 && v<=n);
            if(localU && localV){
                if(distv[u]+w < distv[v]){
                    #pragma omp critical
                    {
                        distv[v]=distv[u]+w;
                        parentv[v]=u;
                        affected[v]=1;
                        updatedEdgesLog.push_back(to_string(localToGlobal[u]) + " " + to_string(localToGlobal[v]) + " (updated)");
                    }
                    
                }
                if(distv[v]+w < distv[u]){
                    #pragma omp critical
                    {
                        distv[u]=distv[v]+w;
                        parentv[u]=v;
                        affected[u]=1;
                        updatedEdgesLog.push_back(to_string(localToGlobal[v]) + " " + to_string(localToGlobal[u]) + " (updated)");
                    }
                }
                #pragma omp critical
                if(!edge_exists(u,v)){
                    adj[u].emplace_back(v,w);
                    adj[v].emplace_back(u,w);
                }
            } else {
                threadBuf.emplace_back(u,v,w);
            }
        }
        #pragma omp critical
        for(auto &t:threadBuf) crossBuf.push_back(t);
    }

    #pragma omp barrier
    #pragma omp single
    {
      // use the global nranks set in main()
      int P = nranks;
  
      // 1) build per‐peer send buffers
      vector<vector<long long>> sbuf(P);
      for(auto &e: crossEdges){
        long long d = distv[e.local];
        if(d < INF){
          sbuf[e.peerPart].push_back(e.remoteGlob);
          sbuf[e.peerPart].push_back(d);
        }
      }
  
      // 2) flatten into one big array + build counts/displs
      vector<int> sendCounts(P), sendDispls(P), recvCounts(P), recvDispls(P);
      int totalSend = 0, totalRecv = 0;
      for(int p=0; p<P; p++){
        sendCounts[p] = sbuf[p].size();
        sendDispls[p] = totalSend;
        totalSend += sendCounts[p];
      }
      vector<long long> sendFlat(totalSend);
      for(int p=0; p<P; p++){
        copy(sbuf[p].begin(), sbuf[p].end(), sendFlat.begin() + sendDispls[p]);
      }
  
      // 3) exchange counts
      MPI_Alltoall(sendCounts.data(), 1, MPI_INT,
                   recvCounts.data(), 1, MPI_INT,
                   MPI_COMM_WORLD);
      for(int p=0; p<P; p++){
        recvDispls[p] = totalRecv;
        totalRecv += recvCounts[p];
      }
      vector<long long> recvFlat(totalRecv);
  
      // 4) the variable‐length all‑to‑all
      MPI_Alltoallv(
        sendFlat.data(), sendCounts.data(), sendDispls.data(), MPI_LONG_LONG,
        recvFlat.data(), recvCounts.data(), recvDispls.data(), MPI_LONG_LONG,
        MPI_COMM_WORLD
      );
  
      // 5) apply incoming updates
      for(int i=0; i<totalRecv; i+=2){
        int gV = (int)recvFlat[i];
        long long d = recvFlat[i+1];
        auto it = globalToLocal.find(gV);
        if(it != globalToLocal.end()){
          int loc = it->second;
          if(d < distv[loc]){
            distv[loc] = d;
            affected[loc] = 1;
          }
        }
      }
    }

}

void update_affected(){
    auto t0 = chrono::high_resolution_clock::now();
    queue<int> Q;
    for(int i=1;i<=n;i++) if(affectedDel[i]) Q.push(i);
    while(!Q.empty()){
        int u=Q.front(); Q.pop();
        for(auto [v,w]: adj[u]){
            if(parentv[v]==u && !affectedDel[v]){
                affectedDel[v]=1; affected[v]=1;
                distv[v]=INF; parentv[v]=-1;
                Q.push(v);
            }
        }
    }
    bool any=true; int it=0;
    while(any){
        it++; any=false;
        #pragma omp parallel for reduction(|:any)
        for(int u=1;u<=n;u++){
            if(!affected[u]) continue;
            affected[u]=0;
            for(auto [v,w]: adj[u]){
                if(distv[v]+w < distv[u]){
                    distv[u]=distv[v]+w;
                    parentv[u]=v;
                    any=true;
                    #pragma omp critical
                    updatedEdgesLog.push_back(to_string(localToGlobal[v]) + " " + to_string(localToGlobal[u]) + " (updated iterative)");
                }
                if(distv[u]+w < distv[v]){
                    distv[v]=distv[u]+w;
                    parentv[v]=u;
                    any=true;
                    affected[v]=1;
                    #pragma omp critical
                    updatedEdgesLog.push_back(to_string(localToGlobal[u]) + " " + to_string(localToGlobal[v]) + " (updated iterative)");
                }
            }
        }
    }
}

int main(int argc, char** argv){
    MPI_Init(&argc,&argv);
    int rank; 
    MPI_Comm_rank(MPI_COMM_WORLD,&rank); 
    MPI_Comm_size(MPI_COMM_WORLD,&nranks);

    double total_start = MPI_Wtime(); 

    if (nranks<1) {
        if (rank==0) cerr<<"Requires at least 1 MPI rank\n";
        MPI_Finalize();
        return 1;
      }
      
    if(argc<6){ if(rank==0) cerr<<"Usage: "<<argv[0]<<" partX.txt partX.ids updates.txt cross_edges.txt source\n"; MPI_Finalize(); return 1; }

    string partFile   = argv[1],
           idsFile    = argv[2],
           updatesFile= argv[3],
           crossFile  = argv[4];
    int source       = stoi(argv[5]);

    // load ids
    ifstream iin(idsFile);
    localToGlobal.clear();
    localToGlobal.push_back(-1);
    int g;
    while(iin>>g) localToGlobal.push_back(g);
    n = (int)localToGlobal.size()-1;
    globalToLocal.clear();
    for(int i=1;i<=n;i++) globalToLocal[ localToGlobal[i] ] = i;

    ifstream in(partFile);
    long long m; int fmt;
    in>>g>>m>>fmt; in.ignore(numeric_limits<streamsize>::max(), '\n');
    adj.assign(n+1,{});
    for(int u=1;u<=n;u++){
        string line; getline(in,line);
        istringstream iss(line);
        int v,w;
        while(iss>>v>>w) if(v>=1 && v<=n) adj[u].emplace_back(v,w);
    }
    cout<<"[R"<<rank<<"] n="<<n<<" m_meta="<<m<<"\n";


    ifstream cin_cross(crossFile);
    int u,v,w,p,q;
    while(cin_cross>>u>>v>>w>>p>>q){
      if (p==rank) {
        // u belongs to me, v belongs to peer q
        int loc = globalToLocal[u];
        crossEdges.push_back({loc, v, w, q});
      }
      else if (q==rank) {
        // v belongs to me, u belongs to peer p
        int loc = globalToLocal[v];
        crossEdges.push_back({loc, u, w, p});
      }
    }
    cout<<"[R"<<rank<<"] crossEdges="<<crossEdges.size()<<"\n";
    cout<<"[R"<<rank<<"] crossEdges="<<crossEdges.size()<<"\n";

    distv.resize(n+1); parentv.resize(n+1);
    affectedDel.resize(n+1); affected.resize(n+1);

    init_sssp(source);

    vector<pair<int,int>> dels; vector<tuple<int,int,int>> ins;
    { ifstream in(updatesFile); string op;
      while(in>>op){
        if(op=="D"){int u,v; in>>u>>v; 
           if(globalToLocal.count(u)&& globalToLocal.count(v))
             dels.emplace_back(globalToLocal[u], globalToLocal[v]);
        } else {
          int u,v,w; in>>u>>v>>w;
          ins.emplace_back(u,v,w);
        }
      }
    }

    process_changes(dels,ins);
    update_affected();

    vector<int> counts(nranks), displs(nranks);
    int myCount = n;
    MPI_Gather(&myCount,1,MPI_INT,counts.data(),1,MPI_INT,0,MPI_COMM_WORLD);
    if(rank==0){ displs[0]=0; for(int i=1;i<nranks;i++) displs[i]=displs[i-1]+counts[i-1]; }
    vector<long long> allDist;
    if(rank==0) allDist.resize(displs[nranks-1]+counts[nranks-1]);
    MPI_Gatherv(distv.data()+1, n, MPI_LONG_LONG,
                allDist.data(), counts.data(), displs.data(), MPI_LONG_LONG,
                0, MPI_COMM_WORLD);

    if(rank==0){
        ofstream out("final_output.txt");
        for(int i=0;i<(int)allDist.size();i++){
            long long d=allDist[i];
            out<<i+1<<" "<<(d>=INF?-1:d) <<"\n";
        }
        cout<<"[R0] final_output.txt written\n";

        ofstream logOut("updated_edges_log.txt");
        for(const auto& line : updatedEdgesLog)
            logOut << line << "\n";
        cout << "[R0] updated_edges_log.txt written\n";

    }

    double total_end = MPI_Wtime();  // End total timer
if(rank == 0){
    std::cout << "[TIME] Total execution time: " << (total_end - total_start) * 1000 << " ms\n";
}

    MPI_Finalize();
    return 0;
}
