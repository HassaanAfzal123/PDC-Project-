#include <iostream>
#include <vector>
#include <queue>
#include <tuple>
#include <limits>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <chrono>
#include <mpi.h>
#include <climits>
using namespace std;
using ll = long long;
const ll INF = LLONG_MAX/4;

struct CrossEdge {
    int local;        // local index
    int remoteGlob;   // global ID of the other endpoint
    int w;            // weight
    int peerPart;     // rank of the other partition
};

int main(int argc, char** argv){
    MPI_Init(&argc,&argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nranks);

    if(argc<6){
        if(rank==0)
            cerr<<"Usage: "<<argv[0]
                <<" partX.txt partX.ids updates.txt cross_edges.txt sourceGlob\n";
        MPI_Finalize();
        return 1;
    }

    string partFile    = argv[1],
           idsFile     = argv[2],
           updatesFile = argv[3],
           crossFile   = argv[4];
    int sourceGlob    = stoi(argv[5]);

    ifstream inPart(partFile);
    if(!inPart){
        if(rank==0) cerr<<"Cannot open "<<partFile<<"\n";
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    int header_n, fmt;
    ll m_meta;
    inPart >> header_n >> m_meta >> fmt;
    inPart.ignore(numeric_limits<streamsize>::max(),'\n');
    int n = header_n;

    vector<int> localToGlobal(1, -1);
    {
        ifstream inIDs(idsFile);
        if(!inIDs){
            if(rank==0) cerr<<"Cannot open "<<idsFile<<"\n";
            MPI_Abort(MPI_COMM_WORLD,1);
        }
        int g;
        while(inIDs>>g) localToGlobal.push_back(g);
    }
    if((int)localToGlobal.size()-1 != n){
        if(rank==0) cerr<<"Error: idsFile length != n\n";
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    unordered_map<int,int> globalToLocal;
    globalToLocal.reserve(n);
    for(int i=1;i<=n;i++){
        globalToLocal[ localToGlobal[i] ] = i;
    }

    vector<vector<pair<int,int>>> adj(n+1);
    {
        string line;
        for(int u=1;u<=n;u++){
            getline(inPart,line);
            istringstream iss(line);
            int v,w;
            while(iss>>v>>w){
                auto it = globalToLocal.find(v);
                if(it!=globalToLocal.end()){
                    int vl = it->second;
                    adj[u].emplace_back(vl,w);
                }
                // else: cross‐edge, skip here
            }
        }
    }
    inPart.close();

    vector<CrossEdge> crossEdges;
    {
        ifstream inCross(crossFile);
        if(!inCross){
            if(rank==0) cerr<<"Cannot open "<<crossFile<<"\n";
            MPI_Abort(MPI_COMM_WORLD,1);
        }
        int u_gl, v_gl, w, p, q;
        while(inCross>>u_gl>>v_gl>>w>>p>>q){
            if(p==rank && globalToLocal.count(u_gl)){
                crossEdges.push_back({ globalToLocal[u_gl], v_gl, w, q });
            }
            else if(q==rank && globalToLocal.count(v_gl)){
                crossEdges.push_back({ globalToLocal[v_gl], u_gl, w, p });
            }
        }
    }

    int sourceLocal = globalToLocal[sourceGlob];
    vector<ll> dist(n+1, INF);
    vector<int> parent(n+1, -1);
    dist[sourceLocal] = 0;
    priority_queue<pair<ll,int>,vector<pair<ll,int>>,greater<>> pq;
    pq.push({0,sourceLocal});
    while(!pq.empty()){
        auto [d,u] = pq.top(); pq.pop();
        if(d!=dist[u]) continue;
        for(auto &e: adj[u]){
            int v=e.first, wt=e.second;
            if(dist[v] > d + wt){
                dist[v] = d + wt;
                parent[v] = u;
                pq.push({dist[v],v});
            }
        }
    }

    vector<pair<int,int>> dels;
    vector<tuple<int,int,int>> ins;
    {
        ifstream inUpd(updatesFile);
        string op;
        while(inUpd>>op){
            if(op=="D"){
                int u,v; inUpd>>u>>v;
                if(globalToLocal.count(u)&&globalToLocal.count(v))
                    dels.emplace_back(globalToLocal[u], globalToLocal[v]);
            } else {
                int u,v,w; inUpd>>u>>v>>w;
                ins.emplace_back(u,v,w);
            }
        }
    }

    vector<char> affected(n+1,0), affectedDel(n+1,0);
    vector<pair<int,int>> updatedLog;

    for(auto &d: dels){
        int u=d.first, v=d.second;
        if(parent[v]==u || parent[u]==v){
            int c = (parent[v]==u ? v : u);
            affectedDel[c]=1; affected[c]=1;
            dist[c]=INF; parent[c]=-1;
        }
    }
    for(auto &t: ins){
        int gu,gv,w; tie(gu,gv,w)=t;
        if(globalToLocal.count(gu)&&globalToLocal.count(gv)){
            int u=globalToLocal[gu], v=globalToLocal[gv];
            if(dist[u]+w < dist[v]){
                dist[v]=dist[u]+w; parent[v]=u; affected[v]=1;
                updatedLog.emplace_back(gu,gv);
            }
            if(dist[v]+w < dist[u]){
                dist[u]=dist[v]+w; parent[u]=v; affected[u]=1;
                updatedLog.emplace_back(gv,gu);
            }
            bool ok=false;
            for(auto &e:adj[u]) if(e.first==v){ ok=true; break; }
            if(!ok){
                adj[u].emplace_back(v,w);
                adj[v].emplace_back(u,w);
            }
        }
    }

    {
        vector<vector<ll>> sbuf(nranks);
        for(auto &ce: crossEdges){
            ll d = dist[ce.local];
            if(d<INF){
                sbuf[ce.peerPart].push_back(ce.remoteGlob);
                sbuf[ce.peerPart].push_back(d);
            }
        }
        vector<int> sc(nranks), sd(nranks), rc(nranks), rd(nranks);
        int ts=0,tr=0;
        for(int p=0;p<nranks;p++){
            sc[p] = sbuf[p].size();
            sd[p] = ts; ts += sc[p];
        }
        vector<ll> sflat(ts);
        for(int p=0;p<nranks;p++)
            copy(sbuf[p].begin(), sbuf[p].end(), sflat.begin()+sd[p]);
        MPI_Alltoall(sc.data(),1,MPI_INT, rc.data(),1,MPI_INT, MPI_COMM_WORLD);
        for(int p=0;p<nranks;p++){
            rd[p]=tr; tr+=rc[p];
        }
        vector<ll> rflat(tr);
        MPI_Alltoallv(
          sflat.data(), sc.data(), sd.data(), MPI_LONG_LONG,
          rflat.data(), rc.data(), rd.data(), MPI_LONG_LONG,
          MPI_COMM_WORLD
        );
        for(int i=0;i<tr;i+=2){
            int gV = (int)rflat[i];
            ll  d  = rflat[i+1];
            auto it = globalToLocal.find(gV);
            if(it!=globalToLocal.end()){
                int loc = it->second;
                if(d < dist[loc]){
                    dist[loc]=d;
                    affected[loc]=1;
                    updatedLog.emplace_back(gV, localToGlobal[loc]);
                }
            }
        }
    }

    queue<int> q;
    for(int i=1;i<=n;i++) if(affectedDel[i]) q.push(i);
    while(!q.empty()){
        int u=q.front(); q.pop();
        for(auto &e:adj[u]){
            int v=e.first;
            if(parent[v]==u && !affectedDel[v]){
                affectedDel[v]=1; affected[v]=1;
                dist[v]=INF; parent[v]=-1;
                q.push(v);
            }
        }
    }
    bool any=true;
    while(any){
        any=false;
        for(int u=1;u<=n;u++){
            if(!affected[u]) continue;
            affected[u]=0;
            for(auto &e:adj[u]){
                int v=e.first, w=e.second;
                if(dist[v]+w < dist[u]){
                    dist[u]=dist[v]+w; parent[u]=v; any=true;
                    updatedLog.emplace_back(localToGlobal[v], localToGlobal[u]);
                }
                if(dist[u]+w < dist[v]){
                    dist[v]=dist[u]+w; parent[v]=u; any=true; affected[v]=1;
                    updatedLog.emplace_back(localToGlobal[u], localToGlobal[v]);
                }
            }
        }
    }

    vector<int> counts(nranks), displs(nranks);
    int myCount=n;
    MPI_Gather(&myCount,1,MPI_INT, counts.data(),1,MPI_INT, 0, MPI_COMM_WORLD);
    if(rank==0){
        displs[0]=0;
        for(int i=1;i<nranks;i++)
            displs[i]=displs[i-1]+counts[i-1];
    }
    vector<ll> allDist;
    if(rank==0) allDist.resize(displs[nranks-1]+counts[nranks-1]);
    MPI_Gatherv(dist.data()+1, n, MPI_LONG_LONG,
                allDist.data(), counts.data(), displs.data(), MPI_LONG_LONG,
                0, MPI_COMM_WORLD);

    if(rank==0){
        ofstream fo("final_output.txt");
        for(int i=0;i<(int)allDist.size();i++)
            fo<<i+1<<" "<<(allDist[i]>=INF?-1:allDist[i])<<"\n";
        ofstream lu("updated_edges_log.txt");
        for(auto &e: updatedLog)
            lu<<e.first<<" "<<e.second<<"\n";
    }

    MPI_Finalize();
    return 0;
}
