# lau-graph-theory

A Rust library for **graph theory** — data structures, traversals, shortest paths, minimum spanning trees, network flow, connectivity analysis, matching, spectral graph theory, and agent-network analysis. Built on [nalgebra](https://nalgebra.org), [serde](https://serde.rs), and [rand](https://docs.rs/rand).

## What This Does

This crate provides a complete toolkit for working with graphs: constructing directed and undirected weighted graphs, running classical algorithms (Dijkstra, Bellman-Ford, Floyd-Warshall, Kruskal, Prim, Ford-Fulkerson, Hopcroft-Karp), analyzing structure (connected components, bridges, articulation points, biconnected components), computing spectral properties (Laplacian, algebraic connectivity, Fiedler vector, Cheeger constant), and performing agent-network analysis (betweenness centrality, influence propagation, clustering coefficients).

**78 unit tests** cover every module.

## Key Idea

Graphs model relationships. This library treats a graph as an adjacency list with edge weights, then layers algorithms on top:

- **Traversals** (BFS, DFS) explore reachability.
- **Shortest paths** find optimal routes under different assumptions (non-negative weights → Dijkstra, negative weights → Bellman-Ford, all-pairs → Floyd-Warshall).
- **MST algorithms** find minimum-cost spanning subgraphs.
- **Network flow** pushes maximum flow through capacity-constrained edges.
- **Connectivity** reveals structural weak points (bridges, articulation points).
- **Matching** pairs vertices optimally (bipartite maximum matching).
- **Spectral analysis** uses the Laplacian matrix to measure connectivity and partition quality.
- **Agent networks** apply graph metrics to social/influence networks.

## Install

```toml
[dependencies]
lau-graph-theory = { git = "https://github.com/SuperInstance/lau-graph-theory" }
```

### Dependencies

| Crate | Purpose |
|-------|---------|
| `nalgebra` | Laplacian matrix eigenvalue computation |
| `serde` | Serialization of graphs and results |
| `rand` | Stochastic influence propagation simulation |

## Quick Start

```rust
use lau_graph_theory::*;

// Build a weighted undirected graph
let mut g = Graph::new(5);
g.add_edge(0, 1, 4.0);
g.add_edge(0, 2, 1.0);
g.add_edge(1, 2, 2.0);
g.add_edge(1, 3, 1.0);
g.add_edge(2, 3, 5.0);
g.add_edge(3, 4, 3.0);

// Shortest paths from vertex 0
let (dist, parent) = shortest_path::dijkstra(&g, 0);
println!("Distance 0→4: {}", dist[4]); // 8.0

// Minimum spanning tree
let mst_edges = mst::kruskal(&g);
let total_weight: f64 = mst_edges.iter().map(|e| e.2).sum();
println!("MST weight: {}", total_weight);

// Spectral analysis
let ac = spectral::algebraic_connectivity(&g);
println!("Algebraic connectivity: {}", ac); // > 0 means connected

// Network analysis
let stats = agent_network::network_stats(&g);
println!("Density: {}, Components: {}", stats.density, stats.num_components);
```

## API Reference

### `graph` — Core Data Structure

| Type / Function | Description |
|----------------|-------------|
| `Graph` | Adjacency-list graph with optional directed flag |
| `Graph::new(n)` | Create an empty graph with n vertices |
| `add_edge(u, v, w)` | Add weighted edge (u, v) |
| `add_edge_unit(u, v)` | Add unweighted (w=1.0) edge |
| `edge_count()` | Number of edges |
| `neighbors(v)` | Iterator over (neighbor, weight) pairs |

### `traversal` — BFS & DFS

| Function | Returns |
|----------|---------|
| `bfs(g, start)` | Visited order |
| `dfs(g, start)` | Visited order |
| `bfs_path(g, start, end)` | Shortest unweighted path |
| `dfs_path(g, start, end)` | A path (not necessarily shortest) |

### `shortest_path` — Path Algorithms

| Function | Algorithm | Weights | Returns |
|----------|-----------|---------|---------|
| `dijkstra(g, src)` | Dijkstra | Non-negative | (dist[], parent[]) |
| `bellman_ford(g, src)` | Bellman-Ford | Any (detects cycles) | (dist[], has_negative_cycle) |
| `floyd_warshall(g)` | Floyd-Warshall | Any | (dist[][], next[][]) |

### `mst` — Minimum Spanning Trees

| Function | Algorithm | Returns |
|----------|-----------|---------|
| `kruskal(g)` | Kruskal (union-find) | Vec of (u, v, weight) |
| `prim(g, start)` | Prim (priority queue) | Vec of (u, v, weight) |

### `flow` — Network Flow

| Function | Description |
|----------|-------------|
| `ford_fulkerson(g, src, sink)` | Maximum flow value |
| `max_flow_with_paths(g, src, sink)` | Flow value + augmenting paths traced |

### `connectivity` — Structural Analysis

| Function | Returns |
|----------|---------|
| `connected_components(g)` | Vec of component vertex sets |
| `bridges(g)` | Vec of bridge edges (u, v) |
| `articulation_points(g)` | Vec of articulation vertices |
| `biconnected_components(g)` | Vec of biconnected subgraph edge sets |

### `matching` — Bipartite Matching

| Function | Description |
|----------|-------------|
| `maximum_bipartite_matching(g, left_set, right_set)` | Maximum cardinality matching |

### `spectral` — Spectral Graph Theory

| Function | Returns |
|----------|---------|
| `laplacian_matrix(g)` | DMatrix<f64> — L = D − A |
| `laplacian_spectrum(g)` | Eigenvalues sorted ascending |
| `algebraic_connectivity(g)` | λ₂ (second-smallest eigenvalue) |
| `fiedler_vector(g)` | Eigenvector for λ₂ + eigenvalue |
| `cheeger_constant_approx(g)` | Spectral approximation of edge expansion |
| `cheeger_inequality_bounds(g)` | (λ₂/2, √(2λ₂)) bounds |

### `agent_network` — Network Analysis

| Type / Function | Description |
|----------------|-------------|
| `NetworkStats` | Agent count, connections, degree, density, components, clustering |
| `network_stats(g)` | Compute all summary statistics |
| `clustering_coefficient(g, v)` | Local clustering at vertex v |
| `influence_ranking(g)` | Betweenness centrality ranking (descending) |
| `simulate_influence(g, seeds, β, steps)` | SI model epidemic simulation |
| `diameter(g)` | Longest shortest path |
| `network_center(g)` | Vertex with minimum eccentricity |

## How It Works

### Graph Representation

`Graph` stores an adjacency list: `adj: Vec<Vec<(usize, f64)>>` where `adj[u]` is a list of `(neighbor, weight)` pairs. Edges are stored in both directions for undirected graphs.

### Algorithm Complexity

| Algorithm | Time | Space |
|-----------|------|-------|
| BFS / DFS | O(V + E) | O(V) |
| Dijkstra | O((V + E) log V) | O(V) |
| Bellman-Ford | O(VE) | O(V) |
| Floyd-Warshall | O(V³) | O(V²) |
| Kruskal | O(E log E) | O(V + E) |
| Prim | O(E log V) | O(V) |
| Ford-Fulkerson | O(E · max_flow) | O(V + E) |
| Betweenness centrality | O(V²E) | O(V²) |
| Laplacian spectrum | O(V³) | O(V²) |

### Spectral Flow

```
Graph → Adjacency matrix A → Degree matrix D → Laplacian L = D − A
  → Eigenvalues of L → λ₂ = algebraic connectivity
  → Eigenvector for λ₂ = Fiedler vector → graph partitioning
  → Cheeger inequality: λ₂/2 ≤ h(G) ≤ √(2λ₂)
```

## The Math

### The Graph Laplacian

For a graph with adjacency matrix A and degree matrix D (diagonal, D_ii = Σ_j A_ij):

> L = D − A

Properties:
- L is positive semi-definite
- λ₁ = 0 (always, eigenvector = all-ones)
- λ₂ > 0 ⟺ graph is connected
- λ₂ is the **algebraic connectivity** — larger means better connected

### Algebraic Connectivity & Fiedler Vector

The **Fiedler vector** (eigenvector for λ₂) partitions the graph: vertices with positive entries go in one group, negative in another. This is the foundation of **spectral clustering**.

### Cheeger Inequality

The **Cheeger constant** h(G) measures the "bottleneck" of a graph — the minimum edge cut to separate a fraction of vertices. Computing h(G) exactly is NP-hard, but the spectral approximation satisfies:

> λ₂ / 2 ≤ h(G) ≤ √(2 λ₂)

This connects the algebraic (eigenvalue) and geometric (edge cut) views of connectivity.

### Betweenness Centrality

The betweenness of vertex v is the fraction of shortest paths that pass through v:

> C_B(v) = Σ_{s≠v≠t} σ_{st}(v) / σ_{st}

where σ_{st} is the number of shortest s-t paths and σ_{st}(v) is how many pass through v. High-betweenness vertices are brokers or bridges in the network.

### Ford-Fulkerson Max Flow

The max-flow min-cut theorem states that the maximum flow from source to sink equals the minimum cut capacity separating source from sink. Ford-Fulkerson finds augmenting paths via BFS (Edmonds-Karp variant) and saturates them iteratively.

## License

MIT
