//! Network flow: Ford-Fulkerson max flow using BFS (Edmonds-Karp).

use crate::graph::Graph;
use std::collections::VecDeque;

/// Compute max flow from `source` to `sink` using Edmonds-Karp (BFS-based Ford-Fulkerson).
/// Returns the maximum flow value and the flow on each edge.
pub fn max_flow(g: &Graph, source: usize, sink: usize) -> (f64, Vec<(usize, usize, f64)>) {
    let n = g.n;
    // Build capacity matrix
    let mut cap = vec![vec![0.0f64; n]; n];
    for u in 0..n {
        for &(v, w) in &g.adj[u] {
            cap[u][v] += w;
        }
    }
    let mut flow = vec![vec![0.0f64; n]; n];

    let mut total_flow = 0.0;
    loop {
        // BFS to find augmenting path
        let mut parent = vec![None; n];
        let mut visited = vec![false; n];
        visited[source] = true;
        let mut queue = VecDeque::new();
        queue.push_back(source);
        while let Some(u) = queue.pop_front() {
            if u == sink { break; }
            for v in 0..n {
                if !visited[v] && cap[u][v] - flow[u][v] > 1e-12 {
                    visited[v] = true;
                    parent[v] = Some(u);
                    queue.push_back(v);
                }
            }
        }
        if !visited[sink] { break; }

        // Find bottleneck
        let mut bottleneck = f64::INFINITY;
        let mut v = sink;
        while let Some(u) = parent[v] {
            bottleneck = bottleneck.min(cap[u][v] - flow[u][v]);
            v = u;
        }

        // Update flow
        v = sink;
        while let Some(u) = parent[v] {
            flow[u][v] += bottleneck;
            flow[v][u] -= bottleneck;
            v = u;
        }
        total_flow += bottleneck;
    }

    let mut flow_edges = Vec::new();
    for u in 0..n {
        for v in 0..n {
            if flow[u][v] > 1e-12 {
                flow_edges.push((u, v, flow[u][v]));
            }
        }
    }
    (total_flow, flow_edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_flow_simple() {
        // Classic diamond network
        let mut g = Graph::new_directed(4);
        g.add_edge(0, 1, 3.0);
        g.add_edge(0, 2, 2.0);
        g.add_edge(1, 3, 2.0);
        g.add_edge(2, 3, 3.0);
        let (flow, _) = max_flow(&g, 0, 3);
        assert!((flow - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_max_flow_classic() {
        let mut g = Graph::new_directed(6);
        g.add_edge(0, 1, 16.0);
        g.add_edge(0, 2, 13.0);
        g.add_edge(1, 2, 10.0);
        g.add_edge(1, 3, 12.0);
        g.add_edge(2, 1, 4.0);
        g.add_edge(2, 4, 14.0);
        g.add_edge(3, 2, 9.0);
        g.add_edge(3, 5, 20.0);
        g.add_edge(4, 3, 7.0);
        g.add_edge(4, 5, 4.0);
        let (flow, _) = max_flow(&g, 0, 5);
        assert!((flow - 23.0).abs() < 1e-9);
    }

    #[test]
    fn test_max_flow_no_path() {
        let mut g = Graph::new_directed(3);
        g.add_edge(0, 1, 5.0);
        // no edge from 1 to 2 or 0 to 2
        let (flow, _) = max_flow(&g, 0, 2);
        assert!((flow - 0.0).abs() < 1e-9);
    }
}
