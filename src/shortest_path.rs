//! Shortest path algorithms: Dijkstra, Bellman-Ford, Floyd-Warshall.

use crate::graph::Graph;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq)]
struct State {
    cost: f64,
    node: usize,
}

impl Eq for State {}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.cost.partial_cmp(&self.cost)
    }
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Dijkstra's algorithm. Returns (distances, predecessors) from source `s`.
/// Unreachable vertices have distance f64::INFINITY.
pub fn dijkstra(g: &Graph, s: usize) -> (Vec<f64>, Vec<Option<usize>>) {
    let mut dist = vec![f64::INFINITY; g.n];
    let mut pred = vec![None; g.n];
    let mut visited = vec![false; g.n];
    dist[s] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(State { cost: 0.0, node: s });
    while let Some(State { cost, node }) = heap.pop() {
        if visited[node] { continue; }
        visited[node] = true;
        for &(v, w) in &g.adj[node] {
            let new_dist = cost + w;
            if new_dist < dist[v] {
                dist[v] = new_dist;
                pred[v] = Some(node);
                heap.push(State { cost: new_dist, node: v });
            }
        }
    }
    (dist, pred)
}

/// Bellman-Ford algorithm. Returns `None` if negative cycle detected.
pub fn bellman_ford(g: &Graph, s: usize) -> Option<(Vec<f64>, Vec<Option<usize>>)> {
    let mut dist = vec![f64::INFINITY; g.n];
    let mut pred = vec![None; g.n];
    dist[s] = 0.0;
    for _ in 0..g.n - 1 {
        for u in 0..g.n {
            if dist[u].is_infinite() { continue; }
            for &(v, w) in &g.adj[u] {
                let nd = dist[u] + w;
                if nd < dist[v] {
                    dist[v] = nd;
                    pred[v] = Some(u);
                }
            }
        }
    }
    // Check for negative cycle
    for u in 0..g.n {
        if dist[u].is_infinite() { continue; }
        for &(v, w) in &g.adj[u] {
            if dist[u] + w < dist[v] {
                return None;
            }
        }
    }
    Some((dist, pred))
}

/// Floyd-Warshall algorithm. Returns (distance matrix, next matrix for path reconstruction).
pub fn floyd_warshall(g: &Graph) -> (Vec<Vec<f64>>, Vec<Vec<Option<usize>>>) {
    let n = g.n;
    let mut dist = vec![vec![f64::INFINITY; n]; n];
    let mut next = vec![vec![None; n]; n];
    for i in 0..n {
        dist[i][i] = 0.0;
    }
    for u in 0..n {
        for &(v, w) in &g.adj[u] {
            if w < dist[u][v] {
                dist[u][v] = w;
                next[u][v] = Some(v);
            }
        }
    }
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                if dist[i][k] + dist[k][j] < dist[i][j] {
                    dist[i][j] = dist[i][k] + dist[k][j];
                    next[i][j] = next[i][k];
                }
            }
        }
    }
    (dist, next)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dijkstra() {
        let mut g = Graph::new(5);
        g.add_edge(0, 1, 10.0);
        g.add_edge(0, 2, 3.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(1, 3, 2.0);
        g.add_edge(2, 1, 4.0);
        g.add_edge(2, 3, 8.0);
        g.add_edge(2, 4, 2.0);
        g.add_edge(3, 4, 7.0);
        // Make it directed for this test
        let mut dg = Graph::new_directed(5);
        dg.add_edge(0, 1, 10.0);
        dg.add_edge(0, 2, 3.0);
        dg.add_edge(1, 2, 1.0);
        dg.add_edge(1, 3, 2.0);
        dg.add_edge(2, 1, 4.0);
        dg.add_edge(2, 3, 8.0);
        dg.add_edge(2, 4, 2.0);
        dg.add_edge(3, 4, 7.0);
        let (dist, _) = dijkstra(&dg, 0);
        assert!((dist[0] - 0.0).abs() < 1e-9);
        assert!((dist[1] - 7.0).abs() < 1e-9); // 0->2->1 = 3+4
        assert!((dist[2] - 3.0).abs() < 1e-9);
        assert!((dist[3] - 9.0).abs() < 1e-9); // 0->2->1->3 = 3+4+2
        assert!((dist[4] - 5.0).abs() < 1e-9); // 0->2->4 = 3+2
    }

    #[test]
    fn test_bellman_ford() {
        let mut g = Graph::new_directed(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, -1.0);
        g.add_edge(2, 3, 2.0);
        g.add_edge(0, 3, 5.0);
        let result = bellman_ford(&g, 0).unwrap();
        let dist = result.0;
        assert!((dist[3] - 2.0).abs() < 1e-9); // 0->1->2->3 = 1-1+2
    }

    #[test]
    fn test_bellman_ford_negative_cycle() {
        let mut g = Graph::new_directed(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, -1.0);
        g.add_edge(2, 0, -1.0);
        assert!(bellman_ford(&g, 0).is_none());
    }

    #[test]
    fn test_floyd_warshall() {
        let mut g = Graph::new_directed(4);
        g.add_edge(0, 1, 5.0);
        g.add_edge(0, 3, 10.0);
        g.add_edge(1, 2, 3.0);
        g.add_edge(2, 3, 1.0);
        let (dist, _) = floyd_warshall(&g);
        assert!((dist[0][3] - 9.0).abs() < 1e-9); // 0->1->2->3
    }

    #[test]
    fn test_dijkstra_triangle() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(0, 2, 5.0);
        let (dist, _) = dijkstra(&g, 0);
        assert!((dist[2] - 3.0).abs() < 1e-9); // 0->1->2
    }
}
