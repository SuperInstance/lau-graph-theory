//! Graph traversal: BFS, DFS, topological sort.

use crate::graph::Graph;
use std::collections::VecDeque;

/// BFS from source `s`. Returns vertices in BFS order.
pub fn bfs(g: &Graph, s: usize) -> Vec<usize> {
    let mut visited = vec![false; g.n];
    let mut order = Vec::new();
    let mut queue = VecDeque::new();
    visited[s] = true;
    queue.push_back(s);
    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &(v, _) in &g.adj[u] {
            if !visited[v] {
                visited[v] = true;
                queue.push_back(v);
            }
        }
    }
    order
}

/// DFS from source `s`. Returns vertices in DFS order.
pub fn dfs(g: &Graph, s: usize) -> Vec<usize> {
    let mut visited = vec![false; g.n];
    let mut order = Vec::new();
    dfs_rec(g, s, &mut visited, &mut order);
    order
}

fn dfs_rec(g: &Graph, u: usize, visited: &mut [bool], order: &mut Vec<usize>) {
    visited[u] = true;
    order.push(u);
    for &(v, _) in &g.adj[u] {
        if !visited[v] {
            dfs_rec(g, v, visited, order);
        }
    }
}

/// Topological sort using Kahn's algorithm. Returns `None` if cycle exists.
pub fn topological_sort(g: &Graph) -> Option<Vec<usize>> {
    assert!(g.directed, "Topological sort requires a directed graph");
    let mut in_deg = vec![0usize; g.n];
    for u in 0..g.n {
        for &(v, _) in &g.adj[u] {
            in_deg[v] += 1;
        }
    }
    let mut queue: VecDeque<usize> = VecDeque::new();
    for i in 0..g.n {
        if in_deg[i] == 0 {
            queue.push_back(i);
        }
    }
    let mut order = Vec::new();
    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &(v, _) in &g.adj[u] {
            in_deg[v] -= 1;
            if in_deg[v] == 0 {
                queue.push_back(v);
            }
        }
    }
    if order.len() == g.n { Some(order) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfs_simple() {
        let mut g = Graph::new(4);
        g.add_edge_unit(0, 1);
        g.add_edge_unit(0, 2);
        g.add_edge_unit(1, 3);
        let order = bfs(&g, 0);
        assert_eq!(order[0], 0);
        assert!(order.contains(&1));
        assert!(order.contains(&2));
        assert!(order.contains(&3));
    }

    #[test]
    fn test_dfs_simple() {
        let mut g = Graph::new(3);
        g.add_edge_unit(0, 1);
        g.add_edge_unit(1, 2);
        let order = dfs(&g, 0);
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn test_topological_sort() {
        let mut g = Graph::new_directed(4);
        g.add_edge_unit(0, 1);
        g.add_edge_unit(0, 2);
        g.add_edge_unit(1, 3);
        g.add_edge_unit(2, 3);
        let order = topological_sort(&g).unwrap();
        assert_eq!(order[0], 0);
        assert!(order.iter().position(|&x| x == 1) < order.iter().position(|&x| x == 3));
        assert!(order.iter().position(|&x| x == 2) < order.iter().position(|&x| x == 3));
    }

    #[test]
    fn test_topological_cycle() {
        let mut g = Graph::new_directed(3);
        g.add_edge_unit(0, 1);
        g.add_edge_unit(1, 2);
        g.add_edge_unit(2, 0);
        assert!(topological_sort(&g).is_none());
    }
}
