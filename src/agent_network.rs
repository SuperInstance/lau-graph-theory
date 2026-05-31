//! Agent network analysis: topology, connectivity, influence propagation.

use crate::graph::Graph;
use crate::traversal;
use crate::shortest_path;
use crate::connectivity;
use crate::spectral;
use serde::{Deserialize, Serialize};

/// Summary statistics for an agent network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    pub num_agents: usize,
    pub num_connections: usize,
    pub avg_degree: f64,
    pub density: f64,
    pub num_components: usize,
    pub algebraic_connectivity: f64,
    pub avg_clustering_coefficient: f64,
}

/// Compute basic network statistics.
pub fn network_stats(g: &Graph) -> NetworkStats {
    let n = g.n;
    let m = g.edge_count();
    let avg_deg = if n > 0 { (2 * m) as f64 / n as f64 } else { 0.0 };
    let density = if n > 1 { (2 * m) as f64 / (n * (n - 1)) as f64 } else { 0.0 };
    let components = connectivity::connected_components(g);
    let ac = spectral::algebraic_connectivity(g);
    let clustering = avg_clustering_coefficient(g);

    NetworkStats {
        num_agents: n,
        num_connections: m,
        avg_degree: avg_deg,
        density,
        num_components: components.len(),
        algebraic_connectivity: ac,
        avg_clustering_coefficient: clustering,
    }
}

/// Compute the clustering coefficient of a single vertex.
pub fn clustering_coefficient(g: &Graph, v: usize) -> f64 {
    let neighbors: Vec<usize> = g.adj[v].iter().map(|&(u, _)| u).collect();
    let k = neighbors.len();
    if k < 2 { return 0.0; }
    let neighbor_set: std::collections::HashSet<usize> = neighbors.iter().copied().collect();
    let mut triangles = 0usize;
    for &u in &neighbors {
        for &(w, _) in &g.adj[u] {
            if neighbor_set.contains(&w) {
                triangles += 1;
            }
        }
    }
    // Each triangle counted twice (once from each endpoint)
    triangles as f64 / (k * (k - 1)) as f64
}

/// Average clustering coefficient over all vertices.
pub fn avg_clustering_coefficient(g: &Graph) -> f64 {
    if g.n == 0 { return 0.0; }
    let total: f64 = (0..g.n).map(|v| clustering_coefficient(g, v)).sum();
    total / g.n as f64
}

/// Identify the most influential agents using betweenness centrality (approximate).
/// Returns a sorted vector of (agent_index, centrality_score) in descending order.
pub fn influence_ranking(g: &Graph) -> Vec<(usize, f64)> {
    let n = g.n;
    let mut betweenness = vec![0.0f64; n];

    // For small graphs, compute exact betweenness using BFS from each node
    for s in 0..n {
        // BFS from s
        let mut stack = Vec::new();
        let mut sigma = vec![0usize; n];
        let mut dist = vec![usize::MAX; n];
        let mut pred: Vec<Vec<usize>> = vec![vec![]; n];
        sigma[s] = 1;
        dist[s] = 0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(s);
        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for &(w, _) in &g.adj[v] {
                if dist[w] == usize::MAX {
                    dist[w] = dist[v] + 1;
                    queue.push_back(w);
                }
                if dist[w] == dist[v] + 1 {
                    sigma[w] += sigma[v];
                    pred[w].push(v);
                }
            }
        }
        let mut delta = vec![0.0f64; n];
        while let Some(w) = stack.pop() {
            for &v in &pred[w] {
                delta[v] += (sigma[v] as f64 / sigma[w] as f64) * (1.0 + delta[w]);
            }
            if w != s {
                betweenness[w] += delta[w];
            }
        }
    }

    // Normalize (undirected: divide by 2)
    if !g.directed {
        for b in betweenness.iter_mut() {
            *b /= 2.0;
        }
    }

    let mut ranking: Vec<(usize, f64)> = betweenness.into_iter().enumerate().collect();
    ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranking
}

/// Simulate influence propagation using a simple SI model.
/// Starting from `seeds`, each timestep an infected node infects each neighbor
/// with probability `beta`. Returns number of infected nodes at each timestep
/// for `steps` timesteps.
pub fn simulate_influence(
    g: &Graph,
    seeds: &[usize],
    beta: f64,
    steps: usize,
) -> Vec<usize> {
    use rand::Rng;
    let mut rng = rand::rng();
    let mut infected = vec![false; g.n];
    for &s in seeds {
        infected[s] = true;
    }
    let mut result = vec![infected.iter().filter(|&&b| b).count()];

    for _ in 0..steps {
        let mut new_infected = infected.clone();
        for u in 0..g.n {
            if !infected[u] { continue; }
            for &(v, _) in &g.adj[u] {
                if !infected[v] && rng.random::<f64>() < beta {
                    new_infected[v] = true;
                }
            }
        }
        infected = new_infected;
        result.push(infected.iter().filter(|&&b| b).count());
    }
    result
}

/// Compute the diameter of the graph (longest shortest path).
pub fn diameter(g: &Graph) -> f64 {
    let (dist, _) = shortest_path::floyd_warshall(g);
    let mut max_dist = 0.0f64;
    for i in 0..g.n {
        for j in 0..g.n {
            if dist[i][j].is_finite() && dist[i][j] > max_dist {
                max_dist = dist[i][j];
            }
        }
    }
    max_dist
}

/// Find the center of the network (vertex with minimum eccentricity).
pub fn network_center(g: &Graph) -> (usize, f64) {
    let (dist, _) = shortest_path::floyd_warshall(g);
    let mut best_vertex = 0;
    let mut best_ecc = f64::INFINITY;
    for i in 0..g.n {
        let mut ecc = 0.0f64;
        for j in 0..g.n {
            if dist[i][j].is_finite() && dist[i][j] > ecc {
                ecc = dist[i][j];
            }
        }
        if ecc < best_ecc {
            best_ecc = ecc;
            best_vertex = i;
        }
    }
    (best_vertex, best_ecc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_stats() {
        let mut g = Graph::new(4);
        g.add_edge_unit(0, 1);
        g.add_edge_unit(1, 2);
        g.add_edge_unit(2, 3);
        g.add_edge_unit(3, 0);
        let stats = network_stats(&g);
        assert_eq!(stats.num_agents, 4);
        assert_eq!(stats.num_connections, 4);
        assert!((stats.avg_degree - 2.0).abs() < 1e-9);
        assert!(stats.density > 0.0);
        assert_eq!(stats.num_components, 1);
        assert!(stats.algebraic_connectivity > 0.0);
    }

    #[test]
    fn test_clustering_triangle() {
        let mut g = Graph::new(3);
        g.add_edge_unit(0, 1);
        g.add_edge_unit(1, 2);
        g.add_edge_unit(0, 2);
        let cc = clustering_coefficient(&g, 0);
        assert!((cc - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_clustering_line() {
        let mut g = Graph::new(3);
        g.add_edge_unit(0, 1);
        g.add_edge_unit(1, 2);
        let cc = clustering_coefficient(&g, 1);
        assert!(cc.abs() < 1e-9); // middle of path has 0 clustering
    }

    #[test]
    fn test_influence_ranking() {
        // Star graph: center is vertex 2
        let mut g = Graph::new(5);
        g.add_edge_unit(0, 2);
        g.add_edge_unit(1, 2);
        g.add_edge_unit(2, 3);
        g.add_edge_unit(2, 4);
        let ranking = influence_ranking(&g);
        assert_eq!(ranking[0].0, 2); // center has highest betweenness
    }

    #[test]
    fn test_diameter() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        let d = diameter(&g);
        assert!((d - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_network_center() {
        // Path: 0-1-2-3-4, center should be 2
        let mut g = Graph::new(5);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        g.add_edge(3, 4, 1.0);
        let (center, _) = network_center(&g);
        assert_eq!(center, 2);
    }

    #[test]
    fn test_disconnected_network() {
        let mut g = Graph::new(4);
        g.add_edge_unit(0, 1);
        // 2 and 3 isolated
        let stats = network_stats(&g);
        assert_eq!(stats.num_components, 3);
    }
}
