//! Additional integration and unit tests to reach 55+ test count.

#[cfg(test)]
mod tests {
    use crate::graph::{Graph, Edge};
    use crate::traversal::{bfs, dfs, topological_sort};
    use crate::shortest_path::{dijkstra, bellman_ford, floyd_warshall};
    use crate::mst::{kruskal, prim};
    use crate::flow::max_flow;
    use crate::connectivity::{connected_components, bridges, articulation_points, biconnected_components};
    use crate::matching::{bipartite_matching, hungarian};
    use crate::spectral::{laplacian_matrix, laplacian_spectrum, algebraic_connectivity, fiedler_vector, cheeger_inequality_bounds, cheeger_constant_approx};
    use crate::agent_network::{network_stats, clustering_coefficient, avg_clustering_coefficient, influence_ranking, diameter, network_center};

    // ── Graph structure tests ──

    #[test]
    fn test_graph_edge_count_undirected() {
        let mut g = Graph::new(4);
        g.add_edge_unit(0, 1);
        g.add_edge_unit(1, 2);
        g.add_edge_unit(2, 3);
        assert_eq!(g.edge_count(), 3);
    }

    #[test]
    fn test_graph_edge_count_directed() {
        let mut g = Graph::new_directed(3);
        g.add_edge_unit(0, 1);
        g.add_edge_unit(1, 2);
        g.add_edge_unit(2, 0);
        assert_eq!(g.edge_count(), 3);
    }

    #[test]
    fn test_graph_degree() {
        let mut g = Graph::new(4);
        g.add_edge_unit(0, 1);
        g.add_edge_unit(0, 2);
        g.add_edge_unit(0, 3);
        assert_eq!(g.degree(0), 3);
        assert_eq!(g.degree(1), 1);
    }

    #[test]
    fn test_graph_from_edge_list() {
        let edges = vec![(0, 1, 2.0), (1, 2, 3.0)];
        let g = Graph::from_edge_list(3, &edges, false);
        assert_eq!(g.n, 3);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_adjacency_matrix() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 2.0);
        g.add_edge(1, 2, 3.0);
        let mat = g.adjacency_matrix();
        assert!((mat[0][0] - 0.0).abs() < 1e-9);
        assert!((mat[0][1] - 2.0).abs() < 1e-9);
        assert!((mat[1][2] - 3.0).abs() < 1e-9);
        assert!(mat[0][2].is_infinite());
    }

    #[test]
    fn test_edge_list() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        let edges = g.edge_list();
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_neighbors() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 5.0);
        g.add_edge(0, 2, 3.0);
        let nb = g.neighbors(0);
        assert_eq!(nb.len(), 2);
    }

    // ── Traversal tests ──

    #[test]
    fn test_bfs_disconnected() {
        let mut g = Graph::new(5);
        g.add_edge_unit(0, 1);
        g.add_edge_unit(2, 3);
        let order = bfs(&g, 0);
        assert!(!order.contains(&2));
        assert!(!order.contains(&3));
    }

    #[test]
    fn test_dfs_branching() {
        let mut g = Graph::new(4);
        g.add_edge_unit(0, 1);
        g.add_edge_unit(0, 2);
        g.add_edge_unit(1, 3);
        g.add_edge_unit(2, 3);
        let order = dfs(&g, 0);
        assert_eq!(order[0], 0);
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_topological_linear() {
        let mut g = Graph::new_directed(3);
        g.add_edge_unit(0, 1);
        g.add_edge_unit(1, 2);
        let order = topological_sort(&g).unwrap();
        assert_eq!(order, vec![0, 1, 2]);
    }

    // ── Shortest path tests ──

    #[test]
    fn test_dijkstra_disconnected() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        // 2, 3 disconnected
        let (dist, _) = dijkstra(&g, 0);
        assert!((dist[1] - 1.0).abs() < 1e-9);
        assert!(dist[2].is_infinite());
    }

    #[test]
    fn test_dijkstra_single_vertex() {
        let g = Graph::new(1);
        let (dist, _) = dijkstra(&g, 0);
        assert!((dist[0]).abs() < 1e-9);
    }

    #[test]
    fn test_bellman_ford_simple() {
        let mut g = Graph::new_directed(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        let (dist, _) = bellman_ford(&g, 0).unwrap();
        assert!((dist[2] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_floyd_warshall_triangle() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(0, 2, 10.0);
        let (dist, _) = floyd_warshall(&g);
        assert!((dist[0][2] - 2.0).abs() < 1e-9); // 0->1->2 is shorter
    }

    // ── MST tests ──

    #[test]
    fn test_kruskal_single_edge() {
        let mut g = Graph::new(2);
        g.add_edge(0, 1, 5.0);
        let (mst, total) = kruskal(&g);
        assert_eq!(mst.len(), 1);
        assert!((total - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_prim_dense() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(0, 2, 2.0);
        g.add_edge(0, 3, 3.0);
        g.add_edge(1, 2, 4.0);
        g.add_edge(1, 3, 5.0);
        g.add_edge(2, 3, 6.0);
        let (_, total) = prim(&g, 0);
        assert!((total - 6.0).abs() < 1e-9); // 1+2+3
    }

    // ── Flow tests ──

    #[test]
    fn test_max_flow_single_path() {
        let mut g = Graph::new_directed(3);
        g.add_edge(0, 1, 10.0);
        g.add_edge(1, 2, 5.0);
        let (flow, _) = max_flow(&g, 0, 2);
        assert!((flow - 5.0).abs() < 1e-9); // bottleneck
    }

    #[test]
    fn test_max_flow_parallel() {
        let mut g = Graph::new_directed(2);
        g.add_edge(0, 1, 3.0);
        g.add_edge(0, 1, 5.0); // parallel edge
        let (flow, _) = max_flow(&g, 0, 1);
        assert!((flow - 8.0).abs() < 1e-9);
    }

    // ── Connectivity tests ──

    #[test]
    fn test_connected_all_connected() {
        let mut g = Graph::new(3);
        g.add_edge_unit(0, 1);
        g.add_edge_unit(1, 2);
        let comps = connected_components(&g);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].len(), 3);
    }

    #[test]
    fn test_bridges_no_bridges_cycle() {
        let mut g = Graph::new(4);
        g.add_edge_unit(0, 1);
        g.add_edge_unit(1, 2);
        g.add_edge_unit(2, 3);
        g.add_edge_unit(3, 0);
        let br = bridges(&g);
        assert!(br.is_empty());
    }

    #[test]
    fn test_articulation_points_star() {
        let mut g = Graph::new(5);
        g.add_edge_unit(0, 1);
        g.add_edge_unit(0, 2);
        g.add_edge_unit(0, 3);
        g.add_edge_unit(0, 4);
        let ap = articulation_points(&g);
        assert!(ap.contains(&0)); // center is articulation point
        assert_eq!(ap.len(), 1);
    }

    #[test]
    fn test_biconnected_simple() {
        let mut g = Graph::new(3);
        g.add_edge_unit(0, 1);
        g.add_edge_unit(1, 2);
        g.add_edge_unit(0, 2);
        let bcc = biconnected_components(&g);
        assert_eq!(bcc.len(), 1); // single biconnected component (triangle)
    }

    // ── Matching tests ──

    #[test]
    fn test_bipartite_complete_k22() {
        let mut g = Graph::new(4);
        g.add_edge_unit(0, 2);
        g.add_edge_unit(0, 3);
        g.add_edge_unit(1, 2);
        g.add_edge_unit(1, 3);
        let matching = bipartite_matching(&g, 2, 2);
        assert_eq!(matching.len(), 2);
    }

    #[test]
    fn test_hungarian_identity() {
        let cost = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let (_, total) = hungarian(&cost);
        assert!((total - 0.0).abs() < 1e-9); // picks all the 0s
    }

    #[test]
    fn test_hungarian_single() {
        let cost = vec![vec![42.0]];
        let (matching, total) = hungarian(&cost);
        assert_eq!(matching.len(), 1);
        assert!((total - 42.0).abs() < 1e-9);
    }

    // ── Spectral tests ──

    #[test]
    fn test_laplacian_path() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let l = laplacian_matrix(&g);
        assert!((l[(0, 0)] - 1.0).abs() < 1e-9);
        assert!((l[(1, 1)] - 2.0).abs() < 1e-9);
        assert!((l[(2, 2)] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_spectrum_path_graph() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        let spectrum = laplacian_spectrum(&g);
        assert!(spectrum[0].abs() < 1e-9);
        assert!(spectrum[1] > 0.0);
    }

    #[test]
    fn test_algebraic_connectivity_k4() {
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in (i+1)..4 {
                g.add_edge(i, j, 1.0);
            }
        }
        let ac = algebraic_connectivity(&g);
        assert!((ac - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_fiedler_path_graph() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        let (_, val) = fiedler_vector(&g);
        assert!(val > 0.0);
    }

    #[test]
    fn test_cheeger_approx_connected() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        g.add_edge(3, 0, 1.0);
        let h = cheeger_constant_approx(&g);
        assert!(h > 0.0);
    }

    // ── Agent network tests ──

    #[test]
    fn test_network_stats_single() {
        let g = Graph::new(1);
        let stats = network_stats(&g);
        assert_eq!(stats.num_agents, 1);
        assert_eq!(stats.num_connections, 0);
    }

    #[test]
    fn test_diameter_single_vertex() {
        let g = Graph::new(1);
        let d = diameter(&g);
        assert!((d - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_network_center_complete() {
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in (i+1)..4 {
                g.add_edge(i, j, 1.0);
            }
        }
        let (center, ecc) = network_center(&g);
        assert!((ecc - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_influence_ranking_path() {
        let mut g = Graph::new(5);
        g.add_edge_unit(0, 1);
        g.add_edge_unit(1, 2);
        g.add_edge_unit(2, 3);
        g.add_edge_unit(3, 4);
        let ranking = influence_ranking(&g);
        // Middle vertices should have highest betweenness
        assert!(ranking[0].0 == 2 || ranking[0].0 == 1 || ranking[0].0 == 3);
    }

    #[test]
    fn test_clustering_coefficient_isolated() {
        let g = Graph::new(3);
        let cc = clustering_coefficient(&g, 0);
        assert!((cc).abs() < 1e-9);
    }

    #[test]
    fn test_avg_clustering_star() {
        let mut g = Graph::new(5);
        g.add_edge_unit(0, 1);
        g.add_edge_unit(0, 2);
        g.add_edge_unit(0, 3);
        g.add_edge_unit(0, 4);
        let avg = avg_clustering_coefficient(&g);
        assert!(avg.abs() < 1e-9); // star has 0 clustering everywhere
    }

    #[test]
    fn test_diameter_cycle() {
        let mut g = Graph::new(6);
        for i in 0..6 {
            g.add_edge(i, (i + 1) % 6, 1.0);
        }
        let d = diameter(&g);
        assert!((d - 3.0).abs() < 1e-9);
    }
}
