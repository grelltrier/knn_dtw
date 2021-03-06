use std::time::Duration;

use super::*;

#[test]
// Test case F1
// Use Data.txt and Query2.txt to find the 1-nn with a warping window of 0.1
fn knn_search() {
    // Input parameters
    let query_name = "Query2.txt";
    let data_name = "Data.txt";
    //let settings = Settings::default();
    let settings = Settings::new(
        1,                        // k nearest Neighbors
        true,                     // sort:
        true,                     // normalize:
        dtw_cost::sq_l2_dist_f64, // cost_fn
        0.10,                     // window_rate:
        100000,                   // epoch:
    );
    println!("{}", settings);
    // Create a reader/iterator to get the data from
    let query_series = crate::utilities::QueryIterator::new(&query_name);
    let data_series = crate::utilities::DataIterator::new(&data_name);

    let mut knn_search = KNNSearch::new(settings);
    knn_search.calculate(data_series, query_series);
    //knn_search.print();

    let mut k_best = Vec::new();
    k_best.push((430264, 14.36940019736433));
    let correct_stats = Some(SearchStats {
        kim: 226827,
        keogh: 409926,
        keogh2: 318994,
    });
    let correct_result = SearchResult {
        k_best,
        duration: Duration::new(3, 0),
        i: 1000000,
        supplemental_stats: correct_stats,
    };
    println!();
    println!("Correct result:");
    correct_result.print();
    println!();
    correct_stats.unwrap().print(correct_result.i);
    println!();
    println!();
    println!("knn_result result:");
    knn_search.print();
    println!();
    println!();
    assert!(correct_result == knn_search.result.unwrap());
}

#[test]
// Test case F2
// Test the insert_into_k_bsf function with various input
fn insert_k_bsf() {
    // Intialize variables
    let k = 1;
    let mut test_k_best = vec![(0, f64::INFINITY); k]; // [(0, f64::INFINITY)]
    let mut correct_k_best = test_k_best.clone(); // [(0, f64::INFINITY)]
    let mut new_loc;
    let mut new_bsf;

    // Better than all others
    new_loc = 1;
    new_bsf = 0.4;
    insert_into_k_bsf((new_loc, new_bsf), &mut test_k_best);
    correct_k_best.insert(0, (new_loc, new_bsf));
    correct_k_best.pop();
    // [(1, 0.4)] ==[(1, 0.4)]
    assert!(correct_k_best == test_k_best);

    // Better than all others
    new_loc = 2;
    new_bsf = 0.3;
    insert_into_k_bsf((new_loc, new_bsf), &mut test_k_best);
    correct_k_best.insert(0, (new_loc, new_bsf));
    correct_k_best.pop();
    // [(2, 0.3)] ==[(2, 0.3)]
    assert!(correct_k_best == test_k_best);

    // Keep TWO best values
    // Initialize again
    let k = 2;
    let mut test_k_best = vec![(0, f64::INFINITY); k]; // [(0, f64::INFINITY)]
    let mut correct_k_best = test_k_best.clone(); // [(0, f64::INFINITY)]
    let mut new_loc;
    let mut new_bsf;

    // Better than all others
    new_loc = 1;
    new_bsf = 0.9;
    insert_into_k_bsf((new_loc, new_bsf), &mut test_k_best);
    correct_k_best.insert(0, (new_loc, new_bsf));
    correct_k_best.pop();
    // [(1, 0.9),(0, f64::INFINITY)] ==[(1, 0.9),(0, f64::INFINITY)]
    assert!(correct_k_best == test_k_best);

    // Better than all others
    new_loc = 2;
    new_bsf = 0.3;
    insert_into_k_bsf((new_loc, new_bsf), &mut test_k_best);
    correct_k_best.insert(0, (new_loc, new_bsf));
    correct_k_best.pop();
    // [(2, 0.3),(1, 0.9)] ==[(2, 0.3),(1, 0.9)]
    assert!(correct_k_best == test_k_best);

    // Better than last value
    new_loc = 1;
    new_bsf = 0.5;
    insert_into_k_bsf((new_loc, new_bsf), &mut test_k_best);
    correct_k_best.insert(1, (new_loc, new_bsf));
    correct_k_best.pop();
    // [(2, 0.3),(1, 0.5)] ==[(2, 0.3),(1, 0.5)]
    assert!(correct_k_best == test_k_best);
}

#[test]
// Test case F3
// Test the lb_kim_hierarchy function with various sequences
fn lb_kim_hierarchy_test() {
    let t1 = [
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
    ];

    let q1 = [1., 2., 3., 4., 5., 4., 3., 2., 1.];
    let q2 = [3., 4., 5., 4., 5., 4., 3., 2., 1.];
    let j = 0;
    let mean = 0.0;
    let std = 1.0;
    let bsf1 = 5.0;
    let bsf2 = 100.0;
    let bsf3 = 120.0;
    let cost_fn = dtw_cost::sq_l2_dist_f64;

    // q1
    let lb = lb_kim_hierarchy(&t1, &q1, j, mean, std, bsf1, &cost_fn);
    assert!((lb - 64.0).abs() < 0.0000000001);
    let lb = lb_kim_hierarchy(&t1, &q1, j, mean, std, bsf2, &cost_fn);
    assert!((lb - 100.0).abs() < 0.0000000001);
    let lb = lb_kim_hierarchy(&t1, &q1, j, mean, std, bsf3, &cost_fn);
    assert!((lb - 116.0).abs() < 0.0000000001);

    // q2
    let lb = lb_kim_hierarchy(&t1, &q2, j, mean, std, bsf1, &cost_fn);
    assert!((lb - 68.0).abs() < 0.0000000001);
    let lb = lb_kim_hierarchy(&t1, &q2, j, mean, std, bsf2, &cost_fn);
    assert!((lb - 105.0).abs() < 0.0000000001);
    let lb = lb_kim_hierarchy(&t1, &q2, j, mean, std, bsf3, &cost_fn);
    assert!((lb - 121.0).abs() < 0.0000000001);

    // Test steps
    // LB > bsf after checking the first point
    // dist > bsf
    let lb = lb_kim_hierarchy(
        &[
            2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
        &[1., 0., 0., 0., 0., 0., 0., 0.],
        j,
        mean,
        std,
        0.5,
        &cost_fn,
    );
    assert!((lb - 1.0).abs() < 0.0000000001);

    // LB > bsf after checking the second point
    // dist > bsf
    let lb = lb_kim_hierarchy(
        &[
            0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
        &[0., 1., 0., 0., 0., 0., 0., 0.],
        j,
        mean,
        std,
        0.5,
        &cost_fn,
    );
    assert!((lb - 1.0).abs() < 0.0000000001);

    // LB > bsf after checking the third point
    // dist > bsf
    let lb = lb_kim_hierarchy(
        &[
            0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
        &[0., 0., 1., 0., 0., 0., 0., 0.],
        j,
        mean,
        std,
        0.5,
        &cost_fn,
    );
    assert!((lb - 1.0).abs() < 0.0000000001);

    // The fourth point is never checked so lb=0 < bsf
    let lb = lb_kim_hierarchy(
        &[
            0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
        &[0., 0., 0., 1., 0., 0., 0., 0.],
        j,
        mean,
        std,
        0.5,
        &cost_fn,
    );
    assert!((lb - 0.0).abs() < 0.0000000001);

    // Distance at the end is greater than bsf

    // LB > bsf after checking the first point from the back
    // dist > bsf
    let lb = lb_kim_hierarchy(
        &[
            0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
        &[0., 0., 0., 0., 0., 0., 0., 1.],
        j,
        mean,
        std,
        0.5,
        &cost_fn,
    );
    assert!((lb - 1.0).abs() < 0.0000000001);

    // LB > bsf after checking the second point from the back
    // dist > bsf
    let lb = lb_kim_hierarchy(
        &[
            0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
        &[0., 0., 0., 0., 0., 0., 1., 0.],
        j,
        mean,
        std,
        0.5,
        &cost_fn,
    );
    assert!((lb - 1.0).abs() < 0.0000000001);

    // LB > bsf after checking the third point from the back
    // dist > bsf
    let lb = lb_kim_hierarchy(
        &[
            0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
        &[0., 0., 0., 0., 0., 1., 0., 0.],
        j,
        mean,
        std,
        0.5,
        &cost_fn,
    );
    assert!((lb - 1.0).abs() < 0.0000000001);

    // The fourth point from the back is never checked so lb=0 < bsf
    let lb = lb_kim_hierarchy(
        &[
            0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
        &[0., 0., 0., 0., 1., 0., 0., 0.],
        j,
        mean,
        std,
        0.5,
        &cost_fn,
    );
    assert!((lb - 0.0).abs() < 0.0000000001);
}

#[test]
// Test case F4
//Test the upper_lower_lemire function with various sequences and warping windows
fn upper_lower_lemire_tests() {
    let time_series = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];

    // w = 0
    let w = 0;
    let (lower, upper) = upper_lower_lemire(&time_series, w);
    let correct_lower = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];
    let correct_upper = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];

    for i in 0..correct_lower.len() {
        assert!((correct_lower[i] - lower[i]).abs() < 0.0000000001);
        assert!((correct_upper[i] - upper[i]).abs() < 0.0000000001);
    }

    // w = 3
    let w = 3;
    let (lower, upper) = upper_lower_lemire(&time_series, w);
    let correct_lower = [0., 0., 0., 0., 1., 2., 3., 4., 5., 6.];
    let correct_upper = [3., 4., 5., 6., 7., 8., 9., 9., 9., 9.];

    for i in 0..correct_lower.len() {
        assert!((correct_lower[i] - lower[i]).abs() < 0.0000000001);
        assert!((correct_upper[i] - upper[i]).abs() < 0.0000000001);
    }

    // Different time series
    let time_series = [0., 1., 2., 3., -4., -5., -6., -7., -8., -9.];
    let w = 3;
    let (lower, upper) = upper_lower_lemire(&time_series, w);
    let correct_lower = [0., -4., -5., -6., -7., -8., -9., -9., -9., -9.];
    let correct_upper = [3., 3., 3., 3., 3., 3., 3., -4., -5., -6.];

    for i in 0..correct_lower.len() {
        assert!((correct_lower[i] - lower[i]).abs() < 0.0000000001);
        assert!((correct_upper[i] - upper[i]).abs() < 0.0000000001);
    }

    // Different time series (Decreasing from first to second point)
    let time_series = [10., 1., 2., 3., -4., -5., -6., -7., -8., -9.];
    let w = 3;
    let (lower, upper) = upper_lower_lemire(&time_series, w);
    let correct_lower = [1., -4., -5., -6., -7., -8., -9., -9., -9., -9.];
    let correct_upper = [10., 10., 10., 10., 3., 3., 3., -4., -5., -6.];

    for i in 0..correct_lower.len() {
        assert!((correct_lower[i] - lower[i]).abs() < 0.0000000001);
        assert!((correct_upper[i] - upper[i]).abs() < 0.0000000001);
    }
}

#[test]
#[should_panic]
// Test case F5
// Test the upper_lower_lemire function with a warping window
// greater than the length of the sequence
fn upper_lower_lemire_w_greater_than_time_series_len_tests() {
    // w > time_series.len()
    let time_series = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];
    let w = 20;
    let (lower, upper) = upper_lower_lemire(&time_series, w);
    let correct_lower = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.];
    let correct_upper = [9., 9., 9., 9., 9., 9., 9., 9., 9., 9.];

    for i in 0..correct_lower.len() {
        assert!((correct_lower[i] - lower[i]).abs() < 0.0000000001);
        assert!((correct_upper[i] - upper[i]).abs() < 0.0000000001);
    }
}
