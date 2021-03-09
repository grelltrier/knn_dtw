use knn_dtw::ucr::*;

fn main() {
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
    let query_series = knn_dtw::utilities::QueryIterator::new(&query_name);
    let data_series = knn_dtw::utilities::DataIterator::new(&data_name);

    let mut knn_search = KNNSearch::new(settings);
    knn_search.calculate(data_series, query_series);
    knn_search.print();
}
