// use ndarray::prelude::*;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::ops::{Div, Sub};
use std::time::{Duration, Instant};

#[derive(Copy, Clone)]
pub struct Settings<F>
where
    F: Fn(&f64, &f64) -> f64,
{
    k_nearest: usize, // k Nearest Neighbors
    jump: bool,
    sort: bool,
    normalize: bool,
    cost_fn: F,
    window_rate: f64,
    epoch: usize,
}
impl<F> Settings<F>
where
    F: Fn(&f64, &f64) -> f64,
{
    pub fn new(
        k_nearest: usize,
        jump: bool,
        sort: bool,
        normalize: bool,
        cost_fn: F,
        window_rate: f64,
        epoch: usize,
    ) -> Self {
        Settings {
            k_nearest,
            jump,
            sort,
            normalize,
            cost_fn,
            window_rate,
            epoch,
        }
    }
}

impl Default for Settings<for<'r, 's> fn(&'r f64, &'s f64) -> f64> {
    fn default() -> Self {
        Settings {
            k_nearest: 1,
            jump: true,
            sort: true,
            normalize: true,
            cost_fn: dtw_cost::sq_l2_dist_f64,
            window_rate: 0.1,
            epoch: 100000,
        }
    }
}

impl<F> std::fmt::Display for Settings<F>
where
    F: Fn(&f64, &f64) -> f64,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let _ = writeln!(f, "Settings:");
        let _ = writeln!(f, "  k_nearest   : {}", self.k_nearest);
        let _ = writeln!(f, "  jump        : {}", self.jump);
        let _ = writeln!(f, "  sort        : {}", self.sort);
        let _ = writeln!(f, "  normalize   : {}", self.normalize);
        let _ = writeln!(f, "  window_rate : {}", self.window_rate);
        writeln!(f, "  epoch       : {}", self.epoch)
    }
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub k_best: Vec<(usize, f64)>,
    pub duration: Duration,
    pub i: usize,
    pub supplemental_stats: Option<SearchStats>,
}

impl SearchResult {
    pub fn print(&self) {
        println!("Calculation result:");
        println!("Sequences scanned    : {}", self.i);
        println!("Total Execution Time : {:?}", self.duration);
        println!("Best {}-results:", self.k_best.len());
        println!("  No: (Location, Distance)");
        for (idx, (loc, dist)) in self.k_best.iter().enumerate() {
            println!("   {}: ({:08}, {:.9})", idx, loc, dist)
        }
        println!();
    }
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        if self.i != other.i {
            return false;
        }
        if self.supplemental_stats != other.supplemental_stats {
            return false;
        }
        if self.k_best.len() != other.k_best.len() {
            return false;
        }
        for idx in 0..self.k_best.len() {
            if self.k_best[idx].0 != other.k_best[idx].0
                || (self.k_best[idx].1 - other.k_best[idx].1).abs() > 0.00000001
            {
                return false;
            }
        }
        true
    }
}
impl Eq for SearchResult {}

// TODO: Check if type of fields should not be usize instead??
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SearchStats {
    pub jump_times: usize,
    pub kim: usize,
    pub keogh: usize,
    pub keogh2: usize,
}

impl SearchStats {
    pub fn print(&self, i: usize) {
        println!("Stats:");
        println!(
            "jump_times: {}, kim: {}, keogh: {}, keogh2: {}",
            self.jump_times, self.kim, self.keogh, self.keogh2
        );
        let i = i as f64;
        let jump_times = self.jump_times as f64;
        let kim = self.kim as f64;
        let keogh = self.keogh as f64;
        let keogh2 = self.keogh2 as f64;

        println!(
            "  Pruned by Jump      : {:>7.4} %",
            (jump_times / i) * 100.0
        );
        println!("  Pruned by LB_Kim    : {:>7.4} %", (kim / i) * 100.0);
        println!("  Pruned by LB_Keogh  : {:>7.4} %", (keogh / i) * 100.0);
        println!("  Pruned by LB_Keogh2 : {:>7.4} %", (keogh2 / i) * 100.0);
        println!(
            "  DTW Calculation     : {:>7.4} %",
            100.0 - ((jump_times + kim + keogh + keogh2) / i * 100.0)
        );
    }
}

/// Inserts the new found bsf into the Vec of the best matches
/// The Vec must already have the length k
/// Input:
///  new_bsf   :  Tuple of the index of the found match and its DTW distance
///  old_k_bsf :  A Vec of the tuples for the current k-best matches
pub fn insert_into_k_bsf<T>(new_bsf: (T, f64), old_k_bsf: &mut Vec<(T, f64)>) {
    let mut new_position = old_k_bsf.len() - 1; // Stores the position at wich the new_bsf needs to be inserted

    // Add an index to the k best distances, then go trough them from worst to best to find out where the new value needs to be inserted
    // We can skip the worst one because we call this function only if we found a better match
    for (i, value) in old_k_bsf.iter().enumerate().rev().skip(1) {
        // If the new value is not better than a value in the old_k_best vec, all following values will also be better than the new value so we found the correct index for the new value
        if new_bsf.1 > value.1 {
            break;
        }
        // If we reach this part, the new value was better so we store the index of the value that was worst
        new_position = i;
    }
    // Once we found the correct index for the new value, we insert it at that index
    old_k_bsf.insert(new_position, new_bsf);
    old_k_bsf.pop(); // We now have too many values so we remove the last one
}

// Input are two sequences of the same length
// The 'end' of the sequence is denoted as the first element of the tuples of the sequences
// The function calculates the cost/distance between the 'end' of the sequences and all other values from the other sequence. The minimal distance is returned
fn min_dist<T, F>(seq_a: (&T, &[T]), seq_b: (&T, &[T]), cost_fn: &F) -> f64
where
    T: Div<Output = T> + Sub<Output = T> + Copy, // TODO: Should this Copy be added?? Compiler error if not added
    F: Fn(&T, &T) -> f64,
{
    // Compare the two 'ends'
    let mut lowest_dist = cost_fn(seq_a.0, seq_b.0);
    // If the sequences not only consist of their 'ends'...
    if !seq_a.1.is_empty() {
        // Variable to be able to iterate over the sequences
        let sequences = [seq_a, seq_b];
        // ..do the following calculation for both sequences:
        for (no, sequence) in sequences.iter().enumerate() {
            // Take each value that is not the 'end' of the sequence
            for value in sequence.1.iter() {
                // .. and calculate the distance between that value and the end of the other sequence
                // .. if the distance is lower then the currently lowest distance, set it to the new value
                lowest_dist = f64::min(lowest_dist, cost_fn(value, sequences[1 - no].0));
            }
        }
    }
    lowest_dist
}

#[derive(Clone)]
pub struct KNNSearch<F>
where
    F: Fn(&f64, &f64) -> f64,
{
    pub result: Option<SearchResult>,
    settings: Settings<F>,
}

impl<F> KNNSearch<F>
where
    F: Fn(&f64, &f64) -> f64 + Copy,
{
    pub fn new(settings: Settings<F>) -> Self {
        Self {
            result: None,
            settings,
        }
    }
    pub fn print(&self) {
        if let Some(result) = &self.result {
            result.print();
            // Print additional stats about pruning
            if let Some(stats) = result.supplemental_stats {
                stats.print(result.i);
            }
        }
    }

    pub fn calculate<DI, QI>(&mut self, mut data_series: DI, query_series: QI)
    where
        DI: std::iter::Iterator<Item = f64>,
        QI: std::iter::Iterator<Item = f64>,
    {
        let Settings {
            k_nearest,
            window_rate,
            sort,
            normalize,
            cost_fn,
            jump,
            epoch,
        } = self.settings;

        let mut k_best = vec![(0, f64::INFINITY); k_nearest]; // Stores the k nearest neighbors (location, DTW distance)
        let mut bsf = k_best[k_nearest - 1].1;
        let mut loc; // Temporarily stores location of best match

        let mut query: Vec<f64> = Vec::new();
        let (mut jump_times, mut kim, mut keogh, mut keogh2) = (0, 0, 0, 0);
        let (mut ex, mut ex2) = (0.0, 0.0);

        // start the clock
        let time_start = Instant::now();

        // Read all lines of query
        for observation in query_series.into_iter() {
            ex += observation;
            ex2 += observation.powi(2);
            query.push(observation);
        }

        // Calculate the mean and std of the query
        let mut mean = ex / query.len() as f64;
        let mut std = f64::sqrt((ex2 / query.len() as f64) - mean.powi(2));

        if normalize {
            // z-normalize the query
            query = query
                .iter_mut()
                .map(|entry| (*entry - mean) / std)
                .collect();
        }

        // TODO: This is done differently in C implementation, double check it
        // sakoe_chiba_band  : size of Sakoe-Chiba warpping band
        let sakoe_chiba_band = if window_rate <= 1.0 {
            (window_rate * query.len() as f64).floor() as usize
        } else {
            window_rate.floor() as usize
        };

        // Create envelope of the query
        let (mut lower_envelop, mut upper_envelop) = upper_lower_lemire(&query, sakoe_chiba_band);

        // The boundary constraint demands that the first and last points have to match so the upper and lower bounds at those indices are the value of the query
        lower_envelop[0] = query[0];
        upper_envelop[0] = query[0];
        lower_envelop[query.len() - 1] = query[query.len() - 1];
        upper_envelop[query.len() - 1] = query[query.len() - 1];

        // Add the index to each query entry
        let mut indexed_query: Vec<(usize, f64)> = Vec::new();
        for (idx, query_entry) in query.iter().enumerate() {
            indexed_query.push((idx, *query_entry));
        }

        // Create more arrays for keeping the (sorted) envelop
        let mut order: Vec<usize> = Vec::new();
        let mut qo: Vec<f64> = Vec::new();
        let mut uo: Vec<f64> = Vec::new();
        let mut lo: Vec<f64> = Vec::new();

        if sort {
            indexed_query.sort_by(|a, b| {
                (b.1.abs())
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(Ordering::Equal)
            });

            indexed_query.iter().for_each(|(idx, _)| {
                order.push(*idx);
                qo.push(query[*idx]);
                uo.push(upper_envelop[*idx]);
                lo.push(lower_envelop[*idx]);
            })
        } else {
            for i in 0..query.len() {
                order.push(i);
            }
            qo = query.clone();
            uo = upper_envelop;
            lo = lower_envelop;
        }

        // Initialize the cummulative lower bound
        let mut cb = vec![0.0; query.len()];

        let mut j; // j: the starting index of the data in the circular array t
        let mut done = false;
        let mut it = 0;
        let mut ep = 0;

        let mut buffer: Vec<f64> = vec![0.0; epoch];
        let mut t: Vec<f64> = vec![0.0; query.len() * 2];
        let mut tz: Vec<f64> = Vec::new(); // z-normalized candidate sequence
        tz.reserve(query.len());

        while !done {
            // Read the first m-1 points from the data sequence
            if it == 0 {
                for k in 0..(query.len() - 1) {
                    if let Some(data) = data_series.next() {
                        buffer[k] = data;
                    }
                }
            } else {
                for k in 0..(query.len() - 1) {
                    buffer[k] = buffer[epoch - query.len() + 1 + k];
                }
            }

            // Read buffer of size EPOCH or when all data has been read.
            ep = query.len() - 1;
            while ep < epoch {
                if let Some(data) = data_series.next() {
                    buffer[ep] = data;
                    ep += 1;
                } else {
                    break;
                }
            }

            if ep < query.len() {
                done = true;
            } else {
                let (l_buff, u_buff) = upper_lower_lemire(&buffer[..ep], sakoe_chiba_band);

                // Just for printing a dot for approximate a million point. Not much accurate.
                if it % (1000000 / (epoch - query.len() + 1)) == 0 {
                    print!(".");
                }

                ex = 0.0;
                ex2 = 0.0;
                let mut jump_size: usize = 0;

                // Do main task here..
                for i in 0..ep {
                    // A bunch of data has been read and pick one of them at a time to use
                    let data = buffer[i];

                    // Calculate sum and sum square
                    ex += data;
                    ex2 += data.powi(2);

                    // t is a circular array for keeping current data
                    t[i % query.len()] = data;

                    // Double the size for avoiding using modulo "%" operator
                    t[(i % query.len()) + query.len()] = data;

                    jump_size = jump_size.saturating_sub(1);

                    // Start the task when there are more than m-1 points in the current chunk
                    if i >= query.len() - 1 {
                        // compute the start location of the data in the current circular array, t
                        j = (i + 1) % query.len();

                        if !jump || jump_size == 0 {
                            mean = ex / query.len() as f64;
                            std = f64::sqrt((ex2 / query.len() as f64) - mean.powi(2));

                            // the start location of the data in the current chunk
                            let i_cap = i - (query.len() - 1);

                            // Use a constant lower bound to prune the obvious subsequence
                            let (lb_kim, jump_size_tmp) =
                                lb_kim_hierarchy(&t, &query, j, mean, std, bsf, &cost_fn);
                            jump_size = jump_size_tmp;

                            //////
                            if lb_kim < bsf {
                                let (lb_keogh_query, keogh_diffs, jump_tmp) = lb_keogh_cumulative(
                                    &order, &t, &uo, &lo, None, j, mean, std, bsf, false, &cost_fn,
                                );
                                jump_size = jump_tmp;

                                if lb_keogh_query < bsf {
                                    let (lb_keogh_sum, keogh_diffs, jump_tmp) = lb_keogh_cumulative(
                                        &order,
                                        &qo,
                                        &u_buff,
                                        &l_buff,
                                        Some(keogh_diffs),
                                        i_cap,
                                        mean,
                                        std,
                                        bsf,
                                        true,
                                        &cost_fn,
                                    );
                                    jump_size = jump_tmp;

                                    if lb_keogh_sum < bsf {
                                        {
                                            // Cumulativly sum the keogh diffs from the back
                                            // The value at index 0 is always ignored so we don't bother calculating it correctly
                                            for k in (1..query.len() - 2).rev() {
                                                cb[k] = keogh_diffs[k] + cb[k + 1];
                                            }

                                            // Take another linear time to compute z_normalization of t.
                                            // Note that for better optimization, this can merge to the previous function.
                                            if normalize {
                                                tz = t
                                                    .iter_mut()
                                                    .skip(j)
                                                    .take(query.len())
                                                    .map(|entry| (*entry - mean) / std)
                                                    .collect();
                                            }
                                            let dist = dtw::ucr_improved::dtw(
                                                &tz,
                                                &query,
                                                Some(&cb),
                                                sakoe_chiba_band,
                                                bsf,
                                                &cost_fn,
                                            );

                                            if dist < bsf {
                                                // loc is the real starting location of the nearest neighbor in the file
                                                loc = it * (epoch - query.len() + 1) + i + 1
                                                    - query.len();
                                                // Update bsf
                                                insert_into_k_bsf((loc, dist), &mut k_best);
                                                bsf = k_best[k_nearest - 1].1;
                                            }
                                        }
                                    } else {
                                        keogh2 += 1;
                                    }
                                } else {
                                    keogh += 1;
                                }
                            } else {
                                kim += 1;
                            }
                        } else {
                            jump_times += 1
                        }
                        // Reduce obsolute points from sum and sum square
                        ex -= t[j];
                        ex2 -= t[j].powi(2);
                    }
                }

                // If the size of last chunk is less then EPOCH, then no more data and terminate.
                if ep < epoch {
                    done = true;
                } else {
                    it += 1;
                }
            }
        }

        let time_end = Instant::now();
        let duration = time_end.saturating_duration_since(time_start);
        let i = it * (epoch - query.len() + 1) + ep;
        //let kim = kim as f64;
        //let keogh = keogh as f64;
        //let keogh2 = keogh2 as f64;

        let supplemental_stats = Some(SearchStats {
            jump_times,
            kim,
            keogh,
            keogh2,
        });
        self.result = Some(SearchResult {
            k_best,
            duration,
            i,
            supplemental_stats,
        });
    }
}

// Returns the envelope to calculate the LB_Keogh
// This is an implementation of the algorithm proposed in "Faster retrieval with a two-pass dynamic-time-warping lower bound" by Daniel Lemire
// (https://doi.org/10.1016/j.patcog.2008.11.030)
// Inputs:
// time_series : Time series to calculate the envelope for
// w           : Size of the warping constraint (Sakoe-Chiba band)
//               MUST ensure that w < time_series.len() -> otherwise the method panics
pub fn upper_lower_lemire(time_series: &[f64], w: usize) -> (Vec<f64>, Vec<f64>) {
    let len = time_series.len();
    let mut upper = vec![0.0; len];
    let mut lower = upper.clone();
    let mut du: VecDeque<usize> = VecDeque::with_capacity(2 * w + 2);
    let mut dl: VecDeque<usize> = VecDeque::with_capacity(2 * w + 2);

    du.push_back(0);
    dl.push_back(0);
    for i in 1..len {
        if i > w {
            upper[i - w - 1] = time_series[*du.front().unwrap()];
            lower[i - w - 1] = time_series[*dl.front().unwrap()];
        }

        // Pop out the bound that is not maximium or minimum
        // Store the max upper bound and min lower bound within window r
        if time_series[i] > time_series[i - 1] {
            du.pop_back();
            while !du.is_empty() && time_series[i] > time_series[*du.back().unwrap()] {
                du.pop_back();
            }
        } else {
            dl.pop_back();
            while !dl.is_empty() && time_series[i] < time_series[*dl.back().unwrap()] {
                dl.pop_back();
            }
        }
        du.push_back(i);
        dl.push_back(i);

        // Pop out the bound that os out of window r.
        if i == 2 * w + 1 + du.front().unwrap() {
            du.pop_front();
        } else if i == 2 * w + 1 + dl.front().unwrap() {
            dl.pop_front();
        }
    }

    // The envelop of first r points are from r+1 .. r+r, so the last r points' envelop haven't settle down yet.
    for i in len..(len + w + 1) {
        upper[i - w - 1] = time_series[*du.front().unwrap()];
        lower[i - w - 1] = time_series[*dl.front().unwrap()];
        if i - du.front().unwrap() >= 2 * w + 1 {
            du.pop_front();
        }
        if i - dl.front().unwrap() >= 2 * w + 1 {
            dl.pop_front();
        }
    }
    (lower, upper)
}

// TODO: Double check comments for correctness! Since adding support for sequences of different lengths, they most likely have errors
// Calculate the lower bound according to Kim
// The paper "An index-based approach for similarity search supporting time warping in large sequence databases,"
// (https://doi.org/10.1109/ICDE.2001.914875) elaborates on this lower bound
// The time complexity to calculate it is O(1)
// Improvements over the UCR suite ONLY for the case of subsequence search:
//    If the minimal cost between observations is bigger then the bsf, these observations can not be included in the best matching subsequence
//    and all subsequences including these values can be skipped. We can only do so when we look at the observations at the front though,
//    because the observations could be matched with later observations in the next subsequence that could yield a better match.
pub fn lb_kim_hierarchy<F>(
    t: &[f64],
    q: &[f64],
    j: usize,
    mean: f64,
    std: f64,
    bsf: f64,
    cost_fn: &F,
) -> (f64, usize)
where
    F: Fn(&f64, &f64) -> f64 + Copy,
    //where
    //T: Div<Output = T> + Sub<Output = T> + Copy, // TODO: Should this Copy be added?? Compiler error if not added
    // F: Fn(&f64, &f64) -> f64,
    //F: Fn(&T, &T) -> f64,
{
    // Number of points at the beginning and end that are used to calculate the LB_Kim
    // This number MUST be between 0 and 2*q.len() but you probably don't want to change it from the default of 3
    // 0 would render this method obsolete so the lowest sensible input would be 1 meaning the LB_Kim for the first and last points are calculated
    let no_pruning_points = 3;

    let mut dist; // Minimal distance between the values
    let mut lb = 0.0; // LB_Kim

    // To calculate the lb we compare the beginning and the end of the sequences. A subsequence of the front and the back is used for this. The 'end' of that subsequence is not the trivially found end of the actual sequence but it is the 'inner end' (the end towards the center of the sequence)
    let mut q_end_idx; // Index of the end of sequence q
    let mut t_end_idx; // Index of the end of sequence t
    let mut q_range; // Range to access the subsequence excluding the end of sequence q
    let mut end_value; // Value of the end

    let q_begin_idx = [0, q.len() - 1]; // The index from which the points are counted from. The first index is for the front and the second for the back
                                        // It is important to check the front first because it enables us to jump if the distance exceeds the bsf
                                        // This variable is mostly necessary to avoid duplicate code and handle both cases (start at front/back) with one for loop
    let t_begin_idx = [0, t.len() / 2 - 1]; // TODO: Double check if the following comment is still true or if this can be avoided
                                            // The div by 2 is necessary, because t is twice as long to avoid the modulo

    let mut candidate_z = [Vec::new(), Vec::new()]; // Stores the z-normalized values of the candidate query
                                                    // The first Vec is for when the subsequence starts at the front, the second for when it starts at the back

    // The lb is calculated for the no_pruning_points first and last values
    for i in 0..no_pruning_points {
        // idx:0 is for calculating the lb at the FRONT of the sequence
        // idx:1 is for calculating the lb at the BACK of the sequence
        for idx in 0..2 {
            if idx == 0 {
                // If the lb is calculated for values at the front of the sequence..
                q_end_idx = q_begin_idx[idx] + i; // .. the 'end' is i values AFTER the actual beginning of the sequence
                t_end_idx = t_begin_idx[idx] + i; // .. the 'end' is i values AFTER the actual beginning of the sequence
                q_range = 0..q_end_idx; // .. and the subsequence begins at 0 and goes until the 'end'
            } else {
                // If the lb is calculated for values at the end of the sequence..
                q_end_idx = q_begin_idx[idx] - i; // .. the 'end' is i values BEFORE the actual end of the sequence
                t_end_idx = t_begin_idx[idx] - i; // .. the 'end' is i values BEFORE the actual end of the sequence
                q_range = q_end_idx + 1..q.len(); // .. and the subsequence begins at the next index and goes until the end of the sequence
            };

            // Calculate the z-normalized end value
            end_value = (t[j + t_end_idx] - mean) / std;

            // Calculate the minimal distance between the subsequences and the 'end' points and the distance between the 'ends'
            dist = min_dist(
                (&end_value, &candidate_z[idx]),
                (&q[q_end_idx], &q[q_range]),
                cost_fn,
            );
            // Add the distance to the lb. The dtw calculation does a similar many to many comparison and we could never get a lower value than lb
            lb += dist;

            // If the distance was larger then bsf ..
            if dist >= bsf {
                // .. and if we got the minimal distance from the front values, we can jump those values, because there is no warping possible, that
                let jump_size = if idx == 0 { 1 + i } else { 1 };
                return (lb, jump_size);
            }

            // If the lb is greater than bsf, we move to the next candidate subsequence
            if lb >= bsf {
                return (lb, 1);
            }

            // Store the z-normalized value for the next calculations
            candidate_z[idx].push(end_value);
        }
        // If the lb was not greater then bsf, we have a new lb and return it
    }
    (lb, 1)
}

/// This function calculates the differences between the upper and lower envelopes of a time series with another time series
/// It is a cheap calculation compared to DTW
///
/// Variable Explanation,
/// order          : sorted indices for the query
/// data           : a circular array keeping the current data
/// upper_envelope : upper envelope for the sequence
/// lower_envelope : lower envelops for the sequence
/// cum_bound      : previoulsy calculated bound that can be added to
/// j              : index of the starting location in the
///
/// This function only works when order, data, the envelopes and the cum_bound are of the same lengths. Otherwise the method panics!
fn lb_keogh_cumulative<F>(
    order: &[usize],
    data: &[f64],
    upper_envelope: &[f64],
    lower_envelope: &[f64],
    cum_bound: Option<Vec<f64>>,
    j: usize,
    mean: f64,
    std: f64,
    bsf: f64,
    data_bound: bool,
    cost_fn: &F,
) -> (f64, Vec<f64>, usize)
where
    F: Fn(&f64, &f64) -> f64 + Copy,
    //where
    // T: Div<Output = T> + Sub<Output = T>,
    //  F: Fn(&f64, &f64) -> f64,
{
    let mut cum_bound = if let Some(mut bound) = cum_bound {
        {
            bound[0] = 0.0; // Needs to be deleted, otherwise the first point would count twice
            bound[data.len() - 1] = 0.0; // Needs to be deleted, otherwise the last point would count twice
            bound
        }
    } else {
        vec![0.0; data.len()]
    };
    let mut q_z;
    let mut u_z;
    let mut l_z;
    let mut diff;

    let mut lb: f64 = 0.0;
    let mut jump = order[0];

    for i in 0..order.len() {
        if data_bound {
            q_z = data[i];
            u_z = (upper_envelope[j + order[i]] - mean) / std;
            l_z = (lower_envelope[j + order[i]] - mean) / std;
        } else {
            q_z = (data[(j + order[i])] - mean) / std;
            u_z = upper_envelope[i];
            l_z = lower_envelope[i];
        }

        if order[i] < jump {
            jump = order[i]
        }

        if q_z > u_z {
            diff = cost_fn(&u_z, &q_z);
        } else if q_z < l_z {
            diff = cost_fn(&l_z, &q_z);
        } else {
            diff = 0.0;
        }

        lb += diff + cum_bound[order[i]];
        cum_bound[order[i]] += diff;

        if lb >= bsf {
            break;
        }
    }
    (lb, cum_bound, jump + 1)
}
