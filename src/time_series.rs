use std::collections::VecDeque;
use std::ops::Index;

pub struct TimeSeries {
    observations_original: VecDeque<f64>,
    observations_normalized: Vec<f64>,
    ex: f64,
    ex2: f64,
    mean: f64,
    std: f64,
}

impl Default for TimeSeries {
    fn default() -> Self {
        TimeSeries::new()
    }
}

impl TimeSeries {
    pub fn new() -> Self {
        TimeSeries {
            observations_original: VecDeque::new(),
            observations_normalized: Vec::new(),
            ex: 0.0,
            ex2: 0.0,
            mean: 0.0,
            std: 0.0,
        }
    }

    // Creates a new TimeSeries struct from an Iterator
    // All observations of the Iterator are part of the TimeSeries
    pub fn from<I: std::iter::Iterator<Item = f64>>(series: I) -> Self {
        let mut time_series = Self::new();
        for observation in series {
            time_series.append(observation);
        }
        time_series
    }

    // Return the length of the TimeSeries
    pub fn len(&self) -> usize {
        self.observations_original.len()
    }

    // Test if the TimeSeries is empty
    pub fn is_empty(&self) -> bool {
        self.observations_original.is_empty()
    }

    // Add an observation to the TimeSeries struct
    // Also updates the fields for ex and ex2 and removes the normalized observations
    pub fn append(&mut self, observation: f64) {
        self.observations_original.push_back(observation);
        self.ex += observation;
        self.ex2 += observation.powi(2);
        self.observations_normalized.clear();
    }

    // Pop the observation at the front
    // Also updates the fields for ex and ex2
    // It DOES NOT remove the normalized observations because this method should only be used in conjunction
    // with the append method and then the normalized observations would be cleared twice
    pub fn pop_front(&mut self) {
        if let Some(front) = self.observations_original.pop_front() {
            self.ex -= front;
            self.ex2 -= front.powi(2);
        }

        // Commented out because this method should only be called by append_shift
        // and then it is cleared from calling append
        // self.observations_normalized.clear();
    }

    // Appends the observation to the back and pops the observation from the front
    // Can be used to shift the TimeSeries of the subsequence one observation to the right in the data series
    pub fn append_shift(&mut self, observation: f64) {
        self.pop_front();
        self.append(observation);
    }

    // Calculate the mean and std of the time series
    pub fn calc_mean_std(&mut self) {
        self.mean = self.ex / self.len() as f64;
        self.std = f64::sqrt((self.ex2 / self.len() as f64) - self.mean.powi(2));
    }

    // Normalize all observations and save them in the observations_normalized field
    pub fn normalize_all(&mut self) {
        let mean = self.mean;
        let std = self.std;
        self.observations_normalized = self
            .observations_original
            .iter_mut()
            .map(|&mut observation| (observation - mean) / std)
            .collect();
    }

    // Normalizes the observation at the index and appends it to the observations_normalized field
    pub fn normalize(&mut self, index: usize) {
        self.observations_normalized
            .push((self.observations_original[index] - self.mean) / self.std);
    }
}

// Allows indexing the TimeSeries struct
impl Index<usize> for TimeSeries {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        // Possibly check if normalized before??
        &self.observations_normalized[index]
    }
}

/*impl IndexMut<usize> for TimeSeries {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.observations[index]
    }
}*/
