use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

#[derive(Clone)]
pub struct DataIterator {
    pub data: Vec<f64>,
    index: usize,
}

impl DataIterator {
    pub fn new(filename: &str) -> Self {
        // Open the file in read-only mode.
        let mut file = File::open(filename).unwrap();
        let mut content = String::new();
        // Read all the file content into a variable (ignoring the result of the operation).
        file.read_to_string(&mut content).unwrap();
        let data = content
            .split_whitespace()
            .filter_map(|w| w.parse::<f64>().ok())
            .collect();
        DataIterator { data, index: 0 }
    }
}

impl Iterator for DataIterator {
    type Item = f64;
    fn next(&mut self) -> Option<f64> {
        let result = self.data.get(self.index);
        self.index += 1;
        result.cloned()
    }
}

pub struct QueryIterator {
    buf_reader: BufReader<File>,
    buffer: String,
}

impl QueryIterator {
    pub fn new(filename: &str) -> Self {
        // Open the file in read-only mode.
        let file = File::open(filename).unwrap();
        let buffer = String::new(); // Buffer to read to
        let buf_reader = BufReader::new(file);
        QueryIterator { buf_reader, buffer }
    }
}

impl Iterator for QueryIterator {
    type Item = f64;
    fn next(&mut self) -> Option<f64> {
        loop {
            // If the end of the file was reached, return None
            if self.buf_reader.read_line(&mut self.buffer).unwrap() == 0 {
                return None;
                // If data was read from the file, try parsing it. If that fails, skip to the next observation
            }
            let observation = self.buffer.trim().parse::<f64>().ok();
            self.buffer.clear();
            if observation.is_some() {
                return observation;
            }
        }
    }
}

pub fn make_test_series() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let a1 = vec![1.0, 1.];
    let a2 = vec![2.0, 1.];
    let a3 = vec![3.0, 1.];
    let a4 = vec![2.0, 1.];
    let a5 = vec![2.13, 1.];
    let a6 = vec![1.0, 1.];
    let b1 = vec![1.0, 1.];
    let b2 = vec![1.0, 1.];
    let b3 = vec![2.0, 1.];
    let b4 = vec![2.0, 1.];
    let b5 = vec![2.42, 1.];
    let b6 = vec![3.0, 1.];
    let b7 = vec![2.0, 1.];
    let b8 = vec![1.0, 1.];
    let series_1 = vec![a1, a2, a3, a4, a5, a6];
    let series_2 = vec![b1, b2, b3, b4, b5, b6, b7, b8];
    (series_1, series_2)
}
