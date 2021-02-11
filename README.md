# kNN_DTW
This is an implementation of a k-Nearest Neighbor search using Dynamic Time Warping. It is implemented in 100% safe Rust code and does not rely on any dependencies. It is using all the optimizations suggested in the UCR_URS suite and the EAPrunedDTW. As far as I am aware of, this implementation should yield state-of-the-art time complexity and a space complexity of O(n).

## Usage
Add the dependency to your Cargo.toml and you can use the provided functions
```rust
```

## Fixed bugs/differences in regards to URC suite
- Fixed sorting bug
  The UCR suite suggest sorting the query to improve the speed. It is not properly sorting though
- The cumulative bound (variable cb) ends with zero
  The cb[i] represents the lower bound of the distance that we will accumulate from the index i until the end. Once we reached the end,
  no further cost can accumulate. If cb wasn't zero at the end, we would require the DTW of the candidate sequence to be at least by that much better than the bsf
- Using an implementation similar to the faster EAPrunedDTW
- Cost function can be easily replaced
- Observations can have any type
- Multi dimensional observations possible
- Envelope of the query take the boundary constraint into consideration 
  (upper and lower envelope at first and last point are the value of the query at that point)

## Potential for improvements
- Allow time series of unequal lengths to be compared (The calculation of the lower bounds poses the problem)
- Change envelope of the candidate subsequences too, so they also take the boundary constraint into consideration 
  (upper and lower envelope at first and last point should be the value of the candidate sequence at that point)
- Calculate the lower bound of the data sequence only when needed
- Use the z-normalized data sequence from the calculation of the lower bounds
- parallel computation
- Allow sets of candidate sequences or subsequences
- Allow the calculation for query where observations can be added

## Changelog
