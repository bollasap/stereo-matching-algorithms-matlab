# Stereo Matching Algorithms in MATLAB

Optimized (very fast) stereo matching algorithms in MATLAB. It includes implementations of Block Matching, Dynamic Programming, Semi-Global Matching, Semi-Global Block Matching and Belief Propagation.

There is also the Python port **[Stereo Matching Algorithms in Python](https://github.com/bollasap/stereo-matching-algorithms-python)**.

## Features

### Multiple stereo vision algorithms

- **Block Matching (BM)**
- **Dynamic Programming (DP)**
- **Semi-Global Matching (SGM)**
- **Semi-Global Block Matching (SGBM)**
- **Belief Propagation (BP)**

### Two different versions of Dynamic Programming

- Dynamic Programming with Left–Right Axes DSI
- Dynamic Programming with Left–Disparity Axes DSI

### Three different versions of Belief Propagation

- Belief Propagation with accelerated message update schedule
- Belief Propagation with synchronous message update schedule
- Belief Propagation with synchronous message update schedule (alternative approach)

All algorithms are accelerated for performance using matrix operations.

## Installation

Download the project as ZIP file, unzip it, and run the scripts.

## Usage

The project contains eight MATLAB scripts, each implementing a different stereo matching algorithm. The files `left.png` and `right.png` contain the stereo image pair used as input.
To use a different stereo pair, replace these two images with your own. In this case, you must also adjust the **disparity levels** parameter in the script you are running.
You may optionally modify other parameters as needed. If the input images contain little or no noise, it is recommended not to use the Gaussian filter.

## Results

Below are the disparity maps produced by the different algorithms when using the **Tsukuba stereo pair**.

![Tsukuba Stereo Image](left.png) ![Tsukuba Stereo Image](right.png)

### Block Matching

![Block Matching Disparity Map](results/disparity0.png)

### Dynamic Programming (Left-Right)

![Dynamic Programming (Left-Right) Disparity Map](results/disparity1.png)

### Dynamic Programming (Left-Disparity)

![Dynamic Programming (Left-Disparity) Disparity Map](results/disparity2.png)

### Semi-Global Matching

![Semi-Global Matching Disparity Map](results/disparity3.png)

### Semi-Global Block Matching

![Semi-Global Block Matching Disparity Map](results/disparity4.png)

### Belief Propagation (Accelerated)

![Belief Propagation Accelerated Disparity Map](results/disparity5.png)

### Belief Propagation (Synchronous)

![Belief Propagation Synchronous Disparity Map](results/disparity6.png)

The two different approaches to Belief Propagation produce the same result.

## Related Repositories

- [Stereo Matching Algorithms in Python](https://github.com/bollasap/stereo-matching-algorithms-python)
- [Block Matching for Stereo Matching](https://github.com/bollasap/block-matching-for-stereo)
- [Stereo Matching using Dynamic Programming (Left-Right Axes)](https://github.com/bollasap/stereo-matching-using-dynamic-programming-left-right)
- [Stereo Matching using Dynamic Programming (Left-Disparity Axes)](https://github.com/bollasap/stereo-matching-using-dynamic-programming-left-disparity)
- [Semi-Global Matching](https://github.com/bollasap/semi-global-matching)
- [Semi-Global Block Matching](https://github.com/bollasap/semi-global-block-matching)
- [Stereo Matching using Belief Propagation (Accelerated)](https://github.com/bollasap/stereo-matching-using-belief-propagation-accelerated)
- [Stereo Matching using Belief Propagation (Synchronous)](https://github.com/bollasap/stereo-matching-using-belief-propagation-synchronous)
- [Stereo Matching using Belief Propagation (Synchronous) - a different aproach](https://github.com/aposb/stereo-matching-using-belief-propagation-fast)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
