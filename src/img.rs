use crate::piece::TETROMINO;
use ndarray::{s, Array2, ArrayView2};
use std::cmp::{max, min};

fn fill(stage: &mut Array2<u8>, start: (usize, usize), to: u8) -> isize {
    let &[h, w] = stage.shape() else {
        unreachable!()
    };
    let mut stack = vec![start];
    let from = stage[start];
    stage[start] = to;
    let mut count = 0;
    while let Some((y, x)) = stack.pop() {
        count += 1;
        if y > 0 && stage[(y - 1, x)] != to && stage[(y - 1, x)] == from {
            stage[(y - 1, x)] = to;
            stack.push((y - 1, x));
        }
        if y < h - 1 && stage[(y + 1, x)] != to && stage[(y + 1, x)] == from {
            stage[(y + 1, x)] = to;
            stack.push((y + 1, x));
        }
        if x > 0 && stage[(y, x - 1)] != to && stage[(y, x - 1)] == from {
            stage[(y, x - 1)] = to;
            stack.push((y, x - 1));
        }
        if x < w - 1 && stage[(y, x + 1)] != to && stage[(y, x + 1)] == from {
            stage[(y, x + 1)] = to;
            stack.push((y, x + 1));
        }
    }
    return count;
}

pub struct Segment {
    pub map: Array2<bool>,
    pub map_size: usize,
    pub x: usize,
    pub y: usize,
}

pub fn segment(map: &Array2<bool>) -> Vec<Segment> {
    // warning! slow
    let &[h, w] = map.shape() else { unreachable!() };
    let mut stage = map.mapv(|x| x as u8);
    let mut segment_id = 1;
    let mut segments = vec![];
    for y in 0..h {
        for x in 0..w {
            if stage[(y, x)] == 1 {
                segment_id += 1;
                assert_ne!(segment_id, 0);
                let size = fill(&mut stage, (y, x), segment_id);
                let mut y_min = h;
                let mut y_max = 0;
                let mut x_min = w;
                let mut x_max = 0;
                for ((y, x), _) in stage.indexed_iter().filter(|(_, v)| **v == segment_id) {
                    y_max = max(y_max, y);
                    y_min = min(y_min, y);
                    x_max = max(x_max, x);
                    x_min = min(x_min, x);
                }
                segments.push(Segment {
                    map: stage
                        .slice(s![y_min..(y_max + 1), x_min..(x_max + 1)])
                        .mapv(|v| v == segment_id),
                    x: x_min,
                    y: y_min,
                    map_size: size as usize,
                });
            }
        }
    }
    segments
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum EvalResult {
    Valid {
        /// Number of contiguous filled areas
        chunk: i32,
        /// Number of filled cells
        filled: i32,
        /// Number of filled cells that is touching unfilled cell
        surface: i32,
        /// Number of contiguous unfilled areas
        fragment: i32,
        /// Number of contiguous unfilled areas which their area cannot divide by 4
        fragment_non4: i32,
        /// Number of contiguous unfilled areas that are not connected to edge of map
        hole: i32,
        /// Number of filled cells that is adjacent to edge of map
        edge: i32,
    },
    Invalid,
}

pub fn eval(map: &Array2<bool>, data: &Array2<u8>) -> EvalResult {
    let &[h, w] = map.shape() else { unreachable!() };
    let mut stage: Array2<u8> = Array2::zeros(map.raw_dim());

    let mut filled = 0;
    // lay
    for ((y, x), &v) in data.indexed_iter() {
        if v == 0 {
            continue;
        }
        let piece = TETROMINO[v as usize];
        for (dy, dx) in piece {
            let (y, x) = (y + dy as usize, x + dx as usize);
            if y >= h || x >= w {
                return EvalResult::Invalid; // Out of bound
            }
            if stage[(y, x)] != 0 {
                return EvalResult::Invalid; // Collision
            }
            stage[(y, x)] = 1;
            filled += 1;
        }
    }
    if filled == 0 {
        return EvalResult::Valid {
            chunk: 0,
            filled: 0,
            surface: 0,
            fragment: 1,
            fragment_non4: 0,
            hole: 0,
            edge: 0,
        };
    }

    // find first
    let mut chunk = 0;
    for y in 0..h {
        for x in 0..w {
            if stage[(y, x)] == 1 {
                chunk += 1;
                fill(&mut stage, (y, x), 2);
            }
        }
    }

    // 0 = empty, 2 = filled

    // scan
    let mut surface = 0;
    let mut edge = 0;
    for ((y, x), mapv) in map.indexed_iter() {
        let stagev = &mut stage[(y, x)];
        if *mapv {
            if *stagev == 0 {
                *stagev = 1;

                if false
                    || (y > 0 && stage[(y - 1, x)] == 2)
                    || (y < h - 1 && stage[(y + 1, x)] == 2)
                    || (x > 0 && stage[(y, x - 1)] == 2)
                    || (x < w - 1 && stage[(y, x + 1)] == 2)
                {
                    surface += 1
                }
            } else if *stagev == 2 {
                if (x == 0 || x == w - 1 || y == 0 || y == h - 1)
                    || !map[(y - 1, x)]
                    || !map[(y + 1, x)]
                    || !map[(y, x - 1)]
                    || !map[(y, x + 1)]
                {
                    edge += 1
                }
            }
        } else {
            if *stagev != 0 {
                return EvalResult::Invalid; // piece outside map
            }
        }
    }

    // 0 = empty, 1 = unfilled, 2 = filled

    // find fragmented
    let mut fragment = 0;
    let mut fragment_non4 = 0;
    for y in 0..h {
        for x in 0..w {
            if stage[(y, x)] != 1 {
                continue;
            }
            let chunk = fill(&mut stage, (y, x), 0);
            if chunk % 4 != 0 {
                fragment_non4 += 1;
            }
            fragment += 1;
        }
    }

    // 0 = empty, 2 = filled

    // find hole
    let hole = {
        let mut reachable = 0;
        for ((y, x), mapv) in map.indexed_iter() {
            if !*mapv && stage[(y, x)] == 0 {
                reachable += fill(&mut stage, (y, x), 1);
            }
        }
        for y in 0..h {
            if stage[(y, 0)] == 0 {
                reachable += fill(&mut stage, (y, 0), 1);
            }
            if stage[(y, w - 1)] == 0 {
                reachable += fill(&mut stage, (y, w - 1), 1);
            }
        }
        for x in 1..(w - 1) {
            if stage[(0, x)] == 0 {
                reachable += fill(&mut stage, (0, x), 1);
            }
            if stage[(h - 1, x)] == 0 {
                reachable += fill(&mut stage, (h - 1, x), 1);
            }
        }

        (h * w) as i32 - reachable as i32 - filled
    };

    // 0 = unreachable, 1=empty, 2 = filled

    return EvalResult::Valid {
        chunk,
        filled,
        surface,
        fragment,
        fragment_non4,
        hole,
        edge,
    };
}

pub fn lay(data: &Array2<u8>) -> Array2<bool> {
    let &[h, w] = data.shape() else {
        unreachable!()
    };
    let mut stage = Array2::from_elem(data.raw_dim(), false);
    for ((y, x), &v) in data.indexed_iter() {
        if v == 0 {
            continue;
        }
        let piece = TETROMINO[v as usize];
        for (dy, dx) in piece {
            let (y, x) = (y + dy as usize, x + dx as usize);
            if y >= h || x >= w {
                continue;
            }
            stage[(y, x)] = true;
        }
    }
    stage
}

pub fn transfer(map: ArrayView2<bool>, ref_map: ArrayView2<u8>) -> Array2<u8> {
    let mut c = Array2::zeros(map.raw_dim());
    let mut stage = Array2::from_elem(map.raw_dim(), false);

    let &[h, w] = map.shape() else { unreachable!() };
    'outer: for ((y, x), v) in ref_map.indexed_iter() {
        if *v == 0 {
            continue;
        }
        let piece = TETROMINO[*v as usize];
        for (dy, dx) in piece {
            let (y, x) = (y + dy as usize, x + dx as usize);
            if y >= h || x >= w {
                continue 'outer;
            }
            if stage[(y, x)] || !map[(y, x)] {
                continue 'outer;
            }
            stage[(y, x)] = true;
        }
        c[(y, x)] = *v;
    }
    c
}

/// For visualizing
pub fn dump(map: &Array2<bool>, data: &Array2<u8>) -> Vec<String> {
    const DUMP_COLOR: &[u8] = b"0123456789@#=+*%$ikfgreqzan";
    let &[h, w] = map.shape() else { unreachable!() };
    let mut stage: Array2<u8> = map.mapv(|x| if x { b'.' } else { b' ' });

    let mut counter = 0;
    // fill
    for ((y, x), &v) in data.indexed_iter() {
        if v == 0 {
            continue;
        }
        let piece = TETROMINO[v as usize];
        let color = DUMP_COLOR[counter % DUMP_COLOR.len()];
        counter += 1;
        for (dy, dx) in piece {
            let (y, x) = (y + dy as usize, x + dx as usize);
            if y >= h || x >= w {
                continue;
            }
            if stage[(y, x)] != b'.' {
                stage[(y, x)] = b'X';
            } else {
                stage[(y, x)] = color;
            }
        }
    }
    stage
        .outer_iter()
        .map(|row| {
            std::str::from_utf8(row.as_slice().unwrap())
                .unwrap()
                .to_owned()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn test_simple() {
        let map = array![
            [0, 1, 1], //
            [1, 1, 1],
            [1, 1, 1],
        ]
        .mapv(|x| x != 0);
        let data = array![
            [0, 0, 0], //
            [0, 1, 0],
            [0, 0, 0],
        ];
        let score = eval(&map, &data);
        assert_eq!(
            score,
            EvalResult::Valid {
                filled: 4,
                fragment: 2,
                hole: 0,
                surface: 4,
                fragment_non4: 2,
                chunk: 1,
                edge: 3,
            }
        );
    }
    #[test]
    fn test_valid() {
        let map = array![
            [0, 1, 1, 1], //
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
        .mapv(|x| x != 0);
        let data = array![
            [0, 0, 4, 0], //
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ];
        let score = eval(&map, &data);
        assert_eq!(
            score,
            EvalResult::Valid {
                filled: 8,
                fragment: 1,
                hole: 0,
                surface: 3,
                fragment_non4: 1,
                chunk: 1,
                edge: 7,
            }
        );
    }
    #[test]
    fn test_outofbound() {
        let map = array![
            [0, 1, 1], //
            [1, 1, 1],
            [1, 1, 1],
        ]
        .mapv(|x| x != 0);
        let data = array![
            [0, 0, 0], //
            [0, 0, 0],
            [0, 0, 1],
        ];
        let score = eval(&map, &data);
        assert_eq!(score, EvalResult::Invalid);
    }
    #[test]
    fn test_hole() {
        let map = array![
            [1, 1, 1], //
            [1, 1, 1],
            [1, 1, 1],
        ]
        .mapv(|x| x != 0);
        let data = array![
            [6, 4, 0], //
            [0, 0, 0],
            [0, 0, 0],
        ];
        let score = eval(&map, &data);
        assert_eq!(
            score,
            EvalResult::Valid {
                filled: 8,
                fragment: 1,
                hole: 1,
                surface: 1,
                fragment_non4: 1,
                chunk: 1,
                edge: 8,
            }
        );
    }
    #[test]
    fn test_maphole() {
        let map = array![
            [1, 1, 1], //
            [1, 0, 1],
            [1, 1, 1],
        ]
        .mapv(|x| x != 0);
        let data = array![
            [6, 4, 0], //
            [0, 0, 0],
            [0, 0, 0],
        ];
        let score = eval(&map, &data);
        assert_eq!(
            score,
            EvalResult::Valid {
                filled: 8,
                fragment: 0,
                hole: 0,
                surface: 0,
                fragment_non4: 0,
                chunk: 1,
                edge: 8,
            }
        );
    }
}
