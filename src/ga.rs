use crate::img;
use crate::piece::TETROMINO;
use ndarray::prelude::*;
use ndarray::Array2;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::Hasher;
use std::sync::Arc;
use std::vec;
use std::{cmp::max, cmp::min, fmt::Debug};

#[derive(Debug)]
pub struct Config {
    pub map: Array2<bool>,
    pub map_size: usize,
    pub ref_map: Option<Array2<u8>>,
    pub size: usize,
    pub mutate: usize,
    pub crossover: usize,
    pub good_pool: usize,
    pub score: fn(&Config, &img::EvalResult) -> i32,
    pub score_phase: i32,
    pub score_chunk: i32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            map: Default::default(),
            map_size: 0,
            ref_map: None,
            size: 64,
            mutate: 21,
            crossover: 16,
            good_pool: 32,
            score: score_grow,
            score_phase: 0,
            score_chunk: 1,
        }
    }
}

#[derive(Clone)]
pub struct Candidate {
    pub score: i32,
    pub raw_score: img::EvalResult,
    pub data: Arc<Array2<u8>>,
    hash: u64,
}

pub struct GA {
    pub cfg: Config,
    pub candidate: Vec<Candidate>,
    pub generation: usize,

    rng: Pcg64Mcg,
    empty: Candidate,
}

impl GA {
    pub fn new(mut cfg: Config, seed: u64, start: Option<Array2<u8>>) -> Self {
        cfg.map_size = cfg.map.iter().map(|x| *x as usize).sum();

        let start = start.unwrap_or_else(|| Array2::zeros(cfg.map.raw_dim()));
        let empty = mk_candidate(&cfg, start);
        Self {
            candidate: vec![],
            generation: 0,
            rng: Pcg64Mcg::seed_from_u64(seed),
            empty,
            cfg,
        }
    }

    pub fn add_candidate(&mut self, c: Array2<u8>) {
        self.candidate.push(mk_candidate(&self.cfg, c));
    }
    pub fn rescore(&mut self) {
        let cfg = &self.cfg;
        self.candidate.par_iter_mut().for_each(|c| {
            c.score = (cfg.score)(cfg, &c.raw_score);
        });
        self.empty.score = (cfg.score)(cfg, &self.empty.raw_score);
    }

    pub fn step(&mut self) {
        let cfg = &self.cfg;
        self.generation += 1;
        let &[h, w] = cfg.map.shape() else {
            unreachable!()
        };

        let mut candidate: Vec<_> = self
            .candidate
            .iter()
            .cloned()
            .chain(std::iter::from_fn(|| Some(self.empty.clone())))
            .take(cfg.size)
            .collect();

        enum TaskType {
            Mutate,
            Crossover,
        }
        let mut task = vec![];
        for _ in 0..cfg.mutate {
            task.push((TaskType::Mutate, Pcg64Mcg::from_rng(&mut self.rng).unwrap()));
        }
        for _ in 0..cfg.crossover {
            task.push((
                TaskType::Crossover,
                Pcg64Mcg::from_rng(&mut self.rng).unwrap(),
            ));
        }
        let newc: Vec<_> = task
            .into_par_iter()
            .filter_map(|(t, mut rng)| match t {
                TaskType::Mutate => {
                    let mut c = (*candidate[rng.gen_range(0..cfg.size)].data).clone();
                    if !mutate(&cfg, &mut c, &mut rng) {
                        return None;
                    }
                    Some(mk_candidate(&cfg, c))
                }
                TaskType::Crossover => {
                    let parent = rng.gen_range(0..cfg.size);
                    let mut graft = parent;
                    while graft == parent {
                        graft = rng.gen_range(0..cfg.size);
                    }

                    let (y1, y2) = (rng.gen_range(0..h), rng.gen_range(0..h));
                    let (y1, y2) = (min(y1, y2), max(y1, y2));
                    let (x1, x2) = (rng.gen_range(0..w), rng.gen_range(0..w));
                    let (x1, x2) = (min(x1, x2), max(x1, x2));

                    let mut c = (*candidate[parent].data).clone();
                    c.slice_mut(s![y1..=y2, x1..=x2])
                        .assign(&candidate[graft].data.slice(s![y1..=y2, x1..=x2]));
                    Some(mk_candidate(&cfg, c))
                }
            })
            .collect();
        candidate.extend(newc);

        {
            // Filter dup and invalid
            let mut valid = HashSet::new();
            let mut new_candidate = vec![];
            for c in candidate.into_iter() {
                if c.score < 0 {
                    continue;
                }
                let index = (c.score, c.hash);
                if valid.insert(index) {
                    new_candidate.push(c);
                }
            }
            candidate = new_candidate;
        }
        // Rank
        candidate.sort_by_key(|c| -c.score);
        self.candidate.clear();
        if candidate.len() > cfg.size {
            let (good, bad) = candidate.split_at(cfg.good_pool);
            self.candidate.extend_from_slice(good);
            let bad_size = cfg.size - cfg.good_pool;
            let bad_start = (bad.len() - bad_size) / 2;
            self.candidate
                .extend_from_slice(&bad[bad_start..(bad_start + bad_size)]);
            assert_eq!(self.candidate.len(), cfg.size);
        } else {
            self.candidate = candidate;
        }
    }
}

fn mk_candidate(cfg: &Config, data: Array2<u8>) -> Candidate {
    let raw_score = img::eval(&cfg.map, &data);

    let mut hasher = DefaultHasher::new();
    std::hash::Hash::hash_slice(img::lay(&data).as_slice().unwrap(), &mut hasher);
    let hash = hasher.finish();

    Candidate {
        score: (cfg.score)(cfg, &raw_score),
        raw_score,
        hash,
        data: Arc::new(data),
    }
}

pub fn score_grow(cfg: &Config, raw: &img::EvalResult) -> i32 {
    let &img::EvalResult::Valid {
        chunk,
        filled,
        surface,
        fragment,
        fragment_non4,
        hole,
        ..
    } = raw
    else {
        return -100;
    };

    if filled == 0 {
        return 0;
    }

    if chunk > cfg.score_chunk {
        return -101;
    }

    if cfg.map_size % 4 == 0 {
        if fragment_non4 > 0 {
            return -102;
        }
    } else {
        if fragment_non4 > 1 {
            return -102;
        }
    }

    if cfg.score_phase == 0 {
        return max(0, filled * 4 - surface * 2 + 10 - 10 * fragment - 10 * hole);
    } else {
        return max(0, filled * 4 - surface * 2 - 10 * hole);
    }
}

pub fn score_trim(cfg: &Config, raw: &img::EvalResult) -> i32 {
    let &img::EvalResult::Valid {
        filled,
        surface,
        fragment,
        hole,
        edge,
        ..
    } = raw
    else {
        return -104;
    };

    if filled == 0 {
        return 1000000;
    }
    let too_much_fill = max(0, filled - cfg.score_phase);

    return max(
        0,
        1000000 - 2 * edge - 5 * too_much_fill - surface - 50 * fragment - 10 * hole,
    );
}

fn mutate(cfg: &Config, c: &mut Array2<u8>, rng: &mut dyn RngCore) -> bool {
    let &[h, w] = c.shape() else { unreachable!() };
    let piece_count = c.iter().filter(|i| **i != 0).count();
    let can_add = cfg.map_size - 4 * piece_count >= 4;
    let can_remove = piece_count > 0;

    if !can_add && !can_remove {
        return false;
    }
    let do_add = if can_add && can_remove {
        rng.gen()
    } else {
        can_add
    };

    if do_add {
        // Add piece
        let stage = img::lay(&c);
        'outer: for _ in 0..3000 {
            let pos = rng.gen_range(0..(w * h));
            let (y, x) = (pos / w, pos % w);
            let piece_type = rng.gen_range(1..TETROMINO.len());
            for (dy, dx) in TETROMINO[piece_type] {
                let (y, x) = (y + dy as usize, x + dx as usize);
                if y >= h || x >= w {
                    continue 'outer;
                }
                if stage[(y, x)] || !cfg.map[(y, x)] {
                    continue 'outer;
                }
            }
            c[(y, x)] = piece_type as u8;
            return true;
        }
        return false;
    } else {
        // Remove piece
        // TODO: remove only outermost piece
        let pos = rng.gen_range(0..piece_count);
        let v = c.iter_mut().filter(|i| **i != 0).skip(pos).next().unwrap();
        *v = 0;
        return true;
    }
}

impl Debug for GA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "-- GA State --")?;
        writeln!(f, "Generation: {}", self.generation)?;
        for (i, c) in self.candidate.iter().enumerate() {
            writeln!(f, "[{:2}]: {:+}", i, c.score)?;
            for row in img::dump(&self.cfg.map, &c.data) {
                writeln!(f, "    |{}|", row)?;
            }
            writeln!(f, "{} = {:?}", c.score, c.raw_score)?;
        }
        Ok(())
    }
}
