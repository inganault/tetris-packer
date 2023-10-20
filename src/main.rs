use std::{
    cmp::max,
    collections::VecDeque,
    time::{Duration, Instant},
};

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use ndarray::prelude::*;
use ndarray_npy::{NpzReader, NpzWriter};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

mod ga;
mod img;
mod piece;

#[derive(Parser, Debug)]
struct Args {
    /// Input .npz
    file: String,
    /// Reference .npz (output file from last frame)
    ref_file: Option<String>,
    #[arg(short)]
    output_path: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let map = {
        let fp = std::fs::File::open(&args.file).with_context(|| anyhow!("file not found"))?;
        let mut npz = NpzReader::new(fp).with_context(|| anyhow!("cannot open npz"))?;
        let raw: Array2<u8> = (npz.by_name("map"))
            .or_else(|_| npz.by_name("map.npy"))
            .with_context(|| anyhow!("map var not found"))?;
        raw.mapv(|x| x != 0)
    };
    let ref_map = if let Some(ref_file) = &args.ref_file {
        let fp = std::fs::File::open(ref_file).with_context(|| anyhow!("ref file not found"))?;
        let mut npz = NpzReader::new(fp).with_context(|| anyhow!("cannot open ref npz"))?;
        let raw: Array2<u8> = npz
            .by_name("piece")
            .with_context(|| anyhow!("piece var not found"))?;
        Some(raw)
    } else {
        None
    };

    let mut composite: Array2<u8> = Array2::zeros(map.raw_dim());

    for seg in img::segment(&map) {
        if seg.map_size < 4 {
            continue;
        }
        let map = trim_remainder(&seg);
        let &[h, w] = map.shape() else { unreachable!() };

        let ref_map = ref_map
            .as_ref()
            .map(|m| m.slice(s![seg.y..(seg.y + h), seg.x..(seg.x + w),]));

        let goal = map.iter().map(|x| *x as i32).sum::<i32>() / 4 * 4;
        let mut candidate = vec![];
        let mut last_ga = None;
        let mut success = false;
        let start = Instant::now();
        for seed in 0..20 {
            // transfer ref
            let ref_map = ref_map.map(|ref_map| {
                for seed2 in 0..10 {
                    match trim(map.view(), ref_map.view(), seed + 1000 * seed2) {
                        Ok(out) => return out,
                        Err(_) => continue,
                    }
                }
                std::process::exit(-1);
            });

            match grow(
                map.view(),
                ref_map.as_ref().map(|x| x.view()),
                seed,
                goal,
                false,
            ) {
                Ok(ga) => {
                    success = true;
                    candidate.push(ga.candidate[0].clone());
                    last_ga = Some(ga);
                    if candidate.len() >= 3 || ref_map.is_none() {
                        break;
                    }
                }
                Err(ga) => {
                    last_ga = Some(ga);
                }
            }
        }
        println!("elapsed: {:?}", start.elapsed());
        let ga = last_ga.as_ref().unwrap();
        if success {
            println!("generation: {}", ga.generation);
            for row in img::dump(&ga.cfg.map, &candidate[0].data) {
                println!("|{}|", row);
            }
            println!("  {:?} = {}", candidate[0].raw_score, ga.candidate[0].score);
        } else {
            println!("{:?}", ga);
            println!("Failed");
            // add failed candidate for base line
            candidate.push(ga.candidate[0].clone());

            // try hard mode
            for seed in 0..5 {
                // transfer ref
                let ref_map = ref_map.map(|ref_map| {
                    for seed2 in 0..10 {
                        match trim(map.view(), ref_map.view(), seed + 1000 * seed2) {
                            Ok(out) => return out,
                            Err(_) => continue,
                        }
                    }
                    std::process::exit(-1);
                });

                let goal = seg.map.iter().map(|x| *x as i32).sum::<i32>() / 4 * 4;
                match grow(
                    seg.map.view(),
                    ref_map.as_ref().map(|x| x.view()),
                    seed,
                    goal,
                    true,
                ) {
                    Ok(ga) => {
                        candidate.push(ga.candidate[0].clone());
                        break;
                    }
                    Err(ga) => {
                        candidate.push(ga.candidate[0].clone());
                    }
                }
            }
        }

        // rank candidate by similarity
        candidate.par_iter_mut().for_each(|c| {
            let similarity = ref_map
                .map(|ref_map| {
                    c.data
                        .iter()
                        .zip(ref_map)
                        .map(|(a, b)| (*a != 0 && a == b) as i32)
                        .sum()
                })
                .unwrap_or(0i32);
            c.score = if let img::EvalResult::Valid { filled, .. } = c.raw_score {
                filled * 10 + similarity
            } else {
                0
            };
        });
        candidate.sort_by_key(|c| -c.score);

        composite
            .slice_mut(s![seg.y..(seg.y + h), seg.x..(seg.x + w),])
            .zip_mut_with(&candidate[0].data, |comp, diff| {
                *comp += diff;
            });
    }
    println!("Final");
    for row in img::dump(&map, &composite) {
        println!("|{}|", row);
    }

    let output_name: String = args.output_path.unwrap_or_else(|| {
        if args.file.ends_with(".npz") {
            let stem = &args.file[..(args.file.len() - 4)];
            format!("{}_out.npz", stem)
        } else {
            format!("{}_out.npz", &args.file)
        }
    });
    let fp = std::fs::File::create(output_name).with_context(|| "Cannot create output file")?;
    let mut npz = NpzWriter::new(fp);
    npz.add_array("piece", &composite).unwrap();
    npz.finish().with_context(|| "Cannot write output file")?;

    Ok(())
}

fn trim_remainder(seg: &img::Segment) -> Array2<bool> {
    let &[h, w] = seg.map.shape() else {
        unreachable!()
    };
    if seg.map_size % 4 != 0 {
        let mut map = seg.map.clone();
        let mut need_remove = seg.map_size % 4;
        'outer: for y in 0..h {
            for x in 0..w {
                if map[(y, x)] {
                    map[(y, x)] = false;
                    if img::segment(&map).len() != 1 {
                        map[(y, x)] = true;
                        continue;
                    }
                    need_remove -= 1;
                    if need_remove == 0 {
                        break 'outer;
                    }
                }
            }
        }
        map
    } else {
        seg.map.clone()
    }
}

fn trim(map: ArrayView2<bool>, ref_map: ArrayView2<u8>, seed: u64) -> Result<Array2<u8>> {
    let new_ref = img::transfer(map.view(), ref_map);
    let mut ga = ga::GA::new(
        ga::Config {
            map: map.to_owned(),
            score: ga::score_trim,
            ..Default::default()
        },
        seed,
        Some(new_ref),
    );
    ga.cfg.score_phase = ga.cfg.map_size as i32;
    let success = loop {
        if ga.generation % 100 == 0 {
            ga.cfg.score_phase -= 4;
            ga.rescore();
        }
        ga.step();
        if matches!(ga.candidate[0].raw_score,
            img::EvalResult::Valid { fragment, .. } if fragment <= 1)
        {
            break true;
        }
        if ga.generation >= 100000 {
            break false;
        }
    };
    if !success {
        bail!("trim failed")
    }
    println!("trim: {}", success);
    for row in img::dump(&ga.cfg.map, &ga.candidate[0].data) {
        println!("|{}|", row);
    }
    Ok((*ga.candidate[0].data).clone())
}

fn grow(
    map: ArrayView2<bool>,
    ref_map: Option<ArrayView2<u8>>,
    seed: u64,
    goal: i32,
    try_hard: bool,
) -> Result<ga::GA, ga::GA> {
    let mut ga = ga::GA::new(
        ga::Config {
            map: map.to_owned(),
            ref_map: ref_map.map(|m| m.to_owned()),
            ..Default::default()
        },
        seed,
        None,
    );
    if let Some(ref_map) = ref_map {
        ga.add_candidate(ref_map.to_owned());
        if ga.candidate[0].score < 0 {
            ga.cfg.score_chunk = match ga.candidate[0].raw_score {
                img::EvalResult::Valid { chunk, .. } => max(1, chunk),
                _ => 1,
            };
            ga.rescore();
        }
    }
    let mut last_score: VecDeque<_> = [-1, -2, -3].into();
    let mut status_timer = Instant::now();
    loop {
        ga.step();
        if matches!(ga.candidate[0].raw_score,
            img::EvalResult::Valid { filled, .. } if filled >= goal)
        {
            return Ok(ga);
        }
        if ga.generation >= 100000 {
            return Err(ga);
        }
        let score = ga.candidate[0].score;
        if ga.generation % 1000 == 0 {
            if last_score.iter().all(|v| *v == score) {
                println!("seed: {} stuck @ gen: {}", seed, ga.generation);
                if try_hard {
                    if ga.cfg.score_phase == 1 {
                        return Err(ga);
                    }
                    ga.cfg.score_phase = 1;
                    ga.rescore();
                    continue;
                } else {
                    return Err(ga);
                }
            }
            last_score.pop_front();
            last_score.push_back(score);
        }

        // show progress
        if ga.generation % 1000 == 0 && status_timer.elapsed() > Duration::from_secs(3) {
            println!(
                "generation: {}, score: {}",
                ga.generation, ga.candidate[0].score
            );
            for row in img::dump(&ga.cfg.map, &ga.candidate[0].data) {
                println!("|{}|", row);
            }
            println!("   {:?}", ga.candidate[0].raw_score);
            status_timer = Instant::now();
        }
    }
}
