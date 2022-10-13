use std::{
    collections::BTreeSet,
    fs::File,
    io::{BufRead, BufReader},
};

use acrostic_cw::async_cw;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use nanorand::{Rng, WyRand};

fn words(num: usize) -> Vec<Vec<usize>> {
    let mut words = Vec::new();
    for x in 0..num {
        let mut word1 = Vec::new();
        for z in 0..num {
            word1.push(x * num + z);
        }

        words.push(word1);
        let mut word2 = Vec::new();
        for y in 0..num {
            word2.push(x + num * y)
        }

        words.push(word2);
    }
    words
}

fn async_cw_solve(c: &mut Criterion) {
    let file = File::open("dataset/english3.txt").unwrap();

    let buf = BufReader::new(file);
    let bh = buf
        .lines()
        .filter_map(|l| l.ok().map(|s| s.to_lowercase()))
        .collect::<BTreeSet<String>>();
    let dict = fst::Set::from_iter(bh.iter()).unwrap();

    let mut group = c.benchmark_group("async solve");
    let mut rand = WyRand::new_seed(1500);

    let params = async_cw::Parameters::new(words(3));
    group.bench_function("3x3", |b| {
        b.iter_batched(
            || async_cw::Solver::new_seed(&params, &dict, rand.generate()),
            |solver| solver.run().unwrap(),
            BatchSize::PerIteration,
        );
    });

    let mut rand = WyRand::new_seed(1500);
    let params = async_cw::Parameters::new(words(4));
    group.bench_function("4x4", |b| {
        b.iter_batched(
            || async_cw::Solver::new_seed(&params, &dict, rand.generate()),
            |solver| solver.run().unwrap(),
            BatchSize::PerIteration,
        );
    });

    let mut rand = WyRand::new_seed(1500);
    let params = async_cw::Parameters::new(words(5));
    group.bench_function("5x5", |b| {
        b.iter_batched(
            || async_cw::Solver::new_seed(&params, &dict, rand.generate()),
            |solver| solver.run().unwrap(),
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

fn looping_tests(c: &mut Criterion) {
    let mut group = c.benchmark_group("loop");

    let vec: Vec<usize> = (0..100000).collect();
    group.bench_function("iter loop twice", |b| {
        b.iter_with_large_drop(|| {
            let vec1 = vec
                .iter()
                .filter(|i| *i % 2 == 0)
                .copied()
                .collect::<Vec<usize>>();
            let vec2 = vec
                .iter()
                .filter(|i| *i % 3 == 0)
                .copied()
                .collect::<Vec<usize>>();

            (vec1, vec2)
        });
    });

    group.bench_function("for loop once", |b| {
        b.iter_with_large_drop(|| {
            let mut vec1 = Vec::with_capacity(vec.len() / 2 + 1);
            let mut vec2 = Vec::with_capacity(vec.len() / 3 + 1);

            for i in &vec {
                if i % 2 == 0 {
                    vec1.push(*i);
                }
                if i % 3 == 0 {
                    vec2.push(*i);
                }
            }

            (vec1, vec2)
        });
    });

    group.bench_function("for loop twice", |b| {
        b.iter_with_large_drop(|| {
            let mut vec1 = Vec::with_capacity(vec.len() / 2 + 1);
            let mut vec2 = Vec::with_capacity(vec.len() / 3 + 1);

            for i in &vec {
                if i % 2 == 0 {
                    vec1.push(*i);
                }
            }
            for i in &vec {
                if i % 3 == 0 {
                    vec2.push(*i);
                }
            }

            (vec1, vec2)
        });
    });

    group.finish();
}

criterion_group!(benches, looping_tests);
criterion_main!(benches);
