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

    let params = async_cw::Parameters::new(words(4));
    group.bench_function("4x4", |b| {
        b.iter_batched(
            || async_cw::Solver::new_seed(&params, &dict, rand.generate()),
            |solver| solver.run().unwrap(),
            BatchSize::PerIteration,
        );
    });

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

criterion_group!(benches, async_cw_solve);
criterion_main!(benches);
