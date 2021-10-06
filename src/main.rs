use std::{
    collections::BTreeSet,
    fs::File,
    io::{stdin, stdout, BufRead, BufReader, Read, Write},
    time::Instant,
};

use acrostic_cw::{async_cw, Parameters, Solver};
use ascii::{AsciiChar, AsciiString};
use env_logger::Env;

fn pause() {
    let mut stdout = stdout();
    stdout.write_all(b"Press Enter to continue...").unwrap();
    stdout.flush().unwrap();
    stdin().read_exact(&mut [0]).unwrap();
}
fn main() {
    // pause();
    // env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let file =
        File::open("C:/Users/Jono/Development/Personal/acrostic-cw/dataset/english3.txt").unwrap();

    let buf = BufReader::new(file);
    let bh = buf
        .lines()
        .filter_map(|l| l.ok().map(|s| s.to_lowercase()))
        .collect::<BTreeSet<String>>();
    let dict = fst::Set::from_iter(bh.iter()).unwrap();
    let mut words = Vec::new();
    let num = 3;
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

    let params = async_cw::Parameters::new(words);

    // let mut group = c.benchmark_group("5x5");

    for seed in (0..usize::MAX).step_by(usize::MAX / 50) {
        let mut solver = async_cw::Solver::new_seed(&params, &dict, seed as u64);
        let now = Instant::now();
        let board = solver.run().unwrap();
        println!("{:?}", now.elapsed());
    }
    // group.bench_function(format!("solve: {}", seed), |b| {
    //     b.iter(|| {
    // println!("{:?}", board)
    //     })
    // });
    // group.finish();
}
