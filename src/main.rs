use std::{
    collections::BTreeSet,
    fs::File,
    io::{BufRead, BufReader},
    time::Instant,
};

use acrostic_cw::{async_cw, Parameters, Solver};
use ascii::{AsciiChar, AsciiString};
use env_logger::Env;

fn main() {
    // env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let num = 6;
    // let file = File::open("dataset/google-10000-english-no-swears.txt").unwrap();
    let file = File::open("dataset/english3.txt").unwrap();

    let buf = BufReader::new(file);
    let mut bh = buf
        .lines()
        .filter_map(|l| l.ok().map(|s| s.to_lowercase()))
        .collect::<BTreeSet<String>>();
    // let file = File::open("google-10000-english-no-swears.txt").unwrap();
    let file = File::open("dataset/pokemon.txt").unwrap();
    let buf = BufReader::new(file);
    bh.extend(buf.lines().filter_map(|l| l.ok().map(|s| s.to_lowercase())));

    let dict = fst::Set::from_iter(bh.iter()).unwrap();
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

    let params = async_cw::Parameters::new(words);

    // println!("beginning solve");

    // for _ in 0..100 {
    let solver = async_cw::Solver::new(&params, &dict);

    let now = Instant::now();
    let solution = solver.run().unwrap();
    println!("elapsed: {:?}", now.elapsed());
    // }

    for word in params.word_to_board.iter() {
        let mut word_chars = String::new();
        for word_idx in word {
            assert_eq!(solution.board[word_idx].len(), 1);
            let char = solution.board[word_idx].iter().next().unwrap();
            word_chars.push(char);
        }
        println!("{}", word_chars);
    }
}
