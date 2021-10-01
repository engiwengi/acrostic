use std::{
    collections::BTreeSet,
    fs::File,
    io::{BufRead, BufReader},
};

use acrostic_cw::{Parameters, Solver};
use ascii::{AsciiChar, AsciiString};
use env_logger::Env;

fn main() {
    // env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let num = 5;
    let file = File::open("english3.txt").unwrap();

    let buf = BufReader::new(file);
    let mut bh = buf
        .lines()
        .filter_map(|l| l.ok().map(|s| s.to_lowercase()))
        .collect::<BTreeSet<String>>();
    // let file = File::open("google-10000-english-no-swears.txt").unwrap();
    let file = File::open("pokemon.txt").unwrap();
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

    let params = Parameters::new(words, dict);

    println!("beginning solve");

    let solver = Solver::new_seed(&params, 10289213);

    let solution = solver.solve().unwrap();

    for word in params.word_to_board.iter() {
        let mut word_chars = AsciiString::new();
        for word_idx in word {
            assert_eq!(solution[word_idx].len(), 1);
            let char = solution[word_idx].iter().next().unwrap();
            word_chars.push(AsciiChar::from_ascii(char as u8).unwrap());
        }
        println!("{}", word_chars);
    }
}
