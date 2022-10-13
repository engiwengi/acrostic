use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs::{File, OpenOptions},
    hash::Hash,
    io::{stdin, stdout, BufRead, BufReader, BufWriter, Read, Write},
};

use acrostic_cw::async_cw;
use ustr::Ustr;

fn pause() {
    let mut stdout = stdout();
    stdout.write_all(b"Press Enter to continue...").unwrap();
    stdout.flush().unwrap();
    stdin().read_exact(&mut [0]).unwrap();
}
fn main() {
    // pause();
    let file = File::open(
        "C:/Users/Jono/Development/Personal/acrostic-cw/dataset/archive/unigram_freq_new.csv",
    )
    .unwrap();

    let file2 =
        File::open("C:/Users/Jono/Development/Personal/acrostic-cw/dataset/english3.txt").unwrap();

    let buf = BufReader::new(file);
    let buf2 = BufReader::new(file2);
    let bh = buf
        .lines()
        .filter_map(|l| {
            l.ok().and_then(|s| {
                s.split_once(',')
                    .map(|(s, i)| (s.to_owned(), i.parse::<usize>().unwrap()))
            })
        })
        .collect::<HashMap<String, usize>>();
    // let commonwords = buf2
    //     .lines()
    //     .filter_map(|l| l.ok())
    //     .collect::<BTreeSet<String>>();
    // let newfile = OpenOptions::new()
    //     .truncate(true)
    //     .write(true)
    //     .open("C:/Users/Jono/Development/Personal/acrostic-cw/dataset/archive/unigram_freq_new.csv")
    //     .unwrap();
    // let mut writer = BufWriter::new(newfile);
    // for (c, s) in bh
    //     .clone()
    //     .into_iter()
    //     .filter(|(s, _)| s.len() == 6 && commonwords.contains(s))
    //     .map(|(s, i)| (i, s))
    //     .collect::<BTreeMap<usize, String>>()
    // {
    //     writer
    //         .write_all(format!("{},{}\n", s, c).as_bytes())
    //         .unwrap();
    // }
    let dict = fst::Set::from_iter(bh.keys().collect::<BTreeSet<&String>>()).unwrap();
    let mut words = Vec::new();
    let num = 6;
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

    let mut solved_words: HashMap<Ustr, usize> = HashMap::new();
    let mut solutions: HashMap<Ustr, HashMap<Ustr, usize>> = HashMap::new();

    let step = usize::MAX / 50;

    let mut best_boards = BTreeMap::new();
    println!("{}", step);

    for seed in (0..usize::MAX).step_by(step) {
        let solver = async_cw::Solver::new_seed(&params, &dict, seed as u64);
        if let Ok(board) = solver.run() {
            let mut score = 0;

            for word in board.iter() {
                *solved_words.entry(*word.0).or_default() += 1;
                score += (*bh.get(word.0.as_str()).unwrap_or(&0) as f32)
                    .log2()
                    .round() as u32;
                solutions.insert(*word.0, board.clone());
            }

            println!("score: {}\nboard: {:?}", score, board);
            best_boards.insert(score, board);
        }
    }

    for (score, board) in best_boards {
        println!("score: {}\nboard: {:?}", score, board);
    }
}
