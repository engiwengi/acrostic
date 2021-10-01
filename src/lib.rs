#![feature(option_result_contains)]
#![feature(iter_map_while)]

use std::{
    borrow::Borrow, collections::HashSet, convert::TryFrom, str::Utf8Error, thread::LocalKey,
};

use ascii::{AsciiChar, AsciiStr, AsciiString};
use daggy::petgraph::Directed;
use dawg::{WildCardStr, WildCardString};
use fst::{set::StreamBuilder, Automaton, IntoStreamer, Set, Streamer};
use radix_trie::{Trie, TrieKey};
use rand::{prelude::ThreadRng, thread_rng};

mod cw;
mod dawg;

pub use cw::*;

pub struct CrosswordCompiler {
    size: usize,
    rand: ThreadRng,
    dict: Set<Vec<u8>>,
    board: Vec<Square>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Square {
    Shaded,
    Blank,
    Character(AsciiChar),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Location {
    x: usize,
    y: usize,
}

pub struct Crossword {
    words: Vec<Word>,
}

pub struct Word {
    direction: Direction,
    solution: AsciiString,
    location: Location,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Across,
    Down,
}

impl CrosswordCompiler {
    fn new(size: usize, dict: Set<Vec<u8>>) -> Self {
        Self {
            size,
            dict,
            rand: thread_rng(),
            board: vec![Square::Blank; size * size],
        }
    }

    fn get(&self, location: Location) -> &Square {
        &self.board[location.x + self.size * location.y]
    }

    fn get_mut(&mut self, location: Location) -> &mut Square {
        &mut self.board[location.x + self.size * location.y]
    }
    fn create(self) -> Crossword {
        todo!()
    }

    fn get_valid_stream_at(
        &self,
        mut location: Location,
        direction: Direction,
    ) -> StreamBuilder<'_, WildCardString> {
        let pattern = self.get_pattern_at(location, direction);

        self.dict.search(WildCardString::from(pattern))
    }

    fn get_pattern_at(&self, mut location: Location, direction: Direction) -> AsciiString {
        let mut wildmatch = AsciiString::new();

        loop {
            if location.x >= self.size || location.y >= self.size {
                break;
            }

            let square = self.get(location);

            match square {
                Square::Shaded => break,
                Square::Blank => wildmatch.push(AsciiChar::Question),
                Square::Character(char) => wildmatch.push(*char),
            }

            location = match direction {
                Direction::Across => Location {
                    x: location.x + 1,
                    y: location.y,
                },
                Direction::Down => Location {
                    x: location.x,
                    y: location.y + 1,
                },
            };
        }
        wildmatch
    }

    fn place_word_at(&mut self, mut location: Location, direction: Direction) -> Result<(), ()> {
        let stream = self.get_valid_stream_at(location, direction);
        let word = stream.into_stream().next().map(|a| a.to_owned());

        if let Some(word) = word {
            let mut idx = 0;
            loop {
                if location.x >= self.size || location.y >= self.size {
                    break;
                }

                if idx >= word.len() {
                    *self.get_mut(location) = Square::Shaded;
                    break;
                }

                let ascii_char = AsciiChar::from_ascii(word[idx]).unwrap();

                *self.get_mut(location) = Square::Character(ascii_char);

                location = match direction {
                    Direction::Across => Location {
                        x: location.x + 1,
                        y: location.y,
                    },
                    Direction::Down => Location {
                        x: location.x,
                        y: location.y + 1,
                    },
                };
                idx += 1;
            }
        } else {
            return Err(());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeSet, BinaryHeap},
        convert::TryFrom,
        fs::File,
        io::{BufRead, BufReader},
        time::Instant,
    };

    use fst::{Automaton, IntoStreamer, Streamer};
    use vec_map::VecMap;

    use crate::dawg::WildCardStr;

    #[test]
    fn random_string() {
        let file = File::open("google-10000-english-no-swears.txt").unwrap();

        let buf = BufReader::new(file);

        let construction = Instant::now();
        let mut vm: VecMap<Vec<String>> = VecMap::new();
        for str in buf.lines().filter_map(|l| l.ok()) {
            vm.entry(str.len())
                .or_insert_with(Default::default)
                .push(str);
        }

        vm.iter_mut().for_each(|(_, v)| v.sort_unstable());

        let elapsed_c = construction.elapsed();
        println!("time to construct: {:?}", elapsed_c);

        let size = vm.iter().fold(0usize, |acc, (_, s)| {
            acc + s.iter().map(|s| s.len()).sum::<usize>()
        });

        let pattern = "????a??d";

        println!("size: {}", size);
        let instant = Instant::now();
        let results: &[String] = &vm[pattern.len()];

        let wildmatch = wildmatch::WildMatch::new(pattern);

        let mut matches = Vec::new();
        for result in results {
            if wildmatch.matches(result) {
                matches.push(result.clone());
            }
        }

        let elapsed = instant.elapsed();
        println!("time to match: {:?}", elapsed);

        assert_eq!(matches, vec!["cabs", "cats"]);
    }

    #[test]
    fn containment() {
        let file = File::open("google-10000-english-no-swears.txt").unwrap();

        let buf = BufReader::new(file);
        let bh = buf
            .lines()
            .filter_map(|l| l.ok())
            .collect::<BTreeSet<String>>();

        let construction = Instant::now();
        let set = fst::Set::from_iter(bh.iter()).unwrap();
        // let map = fst::Map::from_iter(bh.iter().enumerate().map(|(i, s)| (s, i as u64))).unwrap();

        let elapsed_c = construction.elapsed();
        println!("time to construct: {:?}", elapsed_c);
        println!("len: {}", set.len());

        // println!("set size: {}", set.as_fst().as_bytes().len());

        let instant = Instant::now();

        let pattern = "????a??d";

        let results = set
            .search(WildCardStr::try_from(pattern).unwrap().starts_with())
            .into_stream()
            .into_strs()
            .unwrap();

        let elapsed = instant.elapsed();
        println!("time to match: {:?}", elapsed);

        assert_eq!(results, vec!["test"]);
    }
}
