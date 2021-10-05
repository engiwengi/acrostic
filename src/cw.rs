use std::{
    borrow::Borrow,
    cmp::Ordering,
    collections::{BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque},
    fmt::Debug,
};

use ascii::{AsciiChar, AsciiStr, AsciiString};
use enumset::{EnumSet, EnumSetType};
use log::{debug, info};
use nanorand::{Rng, WyRand};

use fst::{set::StreamBuilder, Automaton, IntoStreamer, Set, Streamer};
use tinyset::SetUsize;
use vec_map::VecMap;

use crate::dawg::{CharAt, Intersections, Length, VaguePattern};

pub struct Parameters {
    pub word_to_board: Vec<Vec<usize>>,
    board_to_word: VecMap<(usize, Option<usize>)>,
    dict: Set<Vec<u8>>,
    preferred_words: HashSet<AsciiString>,
}

pub struct Solver<'a> {
    parameters: &'a Parameters,
    board: VecMap<EnumSet<AsciiCharInternal>>,
    commands: VecDeque<Command>,
    command: Option<Command>,
    history: Vec<History>,
    rng: WyRand,
    words: HashMap<AsciiString, usize>,
    attempted_words: VecMap<HashSet<AsciiString>>,
    words_to_minimize: SetUsize,
}

#[derive(Debug)]
enum History {
    SelectedWord {
        attempted_word: AsciiString,
        word_idx: usize,
    },
    SelectedLetter {
        board_idx: usize,
        chosen_letter: AsciiCharInternal,
        previous_state: EnumSet<AsciiCharInternal>,
    },
    AddedWord {
        word: AsciiString,
    },
    Restricted {
        board_idx: usize,
        previous_state: EnumSet<AsciiCharInternal>,
    },
    Begin,
}

#[derive(Debug)]
enum Command {
    Minimize(usize),     // usize is the index of the word to minimize
    SelectLetter(usize), // usize is the index of the board to select
    BackTrack,
    NoSolution,
    SelectWord(usize),
}

struct ContinueParam {
    word_idx: usize,
    remaining_letters: usize,
    tiles: usize,
}

impl ContinueParam {
    fn letters_per_tile(&self) -> f32 {
        (self.remaining_letters as f32) / (self.tiles as f32)
    }
}

impl<'a> Solver<'a> {
    pub fn new(parameters: &'a Parameters) -> Self {
        Self {
            parameters,
            board: VecMap::new(),
            commands: VecDeque::new(),
            command: None,
            history: Vec::new(),
            rng: WyRand::new(),
            words: HashMap::new(),
            attempted_words: VecMap::new(),
            words_to_minimize: SetUsize::new(),
        }
    }

    pub fn new_seed(parameters: &'a Parameters, seed: u64) -> Self {
        Self {
            parameters,
            board: VecMap::new(),
            commands: VecDeque::new(),
            history: Vec::new(),
            rng: WyRand::new_seed(seed),
            words: HashMap::new(),
            attempted_words: VecMap::new(),
            words_to_minimize: SetUsize::new(),
            command: None,
        }
    }

    #[inline(always)]
    pub fn solve(mut self) -> Result<VecMap<EnumSet<AsciiCharInternal>>, ()> {
        let words_by_constraint = self.order_words_by_constraints();

        self.history.push(History::Begin);

        if let Some(idx) = words_by_constraint.peek() {
            self.words_to_minimize.insert(idx.index);
            // self.commands.push_back(Command::Minimize(idx.index));
        }

        while let Some(cont_param) = self.run() {
            // println!("letters per tile: {}", cont_param.letters_per_tile());
            self.command = Some(Command::SelectWord(cont_param.word_idx));
            // self.commands
            //     .push_back(Command::SelectWord(cont_param.word_idx));
            if !self.words.is_empty() {
                info!("Current words: {:?}", self.words);
            }
        }

        if self.words.len() != self.parameters.word_to_board.len() {
            Err(())
        } else {
            Ok(self.board)
        }
    }

    #[inline(always)]
    fn run(&mut self) -> Option<ContinueParam> {
        loop {
            // while let Some(command) = self.commands.pop_front() {
            //     // println!("executing command: {:?}", command);
            //     match command {
            //         Command::Minimize(idx) => {}
            //         Command::BackTrack => self.back_track(),
            //         Command::SelectLetter(idx) => self.select_letter(idx),
            //         Command::NoSolution => return None,
            //         Command::SelectWord(idx) => self.select_word(idx),
            //     }
            // }

            while let Some(command) = self.command.take() {
                match command {
                    Command::Minimize(idx) => {}
                    Command::BackTrack => self.back_track(),
                    Command::SelectLetter(idx) => self.select_letter(idx),
                    Command::NoSolution => return None,
                    Command::SelectWord(idx) => self.select_word(idx),
                }
            }

            let word_idx = match self.words_to_minimize.iter().next() {
                Some(word_idx) => word_idx,
                None => break,
            };
            self.words_to_minimize.remove(word_idx);

            self.restrict(word_idx);
        }

        self.parameters.word_to_board.iter().enumerate().fold(
            None,
            |continue_param: Option<ContinueParam>, (word_idx, word)| {
                let c = ContinueParam {
                    word_idx,
                    remaining_letters: word.iter().fold(0, |remaining, board_idx| {
                        let add = self
                            .board
                            .entry(*board_idx)
                            .or_insert_with(EnumSet::all)
                            .len();
                        if add == 0 {
                            println!("tile with zero remaining positions")
                        }
                        remaining + add
                    }),
                    tiles: word.len(),
                };

                if c.letters_per_tile() > 1.0 {
                    if let Some(cont) = &continue_param {
                        if cont.letters_per_tile() < c.letters_per_tile() {
                            return continue_param;
                        }
                    }
                    Some(c)
                } else {
                    continue_param
                }
            },
        )
        // self.board
        //     .iter()
        //     .fold(None, |acc, (board_idx, remaining_letters)| {
        //         if remaining_letters.len() > 1 {
        //             if let Some(p) = &acc {
        //                 if p.remaining_letters_per_tile < remaining_letters.len() {
        //                     return acc;
        //                 }
        //             }
        //             Some(ContinueParam {
        //                 board_idx,
        //                 remaining_letters: remaining_letters.len(),
        //             })
        //         } else {
        //             acc
        //         }
        //     })
    }

    #[inline(always)]
    fn select_word(&mut self, idx: usize) {
        info!("selecting random word for board index: {}", idx);
        // println!("selecting word for board index: {}", idx);

        let word: &[usize] = &self.parameters.word_to_board[idx];

        let mut automata = Vec::new();

        for board_idx in word.iter() {
            if let Some(allowed_chars) = self.board.get(*board_idx) {
                // println!("allowed chars: {:?} at {:?}", allowed_chars, char_idx);
                automata.push(Some(*allowed_chars));
            } else {
                automata.push(None);
            }
        }

        let mut possible_words = Vec::<AsciiString>::new();

        let mut preferred_words = Vec::<AsciiString>::new();

        let mut stream = self
            .parameters
            .dict
            .search(VaguePattern::new(&automata))
            .into_stream();

        while let Some(word_bytes) = stream.next() {
            if let Ok(ascii_word) = AsciiStr::from_ascii(word_bytes) {
                if !self
                    .attempted_words
                    .entry(idx)
                    .or_insert_with(HashSet::new)
                    .contains(ascii_word)
                {
                    if self.parameters.preferred_words.contains(ascii_word) {
                        preferred_words.push(ascii_word.to_owned());
                    } else {
                        possible_words.push(ascii_word.to_owned());
                    }
                }
            }
        }

        info!("Possible words len: {}", possible_words.len());

        if possible_words.is_empty() && preferred_words.is_empty() {
            self.attempted_words.remove(idx);
            // self.commands.push_front(Command::BackTrack);
            self.command = Some(Command::BackTrack);
            return;
        }

        let chosen_word;

        if !preferred_words.is_empty() {
            let i = self.rng.generate_range(0..preferred_words.len());

            chosen_word = preferred_words.swap_remove(i);
        } else {
            let i = self.rng.generate_range(0..possible_words.len());

            chosen_word = possible_words.swap_remove(i);
        }

        let mut chars = VecMap::new();
        for (char_idx, ascii_char) in chosen_word.chars().enumerate() {
            chars
                .entry(char_idx)
                .or_insert(EnumSet::new())
                .insert(AsciiCharInternal::from(ascii_char));
        }

        info!("Chosen word: {}", chosen_word);

        if let Some(existing) = self.words.get(&chosen_word) {
            if *existing != idx {
                info!("Chosen word existed at board idx: {}", existing);
                // println!("duplicate word found: {}", chosen_word);
                self.command = Some(Command::BackTrack);
                // self.commands.push_front(Command::BackTrack);
                return;
            } else {
                info!("error: should not happen");
            }
        } else {
            self.words.insert(chosen_word.clone(), idx);
            self.history.push(History::SelectedWord {
                attempted_word: chosen_word.clone(),
                word_idx: idx,
            });
            self.history.push(History::AddedWord { word: chosen_word });
        }

        for (char_idx, set) in chars.into_iter().map(|(_, v)| v).enumerate() {
            let restricted_chars = self.board.entry(word[char_idx]).or_insert(EnumSet::all());
            if !restricted_chars.is_subset(set) {
                self.history.push(History::Restricted {
                    board_idx: word[char_idx],
                    previous_state: *restricted_chars,
                });

                *restricted_chars = restricted_chars.intersection(set);

                // println!(
                //     "board index: {}, restricted chars: \n{:?}",
                //     word[char_idx], restricted_chars
                // );

                if restricted_chars.is_empty() {
                    // println!("no overlap in existing restricted chars");
                    // self.commands.push_front(Command::BackTrack);
                    self.command = Some(Command::BackTrack);
                    return;
                } else if let Some(word_idx) = match self.parameters.board_to_word[word[char_idx]] {
                    (_, Some(word_idx)) if word_idx != idx => Some(word_idx),
                    (word_idx, _) if word_idx != idx => Some(word_idx),
                    _ => None,
                } {
                    info!("Propogating minimization to word: {}", word_idx);
                    // println!("propogating minimization");
                    self.words_to_minimize.insert(word_idx);
                    // self.commands.push_back(Command::Minimize(word_idx));
                }
            }
        }
    }

    #[inline(always)]
    fn back_track(&mut self) {
        // println!("backtracking");
        // self.commands.clear();
        let _ = std::mem::take(&mut self.words_to_minimize);
        info!("backtracking");
        // println!("history len: {}", self.history.len());
        // println!("history: {:?}", self.history);

        while let Some(history) = self.history.pop() {
            match history {
                History::SelectedLetter {
                    board_idx,
                    chosen_letter,
                    mut previous_state,
                } => {
                    if previous_state.len() > 1 {
                        previous_state.remove(chosen_letter);
                        self.board[board_idx] = previous_state;
                        // self.commands.push_back(Command::SelectLetter(board_idx));
                        self.command = Some(Command::SelectLetter(board_idx));
                        break;
                    } else {
                        self.board[board_idx] = previous_state;
                    }
                }
                History::Restricted {
                    board_idx,
                    previous_state,
                } => {
                    self.board[board_idx] = previous_state;
                }
                History::AddedWord { word } => {
                    self.words.remove(&word);
                }
                History::Begin => self.command = Some(Command::NoSolution),
                History::SelectedWord {
                    attempted_word,
                    word_idx,
                } => {
                    self.attempted_words
                        .entry(word_idx)
                        .or_insert_with(HashSet::new)
                        .insert(attempted_word);
                    break;
                }
            }
        }
    }

    #[inline(always)]
    fn select_letter(&mut self, idx: usize) {
        info!("selecting letter for board index: {}", idx);
        let allowed_letters = &mut self.board[idx];

        let i = self.rng.generate_range(0..allowed_letters.len());
        let char = allowed_letters.iter().nth(i).unwrap();
        // println!("allowed letters: \n{:?}", allowed_letters);
        // println!("letter selected: {:?}", char);

        self.history.push(History::SelectedLetter {
            board_idx: idx,
            previous_state: *allowed_letters,
            chosen_letter: char,
        });

        *allowed_letters = EnumSet::from(char);
        // allowed_letters.remove(char);

        let (word1, word2) = self.parameters.board_to_word[idx];

        self.words_to_minimize.insert(word1);

        // self.commands.push_back(Command::Minimize(word1));
        if let Some(word2) = word2 {
            self.words_to_minimize.insert(word2);
            // self.commands.push_back(Command::Minimize(word2));
        }
    }

    #[inline(always)]
    fn restrict(&mut self, idx: usize) {
        info!("Starting minimization for word: {}", idx);
        let word = self.parameters.word_to_board.get(idx).unwrap();

        let mut automata = Vec::new();

        for board_idx in word.iter() {
            if let Some(allowed_chars) = self.board.get(*board_idx) {
                // println!("allowed chars: {:?} at {:?}", allowed_chars, char_idx);
                automata.push(Some(*allowed_chars));
            } else {
                automata.push(None);
            }
        }

        let mut possible_words = self
            .parameters
            .dict
            .search(VaguePattern::new(&automata))
            .into_stream();

        let mut chars = VecMap::new();

        let mut word_count = 0;
        let mut chosen_word = AsciiString::new();

        while let Some(next_word) = possible_words.next() {
            word_count += 1;
            if word_count == 1 {
                chosen_word.extend(AsciiStr::from_ascii(next_word).unwrap());
            }
            for (char_idx, char) in next_word.iter().enumerate() {
                let ascii_char = AsciiChar::from_ascii(*char).unwrap();
                chars
                    .entry(char_idx)
                    .or_insert(EnumSet::new())
                    .insert(AsciiCharInternal::from(ascii_char));
            }
        }

        if word_count == 1 {
            info!("Only 1 available word for word idx: {}", idx);
            if let Some(existing) = self.words.get(&chosen_word) {
                if *existing != idx {
                    info!(
                        "Chosen word {} is duplicated by word idx: {}",
                        chosen_word, existing
                    );
                    // println!("duplicate word found: {}", chosen_word);
                    // self.commands.push_front(Command::BackTrack);
                    self.command = Some(Command::BackTrack);
                    return;
                } else {
                    // println!("error: should not happen");
                }
            } else {
                info!("Chosen word: {}", chosen_word);
                self.words.insert(chosen_word.clone(), idx);
                self.history.push(History::AddedWord { word: chosen_word });
            }
        }

        if chars.is_empty() {
            info!("No available words for word idx: {}", idx);
            // println!("no possible words available");
            // self.commands.push_front(Command::BackTrack);
            self.command = Some(Command::BackTrack);
            return;
        }

        for (char_idx, set) in chars.into_iter().map(|(_, v)| v).enumerate() {
            let restricted_chars = self.board.entry(word[char_idx]).or_insert(EnumSet::all());
            if !restricted_chars.is_subset(set) {
                self.history.push(History::Restricted {
                    board_idx: word[char_idx],
                    previous_state: *restricted_chars,
                });

                *restricted_chars = restricted_chars.intersection(set);

                // println!(
                //     "board index: {}, restricted chars: \n{:?}",
                //     word[char_idx], restricted_chars
                // );

                if restricted_chars.is_empty() {
                    // println!("no overlap in existing restricted chars");
                    // self.commands.push_front(Command::BackTrack);
                    self.command = Some(Command::BackTrack);
                    return;
                } else if let Some(word_idx) = match self.parameters.board_to_word[word[char_idx]] {
                    (_, Some(word_idx)) if word_idx != idx => Some(word_idx),
                    (word_idx, _) if word_idx != idx => Some(word_idx),
                    _ => None,
                } {
                    info!("Propogating minimization to word: {}", word_idx);
                    self.words_to_minimize.insert(word_idx);
                    // self.commands.push_back(Command::Minimize(word_idx));
                }
            }
        }
    }

    #[inline(always)]
    fn order_words_by_constraints(&self) -> BinaryHeap<WordIndex> {
        self.parameters
            .word_to_board
            .iter()
            .enumerate()
            .map(|(word_idx, word)| WordIndex {
                index: word_idx,
                length: word.len(),
                constraint: word.iter().fold(0, |acc, board_idx| {
                    acc + self
                        .parameters
                        .board_to_word
                        .get(*board_idx)
                        .unwrap()
                        .1
                        .is_some() as usize
                }),
            })
            .collect::<BinaryHeap<WordIndex>>()
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
struct WordIndex {
    index: usize,
    length: usize,
    constraint: usize,
}

impl PartialOrd for WordIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.constraint
            .partial_cmp(&other.constraint)
            .and_then(|ord| {
                self.length
                    .partial_cmp(&other.length)
                    .map(|ord_len| ord.then(ord_len))
            })
    }
}

impl Ord for WordIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.constraint
            .cmp(&other.constraint)
            .then(self.length.cmp(&other.length))
    }
}

impl Parameters {
    pub fn new(words: Vec<Vec<usize>>, dict: Set<Vec<u8>>) -> Self {
        Self::with_preferred(words, dict, HashSet::new())
    }
    pub fn with_preferred(
        words: Vec<Vec<usize>>,
        dict: Set<Vec<u8>>,
        preferred_words: HashSet<AsciiString>,
    ) -> Self {
        let mut board_to_word = VecMap::new();
        for (word_idx, word) in words.iter().enumerate() {
            for &board_idx in word {
                let entry = board_to_word.entry(board_idx);
                match entry {
                    vec_map::Entry::Vacant(entry) => {
                        entry.insert((word_idx, None));
                    }
                    vec_map::Entry::Occupied(mut entry) => {
                        entry.get_mut().1 = Some(word_idx);
                    }
                }
            }
        }
        Self {
            word_to_board: words,
            board_to_word,
            dict,
            preferred_words,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufRead, BufReader},
        time::Instant,
    };

    use ascii::AsciiString;
    use env_logger::Env;

    use super::*;
    fn get_set() -> Set<Vec<u8>> {
        let file = File::open("dataset/english3.txt").unwrap();

        let mut rng = WyRand::new();

        let buf = BufReader::new(file);
        let mut bh = buf
            .lines()
            .filter_map(|l| l.ok().map(|s| s.to_lowercase()))
            .collect::<BTreeSet<String>>();
        // let file = File::open("google-10000-english-no-swears.txt").unwrap();
        let file = File::open("dataset/pokemon.txt").unwrap();

        let buf = BufReader::new(file);
        // bh.extend(buf.lines().filter_map(|l| l.ok().map(|s| s.to_lowercase())));

        let set = fst::Set::from_iter(bh.iter()).unwrap();
        set
    }

    fn solve(params: &Parameters) {
        let solver = Solver::new(&params);

        let instant = Instant::now();
        // println!("beginning solve");

        let solution = solver.solve().unwrap();
        let elapsed = instant.elapsed();
        println!("time: {:?}", elapsed);

        // for word in params.word_to_board.iter() {
        //     let mut word_chars = AsciiString::new();
        //     for word_idx in word {
        //         assert_eq!(solution[word_idx].len(), 1);
        //         let char = solution[word_idx].iter().next().unwrap();
        //         word_chars.push(AsciiChar::from_ascii(char as u8).unwrap());
        //     }
        //     println!("{}", word_chars);
        // }
    }
    #[test]
    fn square_all_blank() {
        // env_logger::builder().is_test(true).try_init();
        for _ in 0..100 {
            for num in 5..6 {
                let dict = get_set();
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

                solve(&params);
            }
        }
    }

    #[test]
    fn british_style() {
        let dict = get_set();

        let words = vec![
            vec![0, 1, 2],
            vec![0, 4, 11, 15],
            vec![2, 5, 12, 17, 24, 30, 37, 45, 52],
            vec![15, 16, 17],
            vec![5, 6, 7, 8, 9],
            vec![3, 10, 14, 23],
            vec![18, 19, 20, 21, 22, 23],
            vec![28, 29, 30],
            vec![7, 13, 18, 25, 31, 39, 46, 54],
            vec![20, 26, 32, 41],
            vec![22, 27, 33, 43, 47, 58, 63, 71],
            vec![33, 34, 35],
            vec![28, 36, 44, 50],
            vec![37, 38, 39, 40, 41, 42, 43],
            vec![47, 49, 49],
            vec![50, 51, 52, 53, 54, 55, 56, 57, 58],
            vec![63, 64, 65],
            vec![55, 62, 70],
            vec![53, 61, 68],
            vec![66, 67, 68, 69, 70],
            vec![51, 60, 66],
            vec![49, 59, 65, 72],
        ];

        let params = Parameters::new(words, dict);

        solve(&params);
    }

    #[test]
    fn pokemon() {
        let dict = get_set();

        #[rustfmt::skip]
        let grid = vec![
            0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,
            1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,
            1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,
            1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,1,
            1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,
            1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,
            1,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,
            0,0,1,1,1,1,0,1,0,1,1,1,1,1,1,0,
            0,1,1,1,1,1,1,0,1,0,1,1,1,1,0,0,
            1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,1,
            1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,
            1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,
            1,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,
            0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,
            0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,
            1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,
        ];

        // #[rustfmt::skip]
        // let grid = vec![
        //     0,1,1,1,1,1,0,
        //     1,0,0,0,1,0,1,
        //     1,1,1,1,1,0,1,
        //     1,0,1,0,1,0,1,
        //     1,0,1,1,1,1,1,
        //     1,0,1,0,0,0,1,
        //     0,1,1,1,1,1,0,
        // ];

        let words = words_from_grid(grid, 16, 16);

        println!("words: {:?}", words);

        let mut vecmap = VecMap::new();

        for word in dict.range().into_stream().into_strs().unwrap() {
            *vecmap.entry(word.len()).or_insert(0) += 1;
        }

        println!("{:?}", vecmap);

        let mut vecmap = VecMap::new();

        for word in words.iter() {
            *vecmap.entry(word.len()).or_insert(0) += 1;
        }

        println!("{:?}", vecmap);

        let file = File::open("pokemon.txt").unwrap();
        let buf = BufReader::new(file);
        let bh = buf
            .lines()
            .filter_map(|l| l.ok().map(|s| s.to_lowercase()))
            .collect::<BTreeSet<String>>();

        let fst = fst::Set::from_iter(bh.iter()).unwrap();

        let mut stream = fst.into_stream();

        let mut pokemon = HashSet::new();

        while let Some(word) = stream.next() {
            if let Ok(ascii_word) = AsciiStr::from_ascii(word) {
                pokemon.insert(ascii_word.to_owned());
            }
        }

        let params = Parameters::with_preferred(words, dict, pokemon.clone());

        let mut best_count = 0;

        for _ in 0..1000 {
            let solver = Solver::new(&params);

            let solution = match solver.solve() {
                Ok(s) => s,
                Err(_) => {
                    println!("no solution");
                    continue;
                }
            };
            let mut count = 0;
            let mut strings = Vec::new();
            for word in params.word_to_board.iter() {
                let mut word_chars = AsciiString::new();
                for word_idx in word {
                    assert_eq!(solution[word_idx].len(), 1);
                    let char = solution[word_idx].iter().next().unwrap();
                    word_chars.push(AsciiChar::from_ascii(char as u8).unwrap());
                }
                count += pokemon.contains(&word_chars) as usize;

                strings.push(word_chars);
            }
            if count >= best_count {
                best_count = count;
                println!("new best: {}", count);
                strings.iter().for_each(|s| println!("{}", s));
            } else {
                println!("only got {}", count);
                // strings.iter().for_each(|s| println!("{}", s));
            }
        }
    }

    #[test]
    fn american_style() {
        let dict = get_set();

        let words = vec![
            vec![0, 1, 2, 3, 4, 5],
            vec![8, 9, 10, 11],
            vec![12, 13, 14, 15, 16, 17],
            vec![18, 19, 20, 21, 22, 23],
            vec![24, 25, 26, 27],
            vec![30, 31, 32, 33, 34, 35],
            vec![0, 6, 12, 18, 24, 30],
            vec![13, 19, 25, 31],
            vec![2, 8, 14, 20, 26, 32],
            vec![3, 9, 15, 21, 27, 33],
            vec![4, 10, 16, 22],
            vec![5, 11, 17, 23, 29, 35],
        ];

        let params = Parameters::new(words, dict);

        solve(&params);
    }

    fn words_from_grid(grid: Vec<usize>, width: usize, height: usize) -> Vec<Vec<usize>> {
        let mut words = Vec::new();

        let mut current_word = Vec::new();

        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                if grid[index] > 0 {
                    current_word.push(index);
                } else {
                    let finished_word = std::mem::take(&mut current_word);
                    if finished_word.len() > 1 {
                        words.push(finished_word);
                    }
                }
            }
            let finished_word = std::mem::take(&mut current_word);
            if finished_word.len() > 1 {
                words.push(finished_word);
            }
        }

        for x in 0..width {
            for y in 0..height {
                let index = y * width + x;
                if grid[index] > 0 {
                    current_word.push(index);
                } else {
                    let finished_word = std::mem::take(&mut current_word);
                    if finished_word.len() > 1 {
                        words.push(finished_word);
                    }
                }
            }
            let finished_word = std::mem::take(&mut current_word);
            if finished_word.len() > 1 {
                words.push(finished_word);
            }
        }

        words
    }

    #[test]
    fn ascii_char_internal() {
        let a = AsciiChar::A;
        let b = AsciiChar::b;
        let c = AsciiChar::C;
        let tab = AsciiChar::Tab;
        let nul = AsciiChar::Null;
        let del = AsciiChar::DEL;
        let crg = AsciiChar::CarriageReturn;

        assert_eq!(AsciiCharInternal::A, AsciiCharInternal::from(a));
        assert_eq!(AsciiCharInternal::b, AsciiCharInternal::from(b));
        assert_eq!(AsciiCharInternal::C, AsciiCharInternal::from(c));
        assert_eq!(AsciiCharInternal::Tab, AsciiCharInternal::from(tab));
        assert_eq!(AsciiCharInternal::Null, AsciiCharInternal::from(nul));
        assert_eq!(AsciiCharInternal::DEL, AsciiCharInternal::from(del));
        assert_eq!(
            AsciiCharInternal::CarriageReturn,
            AsciiCharInternal::from(crg)
        );
    }
}

/// An ASCII character. It wraps a `u8`, with the highest bit always zero.
#[derive(PartialOrd, Ord, EnumSetType, Debug)]
#[repr(u8)]
#[allow(non_camel_case_types)]
#[allow(clippy::upper_case_acronyms)]
pub enum AsciiCharInternal {
    /// `'\0'`
    Null = 0,
    /// [Start Of Heading](http://en.wikipedia.org/wiki/Start_of_Heading)
    SOH = 1,
    /// [Start Of teXt](http://en.wikipedia.org/wiki/Start_of_Text)
    SOX = 2,
    /// [End of TeXt](http://en.wikipedia.org/wiki/End-of-Text_character)
    ETX = 3,
    /// [End Of Transmission](http://en.wikipedia.org/wiki/End-of-Transmission_character)
    EOT = 4,
    /// [Enquiry](http://en.wikipedia.org/wiki/Enquiry_character)
    ENQ = 5,
    /// [Acknowledgement](http://en.wikipedia.org/wiki/Acknowledge_character)
    ACK = 6,
    /// [bell / alarm / audible](http://en.wikipedia.org/wiki/Bell_character)
    ///
    /// `'\a'` is not recognized by Rust.
    Bell = 7,
    /// [Backspace](http://en.wikipedia.org/wiki/Backspace)
    ///
    /// `'\b'` is not recognized by Rust.
    BackSpace = 8,
    /// `'\t'`
    Tab = 9,
    /// `'\n'`
    LineFeed = 10,
    /// [Vertical tab](http://en.wikipedia.org/wiki/Vertical_Tab)
    ///
    /// `'\v'` is not recognized by Rust.
    VT = 11,
    /// [Form Feed](http://en.wikipedia.org/wiki/Form_Feed)
    ///
    /// `'\f'` is not recognized by Rust.
    FF = 12,
    /// `'\r'`
    CarriageReturn = 13,
    /// [Shift In](http://en.wikipedia.org/wiki/Shift_Out_and_Shift_In_characters)
    SI = 14,
    /// [Shift Out](http://en.wikipedia.org/wiki/Shift_Out_and_Shift_In_characters)
    SO = 15,
    /// [Data Link Escape](http://en.wikipedia.org/wiki/Data_Link_Escape)
    DLE = 16,
    /// [Device control 1, often XON](http://en.wikipedia.org/wiki/Device_Control_1)
    DC1 = 17,
    /// Device control 2
    DC2 = 18,
    /// Device control 3, Often XOFF
    DC3 = 19,
    /// Device control 4
    DC4 = 20,
    /// [Negative AcKnowledgement](http://en.wikipedia.org/wiki/Negative-acknowledge_character)
    NAK = 21,
    /// [Synchronous idle](http://en.wikipedia.org/wiki/Synchronous_Idle)
    SYN = 22,
    /// [End of Transmission Block](http://en.wikipedia.org/wiki/End-of-Transmission-Block_character)
    ETB = 23,
    /// [Cancel](http://en.wikipedia.org/wiki/Cancel_character)
    CAN = 24,
    /// [End of Medium](http://en.wikipedia.org/wiki/End_of_Medium)
    EM = 25,
    /// [Substitute](http://en.wikipedia.org/wiki/Substitute_character)
    SUB = 26,
    /// [Escape](http://en.wikipedia.org/wiki/Escape_character)
    ///
    /// `'\e'` is not recognized by Rust.
    ESC = 27,
    /// [File Separator](http://en.wikipedia.org/wiki/File_separator)
    FS = 28,
    /// [Group Separator](http://en.wikipedia.org/wiki/Group_separator)
    GS = 29,
    /// [Record Separator](http://en.wikipedia.org/wiki/Record_separator)
    RS = 30,
    /// [Unit Separator](http://en.wikipedia.org/wiki/Unit_separator)
    US = 31,
    /// `' '`
    Space = 32,
    /// `'!'`
    Exclamation = 33,
    /// `'"'`
    Quotation = 34,
    /// `'#'`
    Hash = 35,
    /// `'$'`
    Dollar = 36,
    /// `'%'`
    Percent = 37,
    /// `'&'`
    Ampersand = 38,
    /// `'\''`
    Apostrophe = 39,
    /// `'('`
    ParenOpen = 40,
    /// `')'`
    ParenClose = 41,
    /// `'*'`
    Asterisk = 42,
    /// `'+'`
    Plus = 43,
    /// `','`
    Comma = 44,
    /// `'-'`
    Minus = 45,
    /// `'.'`
    Dot = 46,
    /// `'/'`
    Slash = 47,
    /// `'0'`
    _0 = 48,
    /// `'1'`
    _1 = 49,
    /// `'2'`
    _2 = 50,
    /// `'3'`
    _3 = 51,
    /// `'4'`
    _4 = 52,
    /// `'5'`
    _5 = 53,
    /// `'6'`
    _6 = 54,
    /// `'7'`
    _7 = 55,
    /// `'8'`
    _8 = 56,
    /// `'9'`
    _9 = 57,
    /// `':'`
    Colon = 58,
    /// `';'`
    Semicolon = 59,
    /// `'<'`
    LessThan = 60,
    /// `'='`
    Equal = 61,
    /// `'>'`
    GreaterThan = 62,
    /// `'?'`
    Question = 63,
    /// `'@'`
    At = 64,
    /// `'A'`
    A = 65,
    /// `'B'`
    B = 66,
    /// `'C'`
    C = 67,
    /// `'D'`
    D = 68,
    /// `'E'`
    E = 69,
    /// `'F'`
    F = 70,
    /// `'G'`
    G = 71,
    /// `'H'`
    H = 72,
    /// `'I'`
    I = 73,
    /// `'J'`
    J = 74,
    /// `'K'`
    K = 75,
    /// `'L'`
    L = 76,
    /// `'M'`
    M = 77,
    /// `'N'`
    N = 78,
    /// `'O'`
    O = 79,
    /// `'P'`
    P = 80,
    /// `'Q'`
    Q = 81,
    /// `'R'`
    R = 82,
    /// `'S'`
    S = 83,
    /// `'T'`
    T = 84,
    /// `'U'`
    U = 85,
    /// `'V'`
    V = 86,
    /// `'W'`
    W = 87,
    /// `'X'`
    X = 88,
    /// `'Y'`
    Y = 89,
    /// `'Z'`
    Z = 90,
    /// `'['`
    BracketOpen = 91,
    /// `'\'`
    BackSlash = 92,
    /// `']'`
    BracketClose = 93,
    /// `'_'`
    Caret = 94,
    /// `'_'`
    UnderScore = 95,
    /// `'`'`
    Grave = 96,
    /// `'a'`
    a = 97,
    /// `'b'`
    b = 98,
    /// `'c'`
    c = 99,
    /// `'d'`
    d = 100,
    /// `'e'`
    e = 101,
    /// `'f'`
    f = 102,
    /// `'g'`
    g = 103,
    /// `'h'`
    h = 104,
    /// `'i'`
    i = 105,
    /// `'j'`
    j = 106,
    /// `'k'`
    k = 107,
    /// `'l'`
    l = 108,
    /// `'m'`
    m = 109,
    /// `'n'`
    n = 110,
    /// `'o'`
    o = 111,
    /// `'p'`
    p = 112,
    /// `'q'`
    q = 113,
    /// `'r'`
    r = 114,
    /// `'s'`
    s = 115,
    /// `'t'`
    t = 116,
    /// `'u'`
    u = 117,
    /// `'v'`
    v = 118,
    /// `'w'`
    w = 119,
    /// `'x'`
    x = 120,
    /// `'y'`
    y = 121,
    /// `'z'`
    z = 122,
    /// `'{'`
    CurlyBraceOpen = 123,
    /// `'|'`
    VerticalBar = 124,
    /// `'}'`
    CurlyBraceClose = 125,
    /// `'~'`
    Tilde = 126,
    /// [Delete](http://en.wikipedia.org/wiki/Delete_character)
    DEL = 127,
}

impl From<AsciiChar> for AsciiCharInternal {
    fn from(a: AsciiChar) -> Self {
        unsafe { std::mem::transmute(a) }
    }
}
