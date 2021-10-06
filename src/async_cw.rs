use std::{
    cmp::Ordering,
    collections::HashSet,
    fmt::{Debug, Write},
    iter::FromIterator,
};

use bevy_tasks::{AsyncComputeTaskPool, TaskPool, TaskPoolBuilder};
use crossbeam::channel::{Receiver, Sender};
use fst::{IntoStreamer, Set, Streamer};
use log::info;
use nanorand::{Rng, WyRand};
use parking_lot::{Mutex, RwLock};
use tinyset::{Set64, SetU32};
use ustr::Ustr;
use vec_map::VecMap;

struct Dictionary {
    all_words: Set<Vec<u8>>,
    preferred_words: Set<Vec<u8>>,
}

enum Tile {
    OneWord(usize),
    TwoWords(usize, usize),
}

pub struct Parameters {
    pub word_to_board: Vec<Vec<usize>>, //TODO make this a Word instead of a vec of usize
    // word_to_board2: Vec<Word>,
    board_to_word: VecMap<(usize, Option<usize>)>,
}

impl Parameters {
    pub fn new(words: Vec<Vec<usize>>) -> Self {
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
        }
    }
    fn vertical_words(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.word_to_board.len()).filter(move |i| Self::is_vertical(&self.word_to_board[*i]))
    }

    fn is_vertical(board_indexes: &[usize]) -> bool {
        let mut first = board_indexes[0];
        let mut second = board_indexes[1];
        second != first + 1
    }

    // fn board_indexes(&self, word_index: usize) -> impl Iterator<Item = usize> {
    //     self.word_to_board2[word_index].iter(self.width)
    // }

    // fn all_words(&self) -> impl Iterator<Item = impl Iterator<Item = usize>> + '_ {
    //     self.word_to_board2
    //         .iter()
    //         .map(move |word| word.iter(self.width))
    // }
}

struct Word {
    start: usize,
    length: usize,
    direction: Direction,
}

impl Word {
    fn iter(&self, board_width: usize) -> impl Iterator<Item = usize> {
        match self.direction {
            Direction::Across => (self.start..self.start + self.length).step_by(1),
            Direction::Down => {
                (self.start..self.start + self.length * board_width).step_by(board_width)
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Direction {
    Across,
    Down,
}

#[derive(Clone)]
enum State {
    NoSolution(SolverError),
    Solved,
    SelectWord,
    BackTracking,
}

#[derive(Debug, Clone)]
pub enum SolverError {
    NoPossibleWords,
    RuntimeError(String),
    NoSolutionExists,
    DuplicateWord,
}

#[derive(Debug)]
enum History {
    Begin,
    DeniedWords {
        word_index: usize,
        denied_word: Vec<Ustr>,
    },
    Minimized {
        board_index: usize,
        previous_state: Set64<char>,
    },
    SelectedWord {
        word_index: usize,
        word: Ustr,
    },
    CompletedWord {
        word: Ustr,
        word_index: usize,
    },
}

#[test]
fn test() {
    println!("{}", std::mem::size_of::<Set64<char>>());
}

pub struct Channel<T> {
    sender: Sender<T>,
    receiver: Receiver<T>,
}

impl<T> Channel<T> {
    pub fn bounded(cap: usize) -> Self {
        let (sender, receiver) = crossbeam::channel::bounded(cap);
        Self { sender, receiver }
    }

    pub fn unbounded() -> Self {
        let (sender, receiver) = crossbeam::channel::unbounded();
        Self { sender, receiver }
    }
}

pub struct Solver<'a> {
    parameters: &'a Parameters,
    history: Mutex<Vec<History>>,
    board: Board, // current characters permitted on the board for each board index
    attempted_words: VecMap<Vec<Ustr>>, // words currently attempted for each word index
    selected_words: HashSet<Ustr>, // words selected for each word index
    possible_words: VecMap<Vec<Ustr>>, // possible words remaining for each word index
    state: State,
    rand: WyRand,
    pool: TaskPool,
}

impl<'a> Solver<'a> {
    fn get_possible_words(parameters: &Parameters, dict: &Set<Vec<u8>>) -> VecMap<Vec<Ustr>> {
        let mut possible_words: VecMap<Vec<Ustr>> = VecMap::new();
        let mut word_lens: VecMap<Vec<usize>> = VecMap::new();

        for (word_index, word) in parameters.word_to_board.iter().enumerate() {
            word_lens
                .entry(word.len())
                .or_insert_with(Default::default)
                .push(word_index);
        }

        let mut stream = dict.range().into_stream();

        while let Some(word) = stream.next() {
            let str = match std::str::from_utf8(word) {
                Ok(str) => str,
                Err(_) => continue,
            };

            if let Some(word_indexes) = word_lens.get(str.chars().count()) {
                let ustr = Ustr::from(str);
                for word_index in word_indexes {
                    possible_words
                        .entry(*word_index)
                        .or_insert(Vec::new())
                        .push(ustr);
                }
            }
        }

        possible_words
    }
    pub fn new(parameters: &'a Parameters, dict: &Set<Vec<u8>>) -> Self {
        Self {
            parameters,
            history: Default::default(),
            board: Default::default(),
            attempted_words: Default::default(),
            selected_words: Default::default(),
            possible_words: Self::get_possible_words(parameters, dict),
            state: State::SelectWord,
            rand: WyRand::new(),
            pool: TaskPool::new(),
        }
    }

    pub fn new_seed(parameters: &'a Parameters, dict: &Set<Vec<u8>>, seed: u64) -> Self {
        Self {
            parameters,
            history: Default::default(),
            board: Default::default(),
            attempted_words: Default::default(),
            selected_words: Default::default(),
            possible_words: Self::get_possible_words(parameters, dict),
            state: State::SelectWord,
            rand: WyRand::new_seed(seed),
            pool: TaskPool::new(),
        }
    }
    pub fn run(mut self) -> Result<Board, SolverError> {
        let _ = self.history.get_mut().push(History::Begin);
        self.minimize(Set64::from_iter(self.parameters.vertical_words()));
        info!("starting board: \n{:?}", self.board);
        info!("parameters: {:?}", self.parameters.word_to_board);
        loop {
            // info!("Possible words: \n{:?}", self.possible_words);
            match self.state {
                State::SelectWord => self.select_word(),
                State::BackTracking => self.backtrack(),
                State::NoSolution(e) => return Err(e),
                State::Solved => return Ok(self.board),
            }
        }
    }

    // returns word index
    fn most_constrained(&self) -> Option<usize> {
        self.possible_words
            .iter()
            .filter(|(_, remaining)| !remaining.is_empty())
            .min_by_key(|(_, remaining_words)| remaining_words.len())
            .map(|(word_index, _)| word_index)
    }

    fn select_word(&mut self) {
        let word_index = match self.most_constrained() {
            Some(word_index) => word_index,
            None => {
                self.state = State::Solved;
                return;
            }
        };

        info!("Selecting word at word index: {}", word_index);

        info!(
            "Available words to select: {}",
            self.possible_words[word_index].len()
        );
        let chosen_word_index = self
            .rand
            .generate_range(0..self.possible_words[word_index].len());

        let chosen_word = self.possible_words[word_index].swap_remove(chosen_word_index);

        let _ = self.history.get_mut().push(History::SelectedWord {
            word_index,
            word: chosen_word,
        });

        info!("Chosen word: {}", chosen_word.as_str());

        let denied_words =
            std::mem::replace(&mut self.possible_words[word_index], vec![chosen_word]);

        if !denied_words.is_empty() {
            let _ = self.history.get_mut().push(History::DeniedWords {
                word_index,
                denied_word: denied_words,
            });
        }

        let mut words = Set64::<usize>::with_capacity(1);
        words.insert(word_index);
        self.minimize(words);
    }

    fn backtrack(&mut self) {
        info!("backtracking");
        while let Some(history) = self.history.get_mut().pop() {
            info!("{:?}", history);
            match history {
                History::Minimized {
                    board_index,
                    previous_state,
                } => {
                    // info!("reverting minimization for board index: {}", board_index);
                    self.board.board[board_index] = previous_state;
                }
                History::SelectedWord { word, word_index } => {
                    info!(
                        "Reverting randomly selected word for word index: {}",
                        word_index
                    );
                    let attempted_words =
                        self.attempted_words.entry(word_index).or_insert(Vec::new());
                    if self.possible_words[word_index].len() > 1 {
                        attempted_words.push(word);
                        let s = self.possible_words[word_index].swap_remove(0);
                        info!("removing {} from possible words for {}", s, word_index);

                        // info!("setting state to select word");
                        self.state = State::SelectWord;
                        break;
                    } else {
                        info!("No words remaining, backtracking");
                        self.possible_words[word_index].append(attempted_words);
                    }
                }
                History::Begin => {
                    self.state = State::NoSolution(SolverError::NoSolutionExists);
                    break;
                }
                History::CompletedWord { word, word_index } => {
                    info!(
                        "Undoing completed word: {} for word index: {}, adding back to possible words",
                        word, word_index
                    );
                    self.possible_words[word_index].push(word);
                    self.selected_words.remove(&word);
                }
                History::DeniedWords {
                    word_index,
                    mut denied_word,
                } => {
                    // info!("Undenying words for word index: {}", word_index);
                    self.possible_words[word_index].append(&mut denied_word);
                }
            }
        }
    }
    fn minimize(&mut self, mut words: Set64<usize>) {
        // let mut words = Set64::from_iter(std::iter::once(word_index));

        while !words.is_empty() {
            info!("Propogating board minimization to words: {:?}", words);
            words = match self.minimize_words(words, self.pool.clone()) {
                Ok(words) => words,
                Err(e) => {
                    info!("Minimize error: {:?}", e);
                    self.state = State::BackTracking;
                    break;
                }
            };
            // info!("New board: \n{:?}", self.board);
            // info!("Possible words: \n{:?}", self.possible_words);
        }
    }
    fn minimize_words(
        &mut self,
        word_indexes: Set64<usize>,
        pool: TaskPool,
    ) -> Result<Set64<usize>, SolverError> {
        if word_indexes.is_empty() {
            return Err(SolverError::RuntimeError(
                "Word indexes was empty when minimizing".to_owned(),
            ));
        }

        let results = pool.scope(|scope| {
            for (word_index, possible_words) in self
                .possible_words
                .iter_mut()
                .filter(|(word_index, _)| word_indexes.contains(*word_index))
            {
                let board_indexes = &self.parameters.word_to_board[word_index];
                let board = &self.board;
                let board_to_word = &self.parameters.board_to_word;
                let history = &self.history;
                scope.spawn(async move {
                    let mut words_to_propogate = Set64::<usize>::new();
                    let mut board_updates = Vec::new();
                    let mut denied_words = Vec::new();
                    // should be in parallel since word indexes must

                    let chars =
                        board.determine_chars(board_indexes, possible_words, &mut denied_words);

                    if !denied_words.is_empty() {
                        info!("denying words at index: {}", word_index);
                        let _ = history.lock().push(History::DeniedWords {
                            word_index,
                            denied_word: denied_words,
                        });
                    }

                    for (board_index, new_chars) in board_indexes.iter().copied().zip(chars) {
                        if new_chars.is_empty() {
                            // this can be done a better way
                            return (words_to_propogate, board_updates, word_index);
                        }

                        if board
                            .board
                            .get(board_index)
                            .map_or(true, |old_chars| *old_chars != new_chars)
                        {
                            board_updates.push((board_index, new_chars));
                            // self.board.board.insert(board_index, new_chars);
                            let _ = match board_to_word[board_index] {
                                (i, _) if i != word_index => words_to_propogate.insert(i),
                                (_, Some(i)) if i != word_index => words_to_propogate.insert(i),
                                _ => true,
                            };
                        }
                    }

                    (words_to_propogate, board_updates, word_index)
                });
            }
        });

        let mut next_word_indexes = None;
        let mut error = None;

        for (set, board_updates, word_index) in results {
            // let mut selected_word = Some(String::new());

            for (board_index, new_chars) in board_updates {
                // selected_word = selected_word.and_then(|mut s| {
                //     let mut new_chars_iter = new_chars.iter();
                //     s.push(new_chars_iter.next().unwrap());
                //     Some(s).filter(|_| new_chars_iter.next().is_none())
                // });

                match self.board.board.entry(board_index) {
                    vec_map::Entry::Vacant(entry) => {
                        entry.insert(new_chars);
                    }
                    vec_map::Entry::Occupied(mut entry) => {
                        let _ = self.history.get_mut().push(History::Minimized {
                            board_index,
                            previous_state: entry.insert(new_chars),
                        });
                    }
                }
            }

            if self.possible_words[word_index].len() == 1 {
                let added_word = self.possible_words[word_index].pop().unwrap();
                info!(
                    "Completed word: {} at word index: {}, removing from possible words",
                    added_word, word_index
                );
                if !self.selected_words.insert(added_word) {
                    error = Some(SolverError::DuplicateWord)
                } else {
                    let _ = self.history.get_mut().push(History::CompletedWord {
                        word: added_word,
                        word_index,
                    });
                }
            } else if self.possible_words[word_index].is_empty() {
                error = Some(SolverError::NoPossibleWords);
                continue;
            }

            // if let Some(selected_word) = selected_word {
            //     // self.history.push(History::AddedWord {  word_index });
            // }

            if let Some(mut previous) = next_word_indexes.replace(set) {
                next_word_indexes = next_word_indexes.map(|s| {
                    previous.extend(s);
                    previous
                });
            }
        }

        if let Some(e) = error {
            return Err(e);
        }

        next_word_indexes.ok_or_else(|| {
            SolverError::RuntimeError("Word indexes was empty when minimizing".to_owned())
        })
    }
}

#[derive(Default, Clone)]
pub struct Board {
    pub board: VecMap<Set64<char>>,
}

impl Board {
    fn determine_chars(
        &self,
        board_indexes: &[usize],
        possible_words: &mut Vec<Ustr>,
        denied_words: &mut Vec<Ustr>,
    ) -> Vec<Set64<char>> {
        let mut chars = vec![Set64::<char>::new(); board_indexes.len()];

        let mut i = 0;
        while i < possible_words.len() {
            let word = &possible_words[i];
            if word.chars().zip(board_indexes).all(|(char, board_index)| {
                self.board
                    .get(*board_index)
                    .map_or(true, |s| s.contains(char))
            }) {
                chars.iter_mut().zip(word.chars()).for_each(|(s, char)| {
                    let _ = s.insert(char);
                });
                i += 1;
            } else {
                let denied_word = possible_words.swap_remove(i);
                denied_words.push(denied_word);
            }
        }

        chars
    }
}

impl Debug for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for tile in &self.board {
            writeln!(f, "{}: {}", tile.0, tile.1.iter().collect::<String>())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determine_chars_works() {
        let board = Board::default();
        let ret = board.determine_chars(&[0, 1, 2], &mut vec![Ustr::from("tes")], &mut vec![]);

        for chars in ret {
            assert_eq!(chars.len(), 1)
        }
    }

    #[test]
    fn determine_chars_works2() {
        let board = Board::default();
        let ret = board.determine_chars(
            &[0, 1, 2],
            &mut vec![Ustr::from("tes"), Ustr::from("sat")],
            &mut vec![],
        );

        for chars in ret {
            assert_eq!(chars.len(), 2)
        }
    }
}
