use std::{collections::HashMap, fmt::Debug, iter::FromIterator};

use bevy_tasks::TaskPool;
use byte_set::ByteSet;
use fst::{IntoStreamer, Set, Streamer};
use nanorand::{Rng, WyRand};
use parking_lot::Mutex;
use tinyset::Set64;
use ustr::Ustr;
use vec_map::VecMap;

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
        let first = board_indexes[0];
        let second = board_indexes[1];
        second != first + 1
    }
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
        denied_words: Vec<Ustr>,
    },
    Minimized {
        board_index: usize,
        previous_state: ByteSet,
    },
    SelectedWord {
        word_index: usize,
    },
    CompletedWord {
        word: Ustr,
        word_index: usize,
    },
}

pub struct Solver<'a> {
    parameters: &'a Parameters,
    history: Mutex<Vec<History>>,
    board: Board, // current characters permitted on the board for each board index
    attempted_words: VecMap<Vec<Ustr>>, // words currently attempted for each word index
    selected_words: HashMap<Ustr, usize>, // words selected for each word index
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
    pub fn run(mut self) -> Result<HashMap<Ustr, usize>, SolverError> {
        let _ = self.history.get_mut().push(History::Begin);
        self.minimize(Set64::from_iter(self.parameters.vertical_words()));
        loop {
            match self.state {
                State::SelectWord => self.select_word(),
                State::BackTracking => self.backtrack(),
                State::NoSolution(e) => return Err(e),
                State::Solved => {
                    return Ok(self.selected_words);
                }
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
        let chosen_word_index = self
            .rand
            .generate_range(0..self.possible_words[word_index].len());

        let chosen_word = self.possible_words[word_index].swap_remove(chosen_word_index);

        self.history
            .get_mut()
            .push(History::SelectedWord { word_index });

        let denied_words =
            std::mem::replace(&mut self.possible_words[word_index], vec![chosen_word]);

        if !denied_words.is_empty() {
            self.history.get_mut().push(History::DeniedWords {
                word_index,
                denied_words,
            });
        }

        let mut words = Set64::<usize>::with_capacity(1);
        words.insert(word_index);
        self.minimize(words);
    }

    fn backtrack(&mut self) {
        while let Some(history) = self.history.get_mut().pop() {
            match history {
                History::Minimized {
                    board_index,
                    previous_state,
                } => {
                    self.board.board[board_index] = previous_state;
                }
                History::SelectedWord { word_index } => {
                    let attempted_words =
                        self.attempted_words.entry(word_index).or_insert(Vec::new());
                    if self.possible_words[word_index].len() > 1 {
                        // selected word is always at index 0 of possible words
                        let s = self.possible_words[word_index].swap_remove(0);
                        attempted_words.push(s);

                        self.state = State::SelectWord;
                        break;
                    } else {
                        self.possible_words[word_index].append(attempted_words);
                    }
                }
                History::Begin => {
                    self.state = State::NoSolution(SolverError::NoSolutionExists);
                    break;
                }
                History::CompletedWord { word, word_index } => {
                    self.possible_words[word_index].push(word);
                    self.selected_words.remove(&word);
                }
                History::DeniedWords {
                    word_index,
                    mut denied_words,
                } => {
                    self.possible_words[word_index].append(&mut denied_words);
                }
            }
        }
    }
    fn minimize(&mut self, mut words: Set64<usize>) {
        while !words.is_empty() {
            words = match self.minimize_words(words, self.pool.clone()) {
                Ok(words) => words,
                Err(_e) => {
                    self.state = State::BackTracking;
                    break;
                }
            };
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
                    let mut words_to_propagate = Set64::<usize>::new();
                    let mut board_updates = Vec::new();
                    // should be in parallel since word indexes must

                    let WordExploreResult {
                        allowed_chars,
                        denied_words,
                    } = board.determine_chars(board_indexes, possible_words);

                    if !denied_words.is_empty() {
                        let _ = history.lock().push(History::DeniedWords {
                            word_index,
                            denied_words,
                        });
                    }

                    for (board_index, new_chars) in board_indexes.iter().copied().zip(allowed_chars)
                    {
                        if new_chars.is_empty() {
                            // this can be done a better way
                            return (words_to_propagate, board_updates, word_index);
                        }

                        if board
                            .board
                            .get(board_index)
                            .map_or(true, |old_chars| *old_chars != new_chars)
                        {
                            board_updates.push((board_index, new_chars));
                            // self.board.board.insert(board_index, new_chars);
                            let _ = match board_to_word[board_index] {
                                (i, _) if i != word_index => words_to_propagate.insert(i),
                                (_, Some(i)) if i != word_index => words_to_propagate.insert(i),
                                _ => true,
                            };
                        }
                    }

                    (words_to_propagate, board_updates, word_index)
                });
            }
        });

        let mut next_word_indexes = None;
        let mut error = None;

        for (set, board_updates, word_index) in results {
            for (board_index, previous_state) in self.board.apply(board_updates.into_iter()) {
                let _ = self.history.get_mut().push(History::Minimized {
                    board_index,
                    previous_state,
                });
            }

            if self.possible_words[word_index].len() == 1 {
                let added_word = self.possible_words[word_index].pop().unwrap();
                match self.selected_words.entry(added_word) {
                    std::collections::hash_map::Entry::Occupied(_) => {
                        self.possible_words[word_index].push(added_word);
                        error = Some(SolverError::DuplicateWord);
                    }
                    std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                        vacant_entry.insert(word_index);
                        self.history.get_mut().push(History::CompletedWord {
                            word: added_word,
                            word_index,
                        });
                    }
                };
            } else if self.possible_words[word_index].is_empty() {
                error = Some(SolverError::NoPossibleWords);
                continue;
            }

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
    pub board: VecMap<ByteSet>,
}

impl Board {
    fn apply<'a: 'i, 'i, I: 'i + Iterator<Item = (usize, ByteSet)>>(
        &'a mut self,
        i: I,
    ) -> impl Iterator<Item = (usize, ByteSet)> + 'i {
        i.filter_map(move |(i, charset)| match self.board.entry(i) {
            vec_map::Entry::Vacant(e) => {
                e.insert(charset);
                None
            }
            vec_map::Entry::Occupied(mut e) => Some((i, e.insert(charset))),
        })
    }
    fn determine_chars(
        &self,
        board_indexes: &[usize],
        possible_words: &mut Vec<Ustr>,
    ) -> WordExploreResult {
        let mut allowed_chars = vec![ByteSet::new(); board_indexes.len()];
        let mut denied_words = Vec::new();

        let mut i = 0;
        while i < possible_words.len() {
            let word = &possible_words[i];
            if word.bytes().zip(board_indexes).all(|(char, board_index)| {
                self.board
                    .get(*board_index)
                    .map_or(true, |s| s.contains(char))
            }) {
                allowed_chars
                    .iter_mut()
                    .zip(word.bytes())
                    .for_each(|(s, char)| {
                        let _ = s.insert(char);
                    });
                i += 1;
            } else {
                let denied_word = possible_words.swap_remove(i);
                denied_words.push(denied_word);
            }
        }

        WordExploreResult {
            allowed_chars,
            denied_words,
        }
    }
}

struct WordExploreResult {
    allowed_chars: Vec<ByteSet>,
    denied_words: Vec<Ustr>,
}

impl Debug for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for tile in &self.board {
            writeln!(
                f,
                "{}: {}",
                tile.0,
                tile.1.into_iter().map(|b| b as char).collect::<String>()
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use bevy_tasks::TaskPoolBuilder;

    use super::*;

    #[test]
    fn determine_chars_works() {
        let board = Board::default();
        let WordExploreResult {
            allowed_chars,
            denied_words,
        } = board.determine_chars(&[0, 1, 2], &mut vec![Ustr::from("tes")]);

        for chars in allowed_chars {
            assert_eq!(chars.len(), 1)
        }
        assert_eq!(denied_words.len(), 0);
    }

    #[test]
    fn determine_chars_works2() {
        let mut board = Board::default();
        let WordExploreResult {
            allowed_chars,
            denied_words: _,
        } = board.determine_chars(&[0, 1, 2], &mut vec![Ustr::from("tes"), Ustr::from("sat")]);

        assert_eq!(allowed_chars[0], ByteSet::from_iter(&[b't', b's']));
        assert_eq!(allowed_chars[1], ByteSet::from_iter(&[b'e', b'a']));
        assert_eq!(allowed_chars[2], ByteSet::from_iter(&[b's', b't']));

        board.apply(allowed_chars.into_iter().enumerate()).count();

        let WordExploreResult {
            allowed_chars: _,
            denied_words,
        } = board.determine_chars(
            &[0, 1, 2],
            &mut vec![Ustr::from("tes"), Ustr::from("sat"), Ustr::from("den")],
        );

        assert_eq!(denied_words[0], Ustr::from("den"));
    }

    #[async_std::test]
    async fn async_test() {
        let task_pool = TaskPoolBuilder::new().num_threads(10).build();

        let task1 = task_pool.spawn(async {
            println!("spawned 1st task");
            std::thread::sleep(Duration::from_secs(1));
            println!("wasted 1 second on 1st task");
            async_std::task::sleep(Duration::from_secs(1)).await;
            println!("slept for 1 second");
        });

        let task2 = task_pool.spawn(async {
            println!("spawned 2nd task");
            async_std::task::sleep(Duration::from_secs(2)).await;
            println!("slept for 2 second on 2nd task");
        });

        let task3 = task_pool.spawn(async {
            println!("spawned 3rd task");
            async_std::task::sleep(Duration::from_secs(3)).await;
            println!("slept for 3 second on third task");
        });

        let task4 = task_pool.spawn(async {
            println!("spawned 4th task");
            async_std::task::sleep(Duration::from_secs(1)).await;
            println!("slept for 1 second on fourth task");
        });

        let task5 = task_pool.spawn(async {
            println!("spawned 5th task");
            async_std::task::sleep(Duration::from_secs(1)).await;
            println!("slept for 1 second on fifth task");
        });

        let task6 = task_pool.spawn(async {
            println!("spawned 6th task");
        });

        let task7 = task_pool.spawn(async {
            println!("spawned 7th task");
            std::thread::sleep(Duration::from_secs(1));
            println!("wasted 1 second on 7th task");
        });

        let task8 = task_pool.spawn(async {
            println!("spawned 8th task");
            std::thread::sleep(Duration::from_secs(1));
            println!("wasted 1 second on 8th task");
        });

        let task9 = task_pool.spawn(async {
            println!("spawned 9th task");
            std::thread::sleep(Duration::from_secs(1));
            println!("wasted 1 second on 9th task");
        });

        futures::join!(task1, task2, task3, task4, task5, task6, task7, task8, task9);
    }
}
